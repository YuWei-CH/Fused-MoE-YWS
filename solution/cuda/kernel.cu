#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ffi = tvm::ffi;

namespace
{

  // ---------------------------------------------------------------------------
  // Global geometry and fixed submission-time tuning choices.
  // ---------------------------------------------------------------------------

  // Fixed DeepSeek-V3 / contest geometry from the reference definition.
  constexpr int64_t kHiddenSize = 7168;
  constexpr int64_t kIntermediateSize = 2048;
  constexpr int64_t kBlock = 128;
  constexpr int64_t kGlobalExperts = 256;
  constexpr int64_t kLocalExperts = 32;
  constexpr int64_t kTopK = 8;
  constexpr int64_t kNumGroups = 8;
  constexpr int64_t kTopKGroups = 4;
  constexpr int64_t kGroupSize = kGlobalExperts / kNumGroups;
  constexpr int kGemm1TileM = 16;
  constexpr int kGemm1TileN = 32;
  constexpr int kGemm1TileK = 64;
  constexpr int kGemm2TileM = 16;
  constexpr int kGemm2TileN = 32;
  constexpr int kGemm2TileK = 64;
  constexpr int64_t kHybridDispatchSeqLenThreshold = 4096;
  constexpr int kGroupedGemm1Threshold = 32;
  constexpr int kGroupedGemm2Threshold = 32;

#define CHECK_CUDA(expr)                                                             \
  do                                                                                 \
  {                                                                                  \
    cudaError_t err__ = (expr);                                                      \
    if (err__ != cudaSuccess)                                                        \
    {                                                                                \
      TVM_FFI_THROW(RuntimeError) << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                                  << ": " << cudaGetErrorString(err__);              \
    }                                                                                \
  } while (0)

  inline void CheckCutlassStatus(cutlass::Status status, const char *context)
  {
    if (status != cutlass::Status::kSuccess)
    {
      TVM_FFI_THROW(RuntimeError) << "CUTLASS error in " << context << ": "
                                  << cutlassGetStatusString(status);
    }
  }

  namespace cutlass_grouped_f32
  {

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<
        typename cutlass::gemm::kernel::DefaultGemmGrouped<
            ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4, ElementB,
            cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4, ElementC,
            cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<
                ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator,
                ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4>::GemmKernel>;

  } // namespace cutlass_grouped_f32

  // ---------------------------------------------------------------------------
  // Validation and lightweight CUDA memory helpers.
  // ---------------------------------------------------------------------------

  void CheckTensor(const ffi::TensorView &tensor, const char *name, int32_t ndim,
                   ffi::ShapeView expected_shape, DLDataType expected_dtype)
  {
    if (tensor.device().device_type != kDLCUDA)
    {
      TVM_FFI_THROW(ValueError) << name << " must be a CUDA tensor";
    }
    if (!tensor.IsContiguous())
    {
      TVM_FFI_THROW(ValueError) << name << " must be contiguous";
    }
    if (tensor.ndim() != ndim)
    {
      TVM_FFI_THROW(ValueError) << name << " must be " << ndim << "D, got " << tensor.ndim();
    }
    for (int32_t i = 0; i < ndim; ++i)
    {
      if (tensor.size(i) != expected_shape[i])
      {
        TVM_FFI_THROW(ValueError) << name << " shape mismatch at dim " << i << ": expected "
                                  << expected_shape[i] << ", got " << tensor.size(i);
      }
    }
    DLDataType dtype = tensor.dtype();
    if (dtype.code != expected_dtype.code || dtype.bits != expected_dtype.bits ||
        dtype.lanes != expected_dtype.lanes)
    {
      TVM_FFI_THROW(TypeError) << name << " dtype mismatch: expected "
                               << ffi::DLDataTypeToString(expected_dtype) << ", got "
                               << ffi::DLDataTypeToString(dtype);
    }
  }

  void CheckScalarVector(const ffi::TensorView &tensor, const char *name, int64_t size,
                         DLDataType expected_dtype)
  {
    CheckTensor(tensor, name, 1, ffi::ShapeView({size}), expected_dtype);
  }

  void CheckSameDevice(const ffi::TensorView &a, const ffi::TensorView &b, const char *a_name,
                       const char *b_name)
  {
    if (a.device().device_type != b.device().device_type ||
        a.device().device_id != b.device().device_id)
    {
      TVM_FFI_THROW(ValueError) << a_name << " and " << b_name << " must be on the same device";
    }
  }

  template <typename T>
  class AsyncBuffer
  {
  public:
    AsyncBuffer() = default;

    AsyncBuffer(size_t count, cudaStream_t stream) : count_(count), stream_(stream)
    {
      if (count_ == 0)
      {
        return;
      }
      CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_, stream_));
    }

    ~AsyncBuffer()
    {
      if (ptr_ != nullptr)
      {
        cudaError_t err = cudaFreeAsync(ptr_, stream_);
        (void)err;
      }
    }

    AsyncBuffer(const AsyncBuffer &) = delete;
    AsyncBuffer &operator=(const AsyncBuffer &) = delete;

    AsyncBuffer(AsyncBuffer &&other) noexcept
        : ptr_(other.ptr_), count_(other.count_), stream_(other.stream_)
    {
      other.ptr_ = nullptr;
      other.count_ = 0;
      other.stream_ = nullptr;
    }

    AsyncBuffer &operator=(AsyncBuffer &&other) noexcept
    {
      if (this != &other)
      {
        if (ptr_ != nullptr)
        {
          cudaError_t err = cudaFreeAsync(ptr_, stream_);
          (void)err;
        }
        ptr_ = other.ptr_;
        count_ = other.count_;
        stream_ = other.stream_;
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.stream_ = nullptr;
      }
      return *this;
    }

    T *get() const { return ptr_; }
    size_t size() const { return count_; }

  private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
    cudaStream_t stream_ = nullptr;
  };

  template <typename T>
  constexpr T AlignUp(T value, T alignment)
  {
    return (value + alignment - 1) / alignment * alignment;
  }

  // A tiny bump allocator over one cudaMallocAsync-backed buffer. This keeps per-call
  // workspace management readable without committing to a large global workspace API.
  class WorkspaceArena
  {
  public:
    WorkspaceArena(size_t bytes, cudaStream_t stream) : storage_(bytes, stream) {}

    template <typename T>
    T *Alloc(size_t count)
    {
      size_t aligned_offset = AlignUp(offset_bytes_, static_cast<size_t>(alignof(T)));
      size_t bytes = sizeof(T) * count;
      if (aligned_offset + bytes > storage_.size())
      {
        TVM_FFI_THROW(RuntimeError) << "WorkspaceArena overflow: requested " << bytes
                                    << " bytes with " << (storage_.size() - offset_bytes_)
                                    << " bytes remaining";
      }
      T *ptr = reinterpret_cast<T *>(storage_.get() + aligned_offset);
      offset_bytes_ = aligned_offset + bytes;
      return ptr;
    }

  private:
    AsyncBuffer<uint8_t> storage_;
    size_t offset_bytes_ = 0;
  };

  struct PreparedExpertWorkload
  {
    int32_t *expert_counts = nullptr;
    int32_t *expert_offsets = nullptr;
    int32_t *reordered_tokens = nullptr;
    int32_t *reordered_local_experts = nullptr;
    int32_t *total_rows = nullptr;
  };

  struct GroupedExpertPlan
  {
    std::vector<int32_t> experts;
    std::vector<int32_t> slots;
  };

  __device__ inline float fp8_to_float(__nv_fp8_e4m3 value)
  {
    return static_cast<float>(value);
  }

  __device__ inline float bf16_to_float(__nv_bfloat16 value)
  {
    return __bfloat162float(value);
  }

  // ---------------------------------------------------------------------------
  // Stage 1: routing.
  // These kernels implement the reference DeepSeek routing logic:
  // logits -> sigmoid/bias -> grouped top-k selection -> normalized expert weights.
  // ---------------------------------------------------------------------------

  // Compute s = sigmoid(logits) and s_with_bias = s + bias.
  __global__ void sigmoid_bias_kernel(const float *routing_logits, const __nv_bfloat16 *routing_bias,
                                      float *s, float *s_with_bias, int64_t t)
  {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = t * kGlobalExperts;
    if (idx >= total)
    {
      return;
    }
    int64_t expert = idx % kGlobalExperts;
    float logit = routing_logits[idx];
    float sigmoid = 1.0f / (1.0f + expf(-logit));
    s[idx] = sigmoid;
    s_with_bias[idx] = sigmoid + bf16_to_float(routing_bias[expert]);
  }

  // One CUDA block per token. This mirrors the reference routing:
  // top-2 per group -> top-4 groups -> final top-8 experts -> normalized weights from s.
  __global__ void routing_select_kernel(const float *s, const float *s_with_bias, int32_t *topk_idx,
                                        float *weights, int64_t t, float routed_scaling_factor)
  {
    int64_t token = blockIdx.x;
    if (token >= t || threadIdx.x != 0)
    {
      return;
    }

    const float *s_row = s + token * kGlobalExperts;
    const float *s_with_bias_row = s_with_bias + token * kGlobalExperts;
    int32_t *topk_row = topk_idx + token * kTopK;
    float *weight_row = weights + token * kGlobalExperts;

    float group_scores[kNumGroups];
    for (int group = 0; group < kNumGroups; ++group)
    {
      float best0 = -INFINITY;
      float best1 = -INFINITY;
      for (int lane = 0; lane < kGroupSize; ++lane)
      {
        float value = s_with_bias_row[group * kGroupSize + lane];
        if (value > best0)
        {
          best1 = best0;
          best0 = value;
        }
        else if (value > best1)
        {
          best1 = value;
        }
      }
      group_scores[group] = best0 + best1;
    }

    bool keep_group[kNumGroups] = {false};
    for (int pick = 0; pick < kTopKGroups; ++pick)
    {
      int best_group = -1;
      float best_score = -INFINITY;
      for (int group = 0; group < kNumGroups; ++group)
      {
        if (!keep_group[group] && group_scores[group] > best_score)
        {
          best_score = group_scores[group];
          best_group = group;
        }
      }
      keep_group[best_group] = true;
    }

    int32_t selected_idx[kTopK];
    float selected_score[kTopK];
    for (int k = 0; k < kTopK; ++k)
    {
      selected_idx[k] = -1;
      selected_score[k] = -INFINITY;
    }

    for (int expert = 0; expert < kGlobalExperts; ++expert)
    {
      if (!keep_group[expert / kGroupSize])
      {
        continue;
      }
      float value = s_with_bias_row[expert];
      int insert_at = -1;
      for (int k = 0; k < kTopK; ++k)
      {
        if (value > selected_score[k])
        {
          insert_at = k;
          break;
        }
      }
      if (insert_at == -1)
      {
        continue;
      }
      for (int k = kTopK - 1; k > insert_at; --k)
      {
        selected_score[k] = selected_score[k - 1];
        selected_idx[k] = selected_idx[k - 1];
      }
      selected_score[insert_at] = value;
      selected_idx[insert_at] = expert;
    }

    for (int expert = 0; expert < kGlobalExperts; ++expert)
    {
      weight_row[expert] = 0.0f;
    }

    float weight_sum = 0.0f;
    for (int k = 0; k < kTopK; ++k)
    {
      topk_row[k] = selected_idx[k];
      if (selected_idx[k] >= 0)
      {
        weight_sum += s_row[selected_idx[k]];
      }
    }
    weight_sum += 1e-20f;

    for (int k = 0; k < kTopK; ++k)
    {
      int32_t expert = selected_idx[k];
      if (expert >= 0)
      {
        weight_row[expert] = (s_row[expert] / weight_sum) * routed_scaling_factor;
      }
    }
  }

  // Final cast back to the contest output dtype.
  __global__ void cast_output_kernel(const float *output_fp32, __nv_bfloat16 *output_bf16,
                                     int64_t total)
  {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
      return;
    }
    output_bf16[idx] = __float2bfloat16(output_fp32[idx]);
  }

  __global__ void count_expert_tokens_kernel(const int32_t *topk_idx, int32_t *expert_counts,
                                             int64_t t, int32_t local_expert_offset)
  {
    // Count how many routed rows belong to each local expert on this rank.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= t * kTopK)
    {
      return;
    }
    int32_t global_expert = topk_idx[idx];
    int32_t local_expert = global_expert - local_expert_offset;
    if (local_expert >= 0 && local_expert < kLocalExperts)
    {
      atomicAdd(&expert_counts[local_expert], 1);
    }
  }

  __global__ void reorder_tokens_kernel(const int32_t *topk_idx, const int32_t *expert_offsets,
                                        int32_t *expert_current_counts, int32_t *reordered_tokens,
                                        int32_t *reordered_local_experts, int64_t t,
                                        int32_t local_expert_offset)
  {
    // Expand each token's top-k routing decisions into a contiguous expert-major row layout.
    int64_t token_id = blockIdx.x;
    if (token_id >= t)
    {
      return;
    }
    for (int k = 0; k < kTopK; ++k)
    {
      int32_t global_expert = topk_idx[token_id * kTopK + k];
      int32_t local_expert = global_expert - local_expert_offset;
      if (local_expert >= 0 && local_expert < kLocalExperts)
      {
        int32_t slot = atomicAdd(&expert_current_counts[local_expert], 1);
        int32_t pos = expert_offsets[local_expert] + slot;
        reordered_tokens[pos] = static_cast<int32_t>(token_id);
        reordered_local_experts[pos] = local_expert;
      }
    }
  }

  __global__ void finalize_expert_offsets_kernel(const int32_t *expert_counts, int32_t *expert_offsets,
                                                 int32_t *total_reordered)
  {
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
      return;
    }
    int32_t total = expert_offsets[kLocalExperts - 1] + expert_counts[kLocalExperts - 1];
    expert_offsets[kLocalExperts] = total;
    total_reordered[0] = total;
  }

  size_t GetExpertScanWorkspaceSize(cudaStream_t stream)
  {
    size_t scan_workspace_size = 0;
    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, scan_workspace_size,
                                             static_cast<int32_t *>(nullptr),
                                             static_cast<int32_t *>(nullptr), kLocalExperts,
                                             stream));
    return scan_workspace_size;
  }

  size_t GetDeviceOnlyWorkspaceBytes(int32_t max_rows, size_t scan_workspace_size)
  {
    return sizeof(int32_t) *
               static_cast<size_t>(3 * kLocalExperts + (kLocalExperts + 1) + 1 + 2 * max_rows) +
           scan_workspace_size +
           sizeof(float) * static_cast<size_t>(max_rows) *
               (2 * kHiddenSize + 3 * kIntermediateSize);
  }

  size_t GetGroupedPrepWorkspaceBytes(int32_t max_rows, size_t scan_workspace_size)
  {
    return sizeof(int32_t) *
               static_cast<size_t>(3 * kLocalExperts + (kLocalExperts + 1) + 1 + 2 * max_rows) +
           scan_workspace_size;
  }

  size_t GetGroupedComputeWorkspaceBytes(int32_t total_rows, size_t grouped_gemm1_count,
                                         size_t grouped_gemm2_count)
  {
    return sizeof(float) * static_cast<size_t>(total_rows) *
               (2 * kIntermediateSize + kIntermediateSize + kHiddenSize + kHiddenSize) +
           sizeof(float) * grouped_gemm1_count * 2 * kIntermediateSize * kHiddenSize +
           sizeof(float) * grouped_gemm2_count * kHiddenSize * kIntermediateSize +
           sizeof(float) * (2 * kIntermediateSize * kHiddenSize +
                            kHiddenSize * kIntermediateSize) +
           sizeof(int32_t) * (grouped_gemm1_count + grouped_gemm2_count);
  }

  GroupedExpertPlan BuildGroupedExpertPlan(const std::vector<int32_t> &expert_counts,
                                           int32_t threshold)
  {
    GroupedExpertPlan plan;
    plan.slots.assign(kLocalExperts, -1);
    for (int32_t expert = 0; expert < kLocalExperts; ++expert)
    {
      if (expert_counts[expert] >= threshold)
      {
        plan.slots[expert] = static_cast<int32_t>(plan.experts.size());
        plan.experts.push_back(expert);
      }
    }
    return plan;
  }

  // ---------------------------------------------------------------------------
  // Stage 2: build the local expert workload.
  // Convert token-level routing outputs into an expert-major row layout that both
  // execution paths consume.
  // ---------------------------------------------------------------------------

  // Build the expert-local workload layout shared by both execution paths:
  // 1) count local assignments
  // 2) exclusive-scan counts into per-expert row offsets
  // 3) reorder (token, expert) pairs into one contiguous row-major expert layout
  PreparedExpertWorkload PrepareExpertWorkload(const int32_t *topk_idx, int64_t t,
                                               int32_t local_expert_offset, int32_t max_rows,
                                               size_t scan_workspace_size,
                                               WorkspaceArena *workspace, cudaStream_t stream)
  {
    PreparedExpertWorkload prepared;
    prepared.expert_counts = workspace->Alloc<int32_t>(kLocalExperts);
    prepared.expert_offsets = workspace->Alloc<int32_t>(kLocalExperts + 1);
    prepared.reordered_tokens = workspace->Alloc<int32_t>(max_rows);
    prepared.reordered_local_experts = workspace->Alloc<int32_t>(max_rows);
    int32_t *tmp_counts = workspace->Alloc<int32_t>(kLocalExperts);
    prepared.total_rows = workspace->Alloc<int32_t>(1);
    uint8_t *scan_workspace = workspace->Alloc<uint8_t>(scan_workspace_size);

    CHECK_CUDA(cudaMemsetAsync(prepared.expert_counts, 0, kLocalExperts * sizeof(int32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(tmp_counts, 0, kLocalExperts * sizeof(int32_t), stream));

    count_expert_tokens_kernel<<<(t * kTopK + 255) / 256, 256, 0, stream>>>(
        topk_idx, prepared.expert_counts, t, local_expert_offset);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(scan_workspace, scan_workspace_size,
                                             prepared.expert_counts, prepared.expert_offsets,
                                             kLocalExperts, stream));
    finalize_expert_offsets_kernel<<<1, 1, 0, stream>>>(prepared.expert_counts,
                                                        prepared.expert_offsets,
                                                        prepared.total_rows);
    CHECK_CUDA(cudaGetLastError());

    reorder_tokens_kernel<<<t, 1, 0, stream>>>(topk_idx, prepared.expert_offsets, tmp_counts,
                                               prepared.reordered_tokens,
                                               prepared.reordered_local_experts, t,
                                               local_expert_offset);
    CHECK_CUDA(cudaGetLastError());

    return prepared;
  }

  // ---------------------------------------------------------------------------
  // Stage 3: materialize activations and merge results.
  // These kernels transform reordered token rows into dense FP32 activations,
  // apply the SwiGLU nonlinearity, and scatter expert outputs back to tokens.
  // ---------------------------------------------------------------------------

  __global__ void fused_gather_dequant_f32_kernel(const __nv_fp8_e4m3 *src,
                                                  const int32_t *token_idx,
                                                  const float *scale_src, float *dst, int64_t m,
                                                  int64_t scale_t)
  {
    // Gather hidden-state rows according to the reordered expert-major layout and
    // dequantize them from FP8 block-scaled storage into FP32 GEMM inputs.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * kHiddenSize)
    {
      return;
    }
    int64_t row = idx / kHiddenSize;
    int64_t col = idx % kHiddenSize;
    int32_t token = token_idx[row];
    if (token < 0)
    {
      dst[idx] = 0.0f;
      return;
    }
    float scale = scale_src[static_cast<int64_t>(col / kBlock) * scale_t + token];
    dst[idx] = fp8_to_float(src[static_cast<int64_t>(token) * kHiddenSize + col]) * scale;
  }

  __global__ void swiglu_f32_kernel(const float *g1, float *c, const int32_t *total_rows_ptr)
  {
    // Apply the gate/up projection epilogue: swiglu(x1, x2) = x1 * silu(x2).
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total_rows = total_rows_ptr[0];
    if (idx >= static_cast<int64_t>(total_rows) * kIntermediateSize)
    {
      return;
    }
    int64_t row = idx / kIntermediateSize;
    int64_t col = idx % kIntermediateSize;
    float x1 = g1[row * (2 * kIntermediateSize) + col];
    float x2 = g1[row * (2 * kIntermediateSize) + kIntermediateSize + col];
    c[idx] = x1 * (x2 / (1.0f + expf(-x2)));
  }

  __global__ void optimized_scatter_add_kernel(const float *o, const int32_t *token_idx,
                                               const int32_t *local_expert_idx,
                                               const float *weights, float *output,
                                               const int32_t *total_rows_ptr,
                                               int32_t local_expert_offset)
  {
    // Accumulate expert outputs back into the original token order with routing weights.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total_reordered = total_rows_ptr[0];
    if (idx >= static_cast<int64_t>(total_reordered) * kHiddenSize)
    {
      return;
    }
    int64_t row = idx / kHiddenSize;
    int32_t token = token_idx[row];
    int32_t local_expert = local_expert_idx[row];
    float weight =
        weights[static_cast<int64_t>(token) * kGlobalExperts + local_expert + local_expert_offset];
    atomicAdd(&output[static_cast<int64_t>(token) * kHiddenSize + (idx % kHiddenSize)],
              o[idx] * weight);
  }

  // ---------------------------------------------------------------------------
  // Stage 4: expert compute backends.
  // The device_only path consumes FP8 weights directly inside custom kernels.
  // The grouped_workload path dequantizes selected experts into FP32 and then
  // dispatches either a small SIMT GEMM or a grouped CUTLASS GEMM.
  // ---------------------------------------------------------------------------

  __global__ void dequant_weights_to_f32_kernel(const __nv_fp8_e4m3 *src, const float *scale,
                                                float *dst, int64_t num_experts, int64_t n,
                                                int64_t k)
  {
    // Dequantize a contiguous set of experts from FP8 block-scaled weights into
    // row-major FP32 weights for the grouped_workload path.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t expert_size = n * k;
    if (idx >= num_experts * expert_size)
    {
      return;
    }
    int64_t expert = idx / expert_size;
    int64_t offset = idx % expert_size;
    int64_t row = offset / k;
    int64_t col = offset % k;
    int64_t n_blocks = n / kBlock;
    int64_t k_blocks = k / kBlock;
    int64_t scale_idx =
        expert * n_blocks * k_blocks + (row / kBlock) * k_blocks + (col / kBlock);
    dst[idx] = fp8_to_float(src[idx]) * scale[scale_idx];
  }

  __global__ void dequant_selected_experts_to_f32_kernel(const int32_t *expert_ids,
                                                         const __nv_fp8_e4m3 *src,
                                                         const float *scale, float *dst,
                                                         int32_t num_selected, int64_t n,
                                                         int64_t k)
  {
    // Dequantize an arbitrary list of experts into a compact FP32 buffer used by
    // the grouped GEMM path. expert_ids[selected_slot] defines which expert each
    // destination slice corresponds to.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t expert_size = n * k;
    if (idx >= static_cast<int64_t>(num_selected) * expert_size)
    {
      return;
    }
    int32_t selected_slot = static_cast<int32_t>(idx / expert_size);
    int32_t local_expert = expert_ids[selected_slot];
    int64_t offset = idx % expert_size;
    int64_t row = offset / k;
    int64_t col = offset % k;
    int64_t n_blocks = n / kBlock;
    int64_t k_blocks = k / kBlock;
    int64_t scale_idx = static_cast<int64_t>(local_expert) * n_blocks * k_blocks +
                        (row / kBlock) * k_blocks + (col / kBlock);
    dst[idx] = fp8_to_float(src[static_cast<int64_t>(local_expert) * expert_size + offset]) *
               scale[scale_idx];
  }

  template <int TileM, int TileN, int TileK>
  __global__ void gemm1_grouped_variable_fp8_kernel(
      const float *__restrict__ a, const int32_t *__restrict__ expert_offsets,
      const int32_t *__restrict__ expert_counts,
      const __nv_fp8_e4m3 *__restrict__ gemm1_weights,
      const float *__restrict__ gemm1_weights_scale, float *__restrict__ g1)
  {
    // Each block computes one [TileM x TileN] output tile for one local expert.
    // We stage A and the dequantized weight tile in shared memory and accumulate in FP32.
    __shared__ float a_tile[TileM][TileK];
    __shared__ float b_tile[TileK][TileN + 1];

    int32_t local_expert = blockIdx.y;
    int32_t rows = expert_counts[local_expert];
    if (rows <= 0)
    {
      return;
    }

    int32_t row_offset = expert_offsets[local_expert];
    const float *a_expert = a + static_cast<int64_t>(row_offset) * kHiddenSize;
    float *g1_expert = g1 + static_cast<int64_t>(row_offset) * (2 * kIntermediateSize);
    const __nv_fp8_e4m3 *w13_fp8 =
        gemm1_weights + static_cast<int64_t>(local_expert) * (2 * kIntermediateSize) * kHiddenSize;
    const float *w13_scale = gemm1_weights_scale +
                             static_cast<int64_t>(local_expert) * ((2 * kIntermediateSize) / kBlock) *
                                 (kHiddenSize / kBlock);

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int tid = local_row * blockDim.x + local_col;
    int col = blockIdx.x * TileN + local_col;

    for (int32_t row0 = 0; row0 < rows; row0 += TileM)
    {
      int row = row0 + local_row;
      float acc = 0.0f;

      for (int64_t k0 = 0; k0 < kHiddenSize; k0 += TileK)
      {
        for (int idx = tid; idx < TileM * TileK; idx += blockDim.x * blockDim.y)
        {
          int tile_row = idx / TileK;
          int tile_k = idx % TileK;
          int global_row = row0 + tile_row;
          int64_t global_k = k0 + tile_k;
          if (global_row < rows && global_k < kHiddenSize)
          {
            a_tile[tile_row][tile_k] =
                a_expert[static_cast<int64_t>(global_row) * kHiddenSize + global_k];
          }
          else
          {
            a_tile[tile_row][tile_k] = 0.0f;
          }
        }

        for (int idx = tid; idx < TileN * TileK; idx += blockDim.x * blockDim.y)
        {
          int tile_col = idx / TileK;
          int tile_k = idx % TileK;
          int global_col = blockIdx.x * TileN + tile_col;
          int64_t global_k = k0 + tile_k;
          if (global_col < 2 * kIntermediateSize && global_k < kHiddenSize)
          {
            int64_t weight_idx =
                static_cast<int64_t>(global_col) * kHiddenSize + global_k;
            int64_t scale_idx = static_cast<int64_t>(global_col / kBlock) * (kHiddenSize / kBlock) +
                                (global_k / kBlock);
            b_tile[tile_k][tile_col] = fp8_to_float(w13_fp8[weight_idx]) * w13_scale[scale_idx];
          }
          else
          {
            b_tile[tile_k][tile_col] = 0.0f;
          }
        }

        __syncthreads();

        if (row < rows && col < 2 * kIntermediateSize)
        {
#pragma unroll
          for (int kk = 0; kk < TileK; ++kk)
          {
            acc += a_tile[local_row][kk] * b_tile[kk][local_col];
          }
        }

        __syncthreads();
      }

      if (row < rows && col < 2 * kIntermediateSize)
      {
        g1_expert[static_cast<int64_t>(row) * (2 * kIntermediateSize) + col] = acc;
      }
    }
  }

  template <int TileM, int TileN, int TileK>
  __global__ void gemm2_grouped_variable_fp8_kernel(
      const float *__restrict__ c, const int32_t *__restrict__ expert_offsets,
      const int32_t *__restrict__ expert_counts,
      const __nv_fp8_e4m3 *__restrict__ gemm2_weights,
      const float *__restrict__ gemm2_weights_scale, float *__restrict__ o)
  {
    // Same structure as GEMM1, but for the second projection C @ W2^T.
    __shared__ float c_tile[TileM][TileK];
    __shared__ float b_tile[TileK][TileN + 1];

    int32_t local_expert = blockIdx.y;
    int32_t rows = expert_counts[local_expert];
    if (rows <= 0)
    {
      return;
    }

    int32_t row_offset = expert_offsets[local_expert];
    const float *c_expert = c + static_cast<int64_t>(row_offset) * kIntermediateSize;
    float *o_expert = o + static_cast<int64_t>(row_offset) * kHiddenSize;
    const __nv_fp8_e4m3 *w2_fp8 =
        gemm2_weights + static_cast<int64_t>(local_expert) * kHiddenSize * kIntermediateSize;
    const float *w2_scale = gemm2_weights_scale +
                            static_cast<int64_t>(local_expert) * (kHiddenSize / kBlock) *
                                (kIntermediateSize / kBlock);

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int tid = local_row * blockDim.x + local_col;
    int col = blockIdx.x * TileN + local_col;

    for (int32_t row0 = 0; row0 < rows; row0 += TileM)
    {
      int row = row0 + local_row;
      float acc = 0.0f;

      for (int64_t k0 = 0; k0 < kIntermediateSize; k0 += TileK)
      {
        for (int idx = tid; idx < TileM * TileK; idx += blockDim.x * blockDim.y)
        {
          int tile_row = idx / TileK;
          int tile_k = idx % TileK;
          int global_row = row0 + tile_row;
          int64_t global_k = k0 + tile_k;
          if (global_row < rows && global_k < kIntermediateSize)
          {
            c_tile[tile_row][tile_k] =
                c_expert[static_cast<int64_t>(global_row) * kIntermediateSize + global_k];
          }
          else
          {
            c_tile[tile_row][tile_k] = 0.0f;
          }
        }

        for (int idx = tid; idx < TileN * TileK; idx += blockDim.x * blockDim.y)
        {
          int tile_col = idx / TileK;
          int tile_k = idx % TileK;
          int global_col = blockIdx.x * TileN + tile_col;
          int64_t global_k = k0 + tile_k;
          if (global_col < kHiddenSize && global_k < kIntermediateSize)
          {
            int64_t weight_idx = static_cast<int64_t>(global_col) * kIntermediateSize + global_k;
            int64_t scale_idx = static_cast<int64_t>(global_col / kBlock) *
                                    (kIntermediateSize / kBlock) +
                                (global_k / kBlock);
            b_tile[tile_k][tile_col] = fp8_to_float(w2_fp8[weight_idx]) * w2_scale[scale_idx];
          }
          else
          {
            b_tile[tile_k][tile_col] = 0.0f;
          }
        }

        __syncthreads();

        if (row < rows && col < kHiddenSize)
        {
#pragma unroll
          for (int kk = 0; kk < TileK; ++kk)
          {
            acc += c_tile[local_row][kk] * b_tile[kk][local_col];
          }
        }

        __syncthreads();
      }

      if (row < rows && col < kHiddenSize)
      {
        o_expert[static_cast<int64_t>(row) * kHiddenSize + col] = acc;
      }
    }
  }

  void RunGemmF32(const float *a, const float *b, float *d, int m, int n, int k,
                  cudaStream_t stream)
  {
    // Small-expert fallback in the grouped path. For tiny M, the launch/setup overhead of the
    // grouped tensor-core kernel can dominate, so a simple SIMT GEMM remains competitive.
    using ElementA = float;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = float;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = float;
    using LayoutC = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                                             LayoutC, float, cutlass::arch::OpClassSimt,
                                             cutlass::arch::Sm80>;
    Gemm gemm_op;
    typename Gemm::Arguments args({m, n, k}, {a, k}, {b, k}, {d, n}, {d, n}, {1.0f, 0.0f});
    CheckCutlassStatus(gemm_op.initialize(args, nullptr, stream), "SIMT GEMM init");
    CheckCutlassStatus(gemm_op.run(stream), "SIMT GEMM run");
  }

  void RunGroupedGemm(cutlass::gemm::GemmCoord *d_problem_sizes,
                      cutlass::gemm::GemmCoord *h_problem_sizes,
                      cutlass_grouped_f32::ElementA **ptr_a,
                      cutlass_grouped_f32::ElementB **ptr_b,
                      cutlass_grouped_f32::ElementC **ptr_c,
                      cutlass_grouped_f32::ElementC **ptr_d, int expert_count, int k, int n,
                      cudaStream_t stream)
  {
    // CUTLASS grouped GEMM for the large-expert subset in the grouped_workload path.
    using GemmGrouped = cutlass_grouped_f32::GemmGrouped;
    using LongIndex = typename GemmGrouped::LayoutA::Stride::LongIndex;

    std::vector<LongIndex> lda(expert_count, k);
    std::vector<LongIndex> ldb(expert_count, k);
    std::vector<LongIndex> ldc(expert_count, n);
    std::vector<LongIndex> ldd(expert_count, n);
    AsyncBuffer<LongIndex> d_lda(expert_count, stream);
    AsyncBuffer<LongIndex> d_ldb(expert_count, stream);
    AsyncBuffer<LongIndex> d_ldc(expert_count, stream);
    AsyncBuffer<LongIndex> d_ldd(expert_count, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_lda.get(), lda.data(), expert_count * sizeof(LongIndex),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ldb.get(), ldb.data(), expert_count * sizeof(LongIndex),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ldc.get(), ldc.data(), expert_count * sizeof(LongIndex),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ldd.get(), ldd.data(), expert_count * sizeof(LongIndex),
                               cudaMemcpyHostToDevice, stream));

    typename GemmGrouped::Arguments args(
        d_problem_sizes, expert_count, 1024, {1.0f, 0.0f}, ptr_a, ptr_b, ptr_c, ptr_d,
        d_lda.get(), d_ldb.get(), d_ldc.get(), d_ldd.get(), h_problem_sizes);

    GemmGrouped gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(args);
    AsyncBuffer<uint8_t> workspace(workspace_size, stream);
    CheckCutlassStatus(gemm_op.initialize(args, workspace.get(), stream), "Grouped GEMM init");
    CheckCutlassStatus(gemm_op.run(stream), "Grouped GEMM run");
  }

  template <int TileM, int TileN, int TileK>
  void LaunchGroupedFp8Gemm1(const float *a_f32, const int32_t *expert_offsets,
                             const int32_t *expert_counts,
                             const __nv_fp8_e4m3 *gemm1_weights,
                             const float *gemm1_weights_scale, float *g1_f32,
                             cudaStream_t stream)
  {
    dim3 block(TileN, TileM);
    dim3 grid((2 * kIntermediateSize + TileN - 1) / TileN, kLocalExperts);
    gemm1_grouped_variable_fp8_kernel<TileM, TileN, TileK><<<grid, block, 0, stream>>>(
        a_f32, expert_offsets, expert_counts, gemm1_weights, gemm1_weights_scale, g1_f32);
    CHECK_CUDA(cudaGetLastError());
  }

  template <int TileM, int TileN, int TileK>
  void LaunchGroupedFp8Gemm2(const float *c_f32, const int32_t *expert_offsets,
                             const int32_t *expert_counts,
                             const __nv_fp8_e4m3 *gemm2_weights,
                             const float *gemm2_weights_scale, float *o_f32,
                             cudaStream_t stream)
  {
    dim3 block(TileN, TileM);
    dim3 grid((kHiddenSize + TileN - 1) / TileN, kLocalExperts);
    gemm2_grouped_variable_fp8_kernel<TileM, TileN, TileK><<<grid, block, 0, stream>>>(
        c_f32, expert_offsets, expert_counts, gemm2_weights, gemm2_weights_scale, o_f32);
    CHECK_CUDA(cudaGetLastError());
  }

  void RunDeviceOnlyGroupedPipeline(const ffi::TensorView &hidden_states,
                                    const ffi::TensorView &hidden_states_scale,
                                    const ffi::TensorView &gemm1_weights,
                                    const ffi::TensorView &gemm1_weights_scale,
                                    const ffi::TensorView &gemm2_weights,
                                    const ffi::TensorView &gemm2_weights_scale, int32_t *topk_idx,
                                    float *weights, float *output_fp32, int64_t t,
                                    int32_t local_expert_offset, cudaStream_t stream)
  {
    // Small/medium-sequence path:
    // keep all scheduling on device and run both GEMMs with the custom FP8 kernels.
    int64_t scale_t = hidden_states_scale.size(1);
    int32_t max_rows = static_cast<int32_t>(t * kTopK);
    size_t scan_workspace_size = GetExpertScanWorkspaceSize(stream);
    size_t workspace_bytes = GetDeviceOnlyWorkspaceBytes(max_rows, scan_workspace_size);
    WorkspaceArena workspace(workspace_bytes, stream);

    // Step 1. Build the expert-major row layout for all local assignments.
    PreparedExpertWorkload prepared =
        PrepareExpertWorkload(topk_idx, t, local_expert_offset, max_rows, scan_workspace_size,
                              &workspace, stream);
    // The workspace packs:
    // counts/offsets/reordered rows + A + G1 + C + O for the full local assignment set.
    float *a_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kHiddenSize);
    float *g1_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * 2 * kIntermediateSize);
    float *c_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kIntermediateSize);
    float *o_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kHiddenSize);

    // Step 2. Gather the routed token rows and dequantize activations to FP32.
    fused_gather_dequant_f32_kernel<<<(static_cast<int64_t>(max_rows) * kHiddenSize + 255) / 256,
                                      256, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3 *>(hidden_states.data_ptr()), prepared.reordered_tokens,
        static_cast<const float *>(hidden_states_scale.data_ptr()), a_f32, max_rows, scale_t);
    CHECK_CUDA(cudaGetLastError());

    // Step 3. Run the two expert projections with the custom device-only FP8 kernels.
    LaunchGroupedFp8Gemm1<kGemm1TileM, kGemm1TileN, kGemm1TileK>(
        a_f32, prepared.expert_offsets, prepared.expert_counts,
        static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()),
        static_cast<const float *>(gemm1_weights_scale.data_ptr()), g1_f32, stream);

    // Step 4. Apply SwiGLU between the two projections.
    swiglu_f32_kernel<<<(static_cast<int64_t>(max_rows) * kIntermediateSize + 255) / 256,
                        256, 0, stream>>>(g1_f32, c_f32, prepared.total_rows);
    CHECK_CUDA(cudaGetLastError());

    LaunchGroupedFp8Gemm2<kGemm2TileM, kGemm2TileN, kGemm2TileK>(
        c_f32, prepared.expert_offsets, prepared.expert_counts,
        static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()),
        static_cast<const float *>(gemm2_weights_scale.data_ptr()), o_f32, stream);

    // Step 5. Scatter the reordered expert rows back into token-major output order.
    optimized_scatter_add_kernel<<<(static_cast<int64_t>(max_rows) * kHiddenSize + 255) / 256,
                                   256, 0, stream>>>(
        o_f32, prepared.reordered_tokens, prepared.reordered_local_experts, weights, output_fp32,
        prepared.total_rows, local_expert_offset);
    CHECK_CUDA(cudaGetLastError());
  }

  void RunGroupedWorkloadPipeline(
      const ffi::TensorView &hidden_states, const ffi::TensorView &hidden_states_scale,
      const ffi::TensorView &gemm1_weights, const ffi::TensorView &gemm1_weights_scale,
      const ffi::TensorView &gemm2_weights, const ffi::TensorView &gemm2_weights_scale,
      int32_t *topk_idx, float *weights, float *output_fp32, int64_t t,
      int32_t local_expert_offset, cudaStream_t stream)
  {
    // Large-sequence path:
    // reuse the same workload preparation, then split experts into:
    // - small experts: dequantize one expert and run a SIMT GEMM
    // - large experts: batch them into one CUTLASS grouped GEMM launch
    int64_t scale_t = hidden_states_scale.size(1);
    int32_t max_rows = static_cast<int32_t>(t * kTopK);
    size_t scan_workspace_size = GetExpertScanWorkspaceSize(stream);
    size_t prep_workspace_bytes = GetGroupedPrepWorkspaceBytes(max_rows, scan_workspace_size);
    WorkspaceArena prep_workspace(prep_workspace_bytes, stream);

    // Step 1. Build the shared expert-major row layout.
    PreparedExpertWorkload prepared =
        PrepareExpertWorkload(topk_idx, t, local_expert_offset, max_rows, scan_workspace_size,
                              &prep_workspace, stream);

    // Step 2. Bring back expert row counts so the host can build grouped GEMM descriptors.
    std::vector<int32_t> h_counts(kLocalExperts);
    std::vector<int32_t> h_offsets(kLocalExperts + 1);
    CHECK_CUDA(cudaMemcpyAsync(h_counts.data(), prepared.expert_counts,
                               kLocalExperts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_offsets.data(), prepared.expert_offsets,
                               (kLocalExperts + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    int32_t total_reordered = h_offsets[kLocalExperts];
    GroupedExpertPlan grouped_gemm1 = BuildGroupedExpertPlan(h_counts, kGroupedGemm1Threshold);
    GroupedExpertPlan grouped_gemm2 = BuildGroupedExpertPlan(h_counts, kGroupedGemm2Threshold);

    if (total_reordered == 0)
    {
      return;
    }

    size_t compute_workspace_bytes = GetGroupedComputeWorkspaceBytes(
        total_reordered, grouped_gemm1.experts.size(), grouped_gemm2.experts.size());
    WorkspaceArena compute_workspace(compute_workspace_bytes, stream);
    float *g1_f32 = compute_workspace.Alloc<float>(
        static_cast<size_t>(total_reordered) * 2 * kIntermediateSize);
    float *c_f32 =
        compute_workspace.Alloc<float>(static_cast<size_t>(total_reordered) * kIntermediateSize);
    float *a_f32 =
        compute_workspace.Alloc<float>(static_cast<size_t>(total_reordered) * kHiddenSize);
    float *w13_grouped_f32 = compute_workspace.Alloc<float>(
        static_cast<size_t>(grouped_gemm1.experts.size()) * 2 * kIntermediateSize * kHiddenSize);
    float *w2_grouped_f32 = compute_workspace.Alloc<float>(
        static_cast<size_t>(grouped_gemm2.experts.size()) * kHiddenSize * kIntermediateSize);
    float *w13_small_tmp = compute_workspace.Alloc<float>(2 * kIntermediateSize * kHiddenSize);
    float *w2_small_tmp = compute_workspace.Alloc<float>(kHiddenSize * kIntermediateSize);
    float *o_grouped_f32 =
        compute_workspace.Alloc<float>(static_cast<size_t>(total_reordered) * kHiddenSize);
    int32_t *d_grouped_gemm1_experts =
        compute_workspace.Alloc<int32_t>(grouped_gemm1.experts.size());
    int32_t *d_grouped_gemm2_experts =
        compute_workspace.Alloc<int32_t>(grouped_gemm2.experts.size());

    // Step 3. Gather and dequantize all routed activation rows once.
    fused_gather_dequant_f32_kernel<<<(static_cast<int64_t>(total_reordered) * kHiddenSize + 255) /
                                          256,
                                      256, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3 *>(hidden_states.data_ptr()), prepared.reordered_tokens,
        static_cast<const float *>(hidden_states_scale.data_ptr()), a_f32, total_reordered,
        scale_t);
    CHECK_CUDA(cudaGetLastError());
    if (!grouped_gemm1.experts.empty())
    {
      // Step 4a. Materialize GEMM1 weights for the large-expert subset.
      CHECK_CUDA(cudaMemcpyAsync(d_grouped_gemm1_experts, grouped_gemm1.experts.data(),
                                 grouped_gemm1.experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
      dequant_selected_experts_to_f32_kernel<<<
          (static_cast<int64_t>(grouped_gemm1.experts.size()) * 2 * kIntermediateSize *
               kHiddenSize +
           255) /
              256,
          256, 0, stream>>>(d_grouped_gemm1_experts,
                             static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()),
                             static_cast<const float *>(gemm1_weights_scale.data_ptr()),
                             w13_grouped_f32, static_cast<int32_t>(grouped_gemm1.experts.size()),
                             2 * kIntermediateSize, kHiddenSize);
      CHECK_CUDA(cudaGetLastError());
    }

    if (!grouped_gemm2.experts.empty())
    {
      // Step 4b. Materialize GEMM2 weights for the large-expert subset.
      CHECK_CUDA(cudaMemcpyAsync(d_grouped_gemm2_experts, grouped_gemm2.experts.data(),
                                 grouped_gemm2.experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
      dequant_selected_experts_to_f32_kernel<<<
          (static_cast<int64_t>(grouped_gemm2.experts.size()) * kHiddenSize * kIntermediateSize +
           255) /
              256,
          256, 0, stream>>>(d_grouped_gemm2_experts,
                             static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()),
                             static_cast<const float *>(gemm2_weights_scale.data_ptr()),
                             w2_grouped_f32, static_cast<int32_t>(grouped_gemm2.experts.size()),
                             kHiddenSize, kIntermediateSize);
      CHECK_CUDA(cudaGetLastError());
    }

    std::vector<cutlass::gemm::GemmCoord> prob1;
    std::vector<cutlass_grouped_f32::ElementA *> ptr_a1;
    std::vector<cutlass_grouped_f32::ElementB *> ptr_b1;
    std::vector<cutlass_grouped_f32::ElementC *> ptr_c1;
    std::vector<cutlass_grouped_f32::ElementC *> ptr_d1;

    // Step 5. Execute GEMM1 using a mixed strategy:
    // tiny experts use the SIMT fallback, large experts are packed into one grouped CUTLASS GEMM.
    for (int i = 0; i < kLocalExperts; ++i)
    {
      if (h_counts[i] <= 0)
      {
        continue;
      }
      if (h_counts[i] < kGroupedGemm1Threshold)
      {
        // Small expert: materialize just one expert's weights and run the SIMT fallback.
        dequant_weights_to_f32_kernel<<<(static_cast<int64_t>(2 * kIntermediateSize) * kHiddenSize +
                                         255) /
                                            256,
                                        256, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()) +
                static_cast<int64_t>(i) * 2 * kIntermediateSize * kHiddenSize,
            static_cast<const float *>(gemm1_weights_scale.data_ptr()) +
                static_cast<int64_t>(i) * ((2 * kIntermediateSize) / kBlock) *
                    (kHiddenSize / kBlock),
            w13_small_tmp, 1, 2 * kIntermediateSize, kHiddenSize);
        CHECK_CUDA(cudaGetLastError());
        RunGemmF32(a_f32 + static_cast<int64_t>(h_offsets[i]) * kHiddenSize, w13_small_tmp,
                   g1_f32 + static_cast<int64_t>(h_offsets[i]) * 2 * kIntermediateSize,
                   h_counts[i], 2 * kIntermediateSize, kHiddenSize, stream);
      }
      else
      {
        // Large expert: enqueue it into the grouped GEMM descriptor arrays.
        int32_t grouped_idx = grouped_gemm1.slots[i];
        prob1.push_back({h_counts[i], static_cast<int>(2 * kIntermediateSize),
                         static_cast<int>(kHiddenSize)});
        ptr_a1.push_back(a_f32 + static_cast<int64_t>(h_offsets[i]) * kHiddenSize);
        ptr_b1.push_back(w13_grouped_f32 +
                         static_cast<int64_t>(grouped_idx) * 2 * kIntermediateSize * kHiddenSize);
        ptr_c1.push_back(g1_f32 + static_cast<int64_t>(h_offsets[i]) *
                                            2 * kIntermediateSize);
        ptr_d1.push_back(g1_f32 + static_cast<int64_t>(h_offsets[i]) *
                                            2 * kIntermediateSize);
      }
    }

    if (!prob1.empty())
    {
      AsyncBuffer<cutlass::gemm::GemmCoord> d_prob1(prob1.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementA *> d_ptr_a1(ptr_a1.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementB *> d_ptr_b1(ptr_b1.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementC *> d_ptr_c1(ptr_c1.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementC *> d_ptr_d1(ptr_d1.size(), stream);
      CHECK_CUDA(cudaMemcpyAsync(d_prob1.get(), prob1.data(),
                                 prob1.size() * sizeof(cutlass::gemm::GemmCoord),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_a1.get(), ptr_a1.data(),
                                 ptr_a1.size() * sizeof(cutlass_grouped_f32::ElementA *),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_b1.get(), ptr_b1.data(),
                                 ptr_b1.size() * sizeof(cutlass_grouped_f32::ElementB *),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_c1.get(), ptr_c1.data(),
                                 ptr_c1.size() * sizeof(cutlass_grouped_f32::ElementC *),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_d1.get(), ptr_d1.data(),
                                 ptr_d1.size() * sizeof(cutlass_grouped_f32::ElementC *),
                                 cudaMemcpyHostToDevice, stream));
      RunGroupedGemm(d_prob1.get(), prob1.data(), d_ptr_a1.get(), d_ptr_b1.get(), d_ptr_c1.get(),
                     d_ptr_d1.get(), static_cast<int>(prob1.size()), kHiddenSize,
                     2 * kIntermediateSize, stream);
    }

    // Step 6. Apply SwiGLU after GEMM1 completes for all reordered rows.
    swiglu_f32_kernel<<<(static_cast<int64_t>(total_reordered) * kIntermediateSize + 255) / 256,
                        256, 0, stream>>>(g1_f32, c_f32, prepared.total_rows);
    CHECK_CUDA(cudaGetLastError());

    std::vector<cutlass::gemm::GemmCoord> prob2;
    std::vector<cutlass_grouped_f32::ElementA *> ptr_a2;
    std::vector<cutlass_grouped_f32::ElementB *> ptr_b2;
    std::vector<cutlass_grouped_f32::ElementC *> ptr_d2;

    // Step 7. Execute GEMM2 with the same small-expert / large-expert split.
    for (int i = 0; i < kLocalExperts; ++i)
    {
      if (h_counts[i] <= 0)
      {
        continue;
      }
      if (h_counts[i] < kGroupedGemm2Threshold)
      {
        dequant_weights_to_f32_kernel<<<(static_cast<int64_t>(kHiddenSize) * kIntermediateSize +
                                         255) /
                                            256,
                                        256, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()) +
                static_cast<int64_t>(i) * kHiddenSize * kIntermediateSize,
            static_cast<const float *>(gemm2_weights_scale.data_ptr()) +
                static_cast<int64_t>(i) * (kHiddenSize / kBlock) * (kIntermediateSize / kBlock),
            w2_small_tmp, 1, kHiddenSize, kIntermediateSize);
        CHECK_CUDA(cudaGetLastError());
        RunGemmF32(c_f32 + static_cast<int64_t>(h_offsets[i]) * kIntermediateSize, w2_small_tmp,
                   o_grouped_f32 + static_cast<int64_t>(h_offsets[i]) * kHiddenSize,
                   h_counts[i], kHiddenSize, kIntermediateSize, stream);
      }
      else
      {
        int32_t grouped_idx = grouped_gemm2.slots[i];
        prob2.push_back(
            {h_counts[i], static_cast<int>(kHiddenSize), static_cast<int>(kIntermediateSize)});
        ptr_a2.push_back(c_f32 + static_cast<int64_t>(h_offsets[i]) * kIntermediateSize);
        ptr_b2.push_back(w2_grouped_f32 +
                         static_cast<int64_t>(grouped_idx) * kHiddenSize * kIntermediateSize);
        ptr_d2.push_back(o_grouped_f32 + static_cast<int64_t>(h_offsets[i]) * kHiddenSize);
      }
    }

    if (!prob2.empty())
    {
      AsyncBuffer<cutlass::gemm::GemmCoord> d_prob2(prob2.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementA *> d_ptr_a2(ptr_a2.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementB *> d_ptr_b2(ptr_b2.size(), stream);
      AsyncBuffer<cutlass_grouped_f32::ElementC *> d_ptr_d2(ptr_d2.size(), stream);
      CHECK_CUDA(cudaMemcpyAsync(d_prob2.get(), prob2.data(),
                                 prob2.size() * sizeof(cutlass::gemm::GemmCoord),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_a2.get(), ptr_a2.data(),
                                 ptr_a2.size() * sizeof(cutlass_grouped_f32::ElementA *),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_b2.get(), ptr_b2.data(),
                                 ptr_b2.size() * sizeof(cutlass_grouped_f32::ElementB *),
                                 cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptr_d2.get(), ptr_d2.data(),
                                 ptr_d2.size() * sizeof(cutlass_grouped_f32::ElementC *),
                                 cudaMemcpyHostToDevice, stream));
      RunGroupedGemm(d_prob2.get(), prob2.data(), d_ptr_a2.get(), d_ptr_b2.get(), d_ptr_d2.get(),
                     d_ptr_d2.get(), static_cast<int>(prob2.size()), kIntermediateSize,
                     kHiddenSize, stream);
    }

    // Step 8. Scatter the reordered expert outputs back to the final token-major output buffer.
    optimized_scatter_add_kernel<<<(static_cast<int64_t>(total_reordered) * kHiddenSize + 255) /
                                       256,
                                   256, 0, stream>>>(
        o_grouped_f32, prepared.reordered_tokens, prepared.reordered_local_experts, weights,
        output_fp32, prepared.total_rows, local_expert_offset);
    CHECK_CUDA(cudaGetLastError());
  }

} // namespace

// Public entrypoint:
// validate tensor contracts, run routing, pick the execution path, and cast the
// accumulated FP32 output back to the required BF16 result tensor.
ffi::Tensor kernel(const ffi::TensorView &routing_logits, const ffi::TensorView &routing_bias,
                   const ffi::TensorView &hidden_states,
                   const ffi::TensorView &hidden_states_scale,
                   const ffi::TensorView &gemm1_weights,
                   const ffi::TensorView &gemm1_weights_scale,
                   const ffi::TensorView &gemm2_weights,
                   const ffi::TensorView &gemm2_weights_scale, int32_t local_expert_offset,
                   float routed_scaling_factor)
{
  int64_t t = routing_logits.size(0);
  int64_t num_hidden_blocks = kHiddenSize / kBlock;
  int64_t num_intermediate_blocks = kIntermediateSize / kBlock;
  int64_t num_gemm1_out_blocks = (2 * kIntermediateSize) / kBlock;

  DLDataType float32_dtype{kDLFloat, 32, 1};
  DLDataType bfloat16_dtype{kDLBfloat, 16, 1};
  DLDataType float8_dtype{kDLFloat8_e4m3fn, 8, 1};

  // Match the definition exactly before launching any CUDA work.
  CheckTensor(routing_logits, "routing_logits", 2, ffi::ShapeView({t, kGlobalExperts}),
              float32_dtype);
  CheckScalarVector(routing_bias, "routing_bias", kGlobalExperts, bfloat16_dtype);
  CheckTensor(hidden_states, "hidden_states", 2, ffi::ShapeView({t, kHiddenSize}), float8_dtype);
  CheckTensor(hidden_states_scale, "hidden_states_scale", 2,
              ffi::ShapeView({num_hidden_blocks, t}), float32_dtype);
  CheckTensor(gemm1_weights, "gemm1_weights", 3,
              ffi::ShapeView({kLocalExperts, 2 * kIntermediateSize, kHiddenSize}), float8_dtype);
  CheckTensor(gemm1_weights_scale, "gemm1_weights_scale", 3,
              ffi::ShapeView({kLocalExperts, num_gemm1_out_blocks, num_hidden_blocks}),
              float32_dtype);
  CheckTensor(gemm2_weights, "gemm2_weights", 3,
              ffi::ShapeView({kLocalExperts, kHiddenSize, kIntermediateSize}), float8_dtype);
  CheckTensor(gemm2_weights_scale, "gemm2_weights_scale", 3,
              ffi::ShapeView({kLocalExperts, num_hidden_blocks, num_intermediate_blocks}),
              float32_dtype);

  CheckSameDevice(routing_logits, routing_bias, "routing_logits", "routing_bias");
  CheckSameDevice(routing_logits, hidden_states, "routing_logits", "hidden_states");
  CheckSameDevice(routing_logits, hidden_states_scale, "routing_logits", "hidden_states_scale");
  CheckSameDevice(routing_logits, gemm1_weights, "routing_logits", "gemm1_weights");
  CheckSameDevice(routing_logits, gemm1_weights_scale, "routing_logits", "gemm1_weights_scale");
  CheckSameDevice(routing_logits, gemm2_weights, "routing_logits", "gemm2_weights");
  CheckSameDevice(routing_logits, gemm2_weights_scale, "routing_logits", "gemm2_weights_scale");

  DLDevice device = routing_logits.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  ffi::Shape output_shape({t, kHiddenSize});
  ffi::Tensor output_tensor = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, output_shape,
                                                        bfloat16_dtype, device);

  constexpr int threads_1d = 256;
  int64_t routing_total = t * kGlobalExperts;
  int64_t output_total = t * kHiddenSize;

  AsyncBuffer<float> s(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<float> s_with_bias(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<int32_t> topk_idx(static_cast<size_t>(t) * kTopK, stream);
  AsyncBuffer<float> weights(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<float> output_fp32(static_cast<size_t>(t) * kHiddenSize, stream);

  CHECK_CUDA(cudaMemsetAsync(output_fp32.get(), 0,
                             sizeof(float) * static_cast<size_t>(t) * kHiddenSize, stream));

  sigmoid_bias_kernel<<<(routing_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
      static_cast<const float *>(routing_logits.data_ptr()),
      static_cast<const __nv_bfloat16 *>(routing_bias.data_ptr()), s.get(), s_with_bias.get(), t);
  CHECK_CUDA(cudaGetLastError());

  routing_select_kernel<<<t, 1, 0, stream>>>(s.get(), s_with_bias.get(), topk_idx.get(),
                                              weights.get(), t, routed_scaling_factor);
  CHECK_CUDA(cudaGetLastError());

  // Hybrid dispatch: small/medium sequences favor the fully device-side custom path, while
  // large sequences amortize the grouped CUTLASS path well enough to offset host setup.
  if (t <= kHybridDispatchSeqLenThreshold)
  {
    RunDeviceOnlyGroupedPipeline(hidden_states, hidden_states_scale, gemm1_weights,
                                 gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                                 topk_idx.get(), weights.get(), output_fp32.get(), t,
                                 local_expert_offset, stream);
  }
  else
  {
    RunGroupedWorkloadPipeline(hidden_states, hidden_states_scale, gemm1_weights,
                               gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                               topk_idx.get(), weights.get(), output_fp32.get(), t,
                               local_expert_offset, stream);
  }

  cast_output_kernel<<<(output_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
      output_fp32.get(), static_cast<__nv_bfloat16 *>(output_tensor.data_ptr()), output_total);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  return output_tensor;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
