#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
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
  constexpr int64_t kHybridDispatchSeqLenThreshold = 512;
  constexpr int kGroupedGemm1Threshold = 1;
  constexpr int kGroupedGemm2Threshold = 1;
  constexpr int kSelectedDequantThreads = 128;
  constexpr bool kUseGroupedF16Gemm1 = true;

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

  class GpuStageTimer
  {
  public:
    explicit GpuStageTimer(const char *stage_name) : stage_name_(stage_name)
    {
      CHECK_CUDA(cudaEventCreate(&start_));
      CHECK_CUDA(cudaEventCreate(&end_));
    }

    ~GpuStageTimer()
    {
      if (start_ != nullptr)
      {
        cudaError_t err = cudaEventDestroy(start_);
        (void)err;
      }
      if (end_ != nullptr)
      {
        cudaError_t err = cudaEventDestroy(end_);
        (void)err;
      }
    }

    GpuStageTimer(const GpuStageTimer &) = delete;
    GpuStageTimer &operator=(const GpuStageTimer &) = delete;

    void Start(cudaStream_t stream)
    {
      has_start_ = true;
      CHECK_CUDA(cudaEventRecord(start_, stream));
    }
    void Stop(cudaStream_t stream)
    {
      has_end_ = true;
      CHECK_CUDA(cudaEventRecord(end_, stream));
    }

    float ElapsedMs() const
    {
      if (!has_start_ || !has_end_)
      {
        return 0.0f;
      }
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start_, end_));
      return ms;
    }

    const char *name() const { return stage_name_; }

  private:
    const char *stage_name_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t end_ = nullptr;
    bool has_start_ = false;
    bool has_end_ = false;
  };

  class CpuWallTimer
  {
  public:
    using Clock = std::chrono::steady_clock;

    void Start() { start_ = Clock::now(); }

    double Stop()
    {
      elapsed_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
      return elapsed_ms_;
    }

    double ElapsedMs() const { return elapsed_ms_; }

  private:
    Clock::time_point start_{};
    double elapsed_ms_ = 0.0;
  };

  struct DeviceOnlyPipelineProfile
  {
    DeviceOnlyPipelineProfile()
        : prepare("prepare_workload"), gather("gather_dequant"), gemm1("gemm1_device_only"),
          swiglu("swiglu"), gemm2("gemm2_device_only"), scatter("scatter")
    {
    }

    GpuStageTimer prepare;
    GpuStageTimer gather;
    GpuStageTimer gemm1;
    GpuStageTimer swiglu;
    GpuStageTimer gemm2;
    GpuStageTimer scatter;
    int64_t t = 0;
    int32_t max_rows = 0;
    int32_t total_rows = 0;
    double pipeline_wall_ms = 0.0;
  };

  struct GroupedWorkloadPipelineProfile
  {
    GroupedWorkloadPipelineProfile()
        : prepare("prepare_workload"), gather("gather_dequant"),
          grouped_gemm1_dequant("dequant_grouped_gemm1_weights"),
          gemm1_small("gemm1_small_simt"), gemm1_grouped("gemm1_grouped_cutlass_tc"),
          swiglu("swiglu"), grouped_gemm2_dequant("dequant_grouped_gemm2_weights"),
          gemm2_small("gemm2_small_simt"), gemm2_grouped("gemm2_grouped_cutlass_tc"),
          scatter("scatter")
    {
    }

    GpuStageTimer prepare;
    GpuStageTimer gather;
    GpuStageTimer grouped_gemm1_dequant;
    GpuStageTimer gemm1_small;
    GpuStageTimer gemm1_grouped;
    GpuStageTimer swiglu;
    GpuStageTimer grouped_gemm2_dequant;
    GpuStageTimer gemm2_small;
    GpuStageTimer gemm2_grouped;
    GpuStageTimer scatter;
    int64_t t = 0;
    int32_t max_rows = 0;
    int32_t total_rows = 0;
    int32_t grouped_gemm1_experts = 0;
    int32_t grouped_gemm2_experts = 0;
    int32_t small_gemm1_experts = 0;
    int32_t small_gemm2_experts = 0;
    double counts_d2h_ms = 0.0;
    double host_plan_ms = 0.0;
    double pipeline_wall_ms = 0.0;
  };

  inline bool ProfileEnabled()
  {
    const char *value = std::getenv("FUSED_MOE_PROFILE");
    if (value == nullptr)
    {
      return false;
    }
    if (value[0] == '\0')
    {
      return false;
    }
    if ((value[0] == '0' && value[1] == '\0') ||
        ((value[0] == 'f' || value[0] == 'F') && (value[1] == 'a' || value[1] == 'A')) ||
        ((value[0] == 'n' || value[0] == 'N') && (value[1] == 'o' || value[1] == 'O')))
    {
      return false;
    }
    return true;
  }

  inline void EmitProfileLine(const char *line)
  {
    if (!ProfileEnabled())
    {
      return;
    }
    std::fputs(line, stdout);
    std::fflush(stdout);

    const char *path = std::getenv("FUSED_MOE_PROFILE_PATH");
    if (path == nullptr || path[0] == '\0')
    {
      return;
    }
    if (FILE *fp = std::fopen(path, "a"))
    {
      std::fputs(line, fp);
      std::fclose(fp);
    }
  }

  inline void PrintDeviceOnlyPipelineProfile(const DeviceOnlyPipelineProfile &profile)
  {
    double prepare_ms = profile.prepare.ElapsedMs();
    double gather_ms = profile.gather.ElapsedMs();
    double gemm1_ms = profile.gemm1.ElapsedMs();
    double swiglu_ms = profile.swiglu.ElapsedMs();
    double gemm2_ms = profile.gemm2.ElapsedMs();
    double scatter_ms = profile.scatter.ElapsedMs();
    double gpu_stage_sum_ms =
        prepare_ms + gather_ms + gemm1_ms + swiglu_ms + gemm2_ms + scatter_ms;
    char line[512];
    std::snprintf(
        line, sizeof(line),
        "[FUSED_MOE_PROFILE] path=device_only t=%lld max_rows=%d total_rows=%d "
        "prepare_ms=%.3f gather_ms=%.3f gemm1_ms=%.3f swiglu_ms=%.3f gemm2_ms=%.3f "
        "scatter_ms=%.3f gpu_stage_sum_ms=%.3f pipeline_wall_ms=%.3f\n",
        static_cast<long long>(profile.t), profile.max_rows, profile.total_rows, prepare_ms,
        gather_ms, gemm1_ms, swiglu_ms, gemm2_ms, scatter_ms, gpu_stage_sum_ms,
        profile.pipeline_wall_ms);
    EmitProfileLine(line);
  }

  inline void PrintGroupedWorkloadPipelineProfile(const GroupedWorkloadPipelineProfile &profile)
  {
    double prepare_ms = profile.prepare.ElapsedMs();
    double gather_ms = profile.gather.ElapsedMs();
    double grouped_gemm1_dequant_ms = profile.grouped_gemm1_dequant.ElapsedMs();
    double gemm1_small_ms = profile.gemm1_small.ElapsedMs();
    double gemm1_grouped_ms = profile.gemm1_grouped.ElapsedMs();
    double swiglu_ms = profile.swiglu.ElapsedMs();
    double grouped_gemm2_dequant_ms = profile.grouped_gemm2_dequant.ElapsedMs();
    double gemm2_small_ms = profile.gemm2_small.ElapsedMs();
    double gemm2_grouped_ms = profile.gemm2_grouped.ElapsedMs();
    double scatter_ms = profile.scatter.ElapsedMs();
    double gpu_stage_sum_ms =
        prepare_ms + gather_ms + grouped_gemm1_dequant_ms + gemm1_small_ms + gemm1_grouped_ms +
        swiglu_ms + grouped_gemm2_dequant_ms + gemm2_small_ms + gemm2_grouped_ms + scatter_ms;
    char line[1024];
    std::snprintf(
        line, sizeof(line),
        "[FUSED_MOE_PROFILE] path=grouped_workload t=%lld max_rows=%d total_rows=%d "
        "grouped_gemm1_experts=%d small_gemm1_experts=%d grouped_gemm2_experts=%d "
        "small_gemm2_experts=%d prepare_ms=%.3f counts_d2h_ms=%.3f host_plan_ms=%.3f "
        "gather_ms=%.3f dequant_grouped_gemm1_ms=%.3f gemm1_small_ms=%.3f "
        "gemm1_cutlass_tc_ms=%.3f swiglu_ms=%.3f dequant_grouped_gemm2_ms=%.3f "
        "gemm2_small_ms=%.3f gemm2_cutlass_tc_ms=%.3f scatter_ms=%.3f gpu_stage_sum_ms=%.3f "
        "pipeline_wall_ms=%.3f\n",
        static_cast<long long>(profile.t), profile.max_rows, profile.total_rows,
        profile.grouped_gemm1_experts, profile.small_gemm1_experts,
        profile.grouped_gemm2_experts, profile.small_gemm2_experts, prepare_ms,
        profile.counts_d2h_ms, profile.host_plan_ms, gather_ms, grouped_gemm1_dequant_ms,
        gemm1_small_ms, gemm1_grouped_ms, swiglu_ms, grouped_gemm2_dequant_ms, gemm2_small_ms,
        gemm2_grouped_ms, scatter_ms, gpu_stage_sum_ms, profile.pipeline_wall_ms);
    EmitProfileLine(line);
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

  namespace cutlass_grouped_f16
  {

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAccumulator = float;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<
        typename cutlass::gemm::kernel::DefaultGemmGrouped<
            ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, ElementB,
            cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, ElementC,
            cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator,
                ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4>::GemmKernel>;

  } // namespace cutlass_grouped_f16

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

  // Wrapper Class for cudaMallocAsync, cudaFreeAsync
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
  class CachedDeviceBuffer
  {
  public:
    // Process-lifetime scratch cache: warmup pays allocation, measured iterations reuse it.
    CachedDeviceBuffer() = default;
    CachedDeviceBuffer(const CachedDeviceBuffer &) = delete;
    CachedDeviceBuffer &operator=(const CachedDeviceBuffer &) = delete;

    T *get(size_t count)
    {
      if (count == 0)
      {
        return ptr_;
      }
      int current_device = 0;
      CHECK_CUDA(cudaGetDevice(&current_device));
      if (ptr_ == nullptr || count > count_ || current_device != device_)
      {
        if (ptr_ != nullptr)
        {
          CHECK_CUDA(cudaFree(ptr_));
        }
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count));
        count_ = count;
        device_ = current_device;
      }
      return ptr_;
    }

    size_t size() const { return count_; }

  private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
    int device_ = -1;
  };

  // Efficient async D2H/H2D copy.
  template <typename T>
  class PinnedHostBuffer
  {
  public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(size_t count) : count_(count)
    {
      if (count_ == 0)
      {
        return;
      }
      CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_));
    }

    ~PinnedHostBuffer()
    {
      if (ptr_ != nullptr)
      {
        cudaError_t err = cudaFreeHost(ptr_);
        (void)err;
      }
    }

    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;

    T *get() const { return ptr_; }
    size_t size() const { return count_; }

    T &operator[](size_t idx) { return ptr_[idx]; }
    const T &operator[](size_t idx) const { return ptr_[idx]; }

  private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
  };

  // Up scale, round up
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
    WorkspaceArena(size_t bytes, cudaStream_t stream)
        : storage_(bytes, stream), storage_ptr_(storage_.get()), storage_bytes_(bytes)
    {
    }

    WorkspaceArena(uint8_t *storage, size_t bytes) : storage_ptr_(storage), storage_bytes_(bytes) {}

    template <typename T>
    T *Alloc(size_t count)
    {
      size_t aligned_offset = AlignUp(offset_bytes_, static_cast<size_t>(alignof(T)));
      size_t bytes = sizeof(T) * count;
      if (aligned_offset + bytes > storage_bytes_)
      {
        TVM_FFI_THROW(RuntimeError) << "WorkspaceArena overflow: requested " << bytes
                                    << " bytes with " << (storage_bytes_ - offset_bytes_)
                                    << " bytes remaining";
      }
      T *ptr = reinterpret_cast<T *>(storage_ptr_ + aligned_offset);
      offset_bytes_ = aligned_offset + bytes;
      return ptr;
    }

  private:
    AsyncBuffer<uint8_t> storage_;
    uint8_t *storage_ptr_ = nullptr;
    size_t storage_bytes_ = 0;
    size_t offset_bytes_ = 0;
  };

  struct PreparedExpertWorkload
  {
    int32_t *expert_counts = nullptr;
    int32_t *expert_offsets = nullptr;
    int32_t *reordered_tokens = nullptr;
    int32_t *reordered_local_experts = nullptr;
    int32_t *token_route_counts = nullptr;
    int32_t *token_route_rows = nullptr;
    int32_t *token_route_experts = nullptr;
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
  __global__ void sigmoid_bias_kernel(const float *routing_logits,
                                      const __nv_bfloat16 *routing_bias, float *s,
                                      float *s_with_bias, int64_t t)
  {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = t * kGlobalExperts;
    if (idx >= total)
      return;
    int64_t expert = idx % kGlobalExperts;
    float logit = routing_logits[idx];
    float sigmoid = 1.0f / (1.0f + expf(-logit));
    s[idx] = sigmoid;
    s_with_bias[idx] = sigmoid + bf16_to_float(routing_bias[expert]);
  }

  // One CUDA block per token. This mirrors the reference routing:
  // top-2 per group -> top-4 groups -> final top-8 experts -> normalized weights from s.
  // routing_logits is FP32
  // Output: topk_idx [t, 8], weights [t, kGlobalExperts]. 8 expert for the token and their weights.
  // TODO: One block process more tokens
  __global__ void routing_select_kernel(const float *s, const float *s_with_bias, int32_t *topk_idx,
                                        float *weights, int64_t t,
                                        float routed_scaling_factor)
  {
    int64_t token = blockIdx.x; // Actually just one thread work.
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

  // Final cast back to the contest output dtype. FP32 -> BF16
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

  // Reorder token-to-expert assignments into expert-major contiguous rows.
  // row 0 -> (token 0, expert 1)
  // row 1 -> (token 1, expert 1)
  // row 2 -> (token 2, expert 2)
  // row 3 -> (token 3, expert 2)

  __global__ void reorder_tokens_kernel(const int32_t *topk_idx, const int32_t *expert_offsets,
                                        int32_t *expert_current_counts, int32_t *reordered_tokens,
                                        int32_t *reordered_local_experts,
                                        int32_t *token_route_counts,
                                        int32_t *token_route_rows,
                                        int32_t *token_route_experts, int64_t t,
                                        int32_t local_expert_offset)
  {
    // Expand each token's top-k routing decisions into a contiguous expert-major row layout.
    int64_t token_id = blockIdx.x;
    if (token_id >= t)
    {
      return;
    }
    int32_t local_route_count = 0;
    bool build_token_routes = token_route_counts != nullptr;
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
        if (build_token_routes)
        {
          int64_t route_pos = token_id * kTopK + local_route_count;
          token_route_rows[route_pos] = pos;
          token_route_experts[route_pos] = global_expert;
          ++local_route_count;
        }
      }
    }
    if (build_token_routes)
    {
      token_route_counts[token_id] = local_route_count;
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
    size_t max_tokens = static_cast<size_t>(max_rows) / kTopK;
    return sizeof(int32_t) *
               static_cast<size_t>(3 * kLocalExperts + (kLocalExperts + 1) + 1 +
                                   4 * max_rows + max_tokens) +
           scan_workspace_size;
  }

  size_t GetGroupedComputeWorkspaceBytes(int32_t total_rows, size_t grouped_gemm1_count,
                                         size_t grouped_gemm2_count)
  {
    size_t grouped_gemm1_weight_values =
        grouped_gemm1_count * static_cast<size_t>(2 * kIntermediateSize) * kHiddenSize;
    size_t grouped_gemm2_weight_values =
        grouped_gemm2_count * static_cast<size_t>(kHiddenSize) * kIntermediateSize;
    if constexpr (kUseGroupedF16Gemm1)
    {
      return sizeof(float) * static_cast<size_t>(total_rows) *
                 (2 * kIntermediateSize + kIntermediateSize + kHiddenSize) +
             sizeof(cutlass::half_t) * grouped_gemm1_weight_values +
             sizeof(float) * grouped_gemm2_weight_values +
             sizeof(int32_t) * (grouped_gemm1_count + grouped_gemm2_count);
    }
    size_t small_weight_values = 0;
    if constexpr (kGroupedGemm1Threshold > 1)
    {
      small_weight_values += static_cast<size_t>(2 * kIntermediateSize) * kHiddenSize;
    }
    if constexpr (kGroupedGemm2Threshold > 1)
    {
      small_weight_values += static_cast<size_t>(kHiddenSize) * kIntermediateSize;
    }
    return sizeof(float) * static_cast<size_t>(total_rows) *
               (2 * kIntermediateSize + kIntermediateSize + kHiddenSize) +
           sizeof(float) * small_weight_values +
           sizeof(float) * (grouped_gemm1_weight_values + grouped_gemm2_weight_values) +
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

  std::vector<int32_t> BuildExpertOffsetsFromCounts(const std::vector<int32_t> &expert_counts)
  {
    std::vector<int32_t> expert_offsets(kLocalExperts + 1, 0);
    for (int32_t expert = 0; expert < kLocalExperts; ++expert)
    {
      expert_offsets[expert + 1] = expert_offsets[expert] + expert_counts[expert];
    }
    return expert_offsets;
  }

  // ---------------------------------------------------------------------------
  // Stage 2: build the local expert workload.
  // Convert token-level routing outputs into an expert-major row layout that both
  // execution paths consume.
  // ---------------------------------------------------------------------------

  // Build the expert-local workload layout shared by both execution paths:
  // 1) count local assignments
  // 2) exclusive-scan counts into per-expert row offsets
  // 3) reorder (token, expert) pairs into one contiguous expert-major row layout
  PreparedExpertWorkload PrepareExpertWorkload(const int32_t *topk_idx, int64_t t,
                                               int32_t local_expert_offset, int32_t max_rows,
                                               size_t scan_workspace_size,
                                               WorkspaceArena *workspace, cudaStream_t stream,
                                               bool build_token_routes)
  {
    PreparedExpertWorkload prepared;
    prepared.expert_counts = workspace->Alloc<int32_t>(kLocalExperts);
    prepared.expert_offsets = workspace->Alloc<int32_t>(kLocalExperts + 1);
    prepared.reordered_tokens = workspace->Alloc<int32_t>(max_rows);
    prepared.reordered_local_experts = workspace->Alloc<int32_t>(max_rows);
    if (build_token_routes)
    {
      prepared.token_route_counts =
          workspace->Alloc<int32_t>(static_cast<size_t>(max_rows) / kTopK);
      prepared.token_route_rows = workspace->Alloc<int32_t>(max_rows);
      prepared.token_route_experts = workspace->Alloc<int32_t>(max_rows);
    }
    int32_t *tmp_counts = workspace->Alloc<int32_t>(kLocalExperts);
    prepared.total_rows = workspace->Alloc<int32_t>(1);
    uint8_t *scan_workspace = workspace->Alloc<uint8_t>(scan_workspace_size);

    CHECK_CUDA(cudaMemsetAsync(prepared.expert_counts, 0, kLocalExperts * sizeof(int32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(tmp_counts, 0, kLocalExperts * sizeof(int32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(prepared.reordered_tokens, 0xFF,
                               max_rows * sizeof(int32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(prepared.reordered_local_experts, 0xFF,
                               max_rows * sizeof(int32_t), stream));

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
                                               prepared.reordered_local_experts,
                                               prepared.token_route_counts,
                                               prepared.token_route_rows,
                                               prepared.token_route_experts, t,
                                               local_expert_offset);
    CHECK_CUDA(cudaGetLastError());

    return prepared;
  }

  // ---------------------------------------------------------------------------
  // Stage 3: materialize activations and merge results.
  // These kernels transform reordered token rows into dense FP32 activations,
  // apply the SwiGLU nonlinearity, and scatter expert outputs back to tokens.
  // ---------------------------------------------------------------------------

  // Hidden states: token-major, -> expert GEMM: expert-major reordered rows
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

  __global__ void fused_gather_dequant_f16_kernel(const __nv_fp8_e4m3 *src,
                                                  const int32_t *token_idx,
                                                  const float *scale_src,
                                                  cutlass::half_t *dst, int64_t m,
                                                  int64_t scale_t)
  {
    int64_t quad_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_quads = (m * kHiddenSize) / 4;
    if (quad_idx >= total_quads)
    {
      return;
    }
    int64_t idx = quad_idx * 4;
    int64_t row = idx / kHiddenSize;
    int64_t col = idx - row * kHiddenSize;
    int32_t token = token_idx[row];
    if (token < 0)
    {
      reinterpret_cast<__half2 *>(dst + idx)[0] = __float2half2_rn(0.0f);
      reinterpret_cast<__half2 *>(dst + idx + 2)[0] = __float2half2_rn(0.0f);
      return;
    }
    float scale = scale_src[static_cast<int64_t>(col / kBlock) * scale_t + token];
    __nv_fp8x4_e4m3 packed =
        reinterpret_cast<const __nv_fp8x4_e4m3 *>(
            src + static_cast<int64_t>(token) * kHiddenSize + col)[0];
    float4 values = static_cast<float4>(packed);
    reinterpret_cast<__half2 *>(dst + idx)[0] =
        __floats2half2_rn(values.x * scale, values.y * scale);
    reinterpret_cast<__half2 *>(dst + idx + 2)[0] =
        __floats2half2_rn(values.z * scale, values.w * scale);
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

  // Token back
  __global__ void optimized_scatter_add_kernel(const float *o, const int32_t *token_idx,
                                               const int32_t *local_expert_idx,
                                               const float *weights, float *output,
                                               const int32_t *total_rows_ptr,
                                               int32_t local_expert_offset)
  {
    // Scatter reordered expert rows back into token-major output order.
    // Multiple routed rows may target the same token, so accumulation uses atomics.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total_rows = total_rows_ptr[0];
    if (idx >= static_cast<int64_t>(total_rows) * kHiddenSize)
    {
      return;
    }

    int64_t row = idx / kHiddenSize;
    int64_t col = idx % kHiddenSize;
    int32_t token = token_idx[row];
    if (token < 0)
    {
      return;
    }
    int32_t local_expert = local_expert_idx[row];
    int32_t global_expert = local_expert + local_expert_offset;
    float weight = weights[static_cast<int64_t>(token) * kGlobalExperts + global_expert];
    atomicAdd(&output[static_cast<int64_t>(token) * kHiddenSize + col],
              o[row * kHiddenSize + col] * weight);
  }

  __global__ void final_reduce_cast_grouped_kernel(const float *o,
                                                   const int32_t *token_route_counts,
                                                   const int32_t *token_route_rows,
                                                   const int32_t *token_route_experts,
                                                   const float *weights,
                                                   __nv_bfloat16 *output_bf16, int64_t t)
  {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = t * kHiddenSize;
    if (idx >= total)
    {
      return;
    }

    int64_t token = idx / kHiddenSize;
    int64_t col = idx - token * kHiddenSize;
    int64_t route_base = token * kTopK;
    int32_t route_count = token_route_counts[token];
    float acc = 0.0f;
    for (int32_t route = 0; route < route_count; ++route)
    {
      int32_t row = token_route_rows[route_base + route];
      int32_t global_expert = token_route_experts[route_base + route];
      float weight = weights[token * kGlobalExperts + global_expert];
      acc += o[static_cast<int64_t>(row) * kHiddenSize + col] * weight;
    }
    output_bf16[idx] = __float2bfloat16(acc);
  }

  // ---------------------------------------------------------------------------
  // Stage 4: expert compute backends.
  // The device_only path consumes FP8 weights directly inside custom kernels.
  // The grouped_workload path keeps the small-expert SIMT fallback, but sends
  // large experts through compact dequantization plus CUTLASS TensorOp grouped GEMM.
  // ---------------------------------------------------------------------------

  // block-scaled FP8 expert weights -> FP32 matrix 的 kernel
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

  __global__ void dequant_selected_experts_blocks_to_f32_kernel(
      const int32_t *__restrict__ expert_ids, const __nv_fp8_e4m3 *__restrict__ src,
      const float *__restrict__ scale, float *__restrict__ dst, int32_t num_selected, int64_t n,
      int64_t k)
  {
    // One CUDA block owns one 128x128 block-scaled tile for one selected expert.
    int32_t k_block = blockIdx.x;
    int32_t n_block = blockIdx.y;
    int32_t selected_slot = blockIdx.z;
    if (selected_slot >= num_selected)
    {
      return;
    }
    int32_t local_expert = expert_ids[selected_slot];
    int64_t n_blocks = n / kBlock;
    int64_t k_blocks = k / kBlock;
    float tile_scale = scale[static_cast<int64_t>(local_expert) * n_blocks * k_blocks +
                             static_cast<int64_t>(n_block) * k_blocks + k_block];
    int64_t expert_size = n * k;
    int64_t expert_src_base = static_cast<int64_t>(local_expert) * expert_size;
    int64_t expert_dst_base = static_cast<int64_t>(selected_slot) * expert_size;
    int64_t tile_base = static_cast<int64_t>(n_block) * kBlock * k +
                        static_cast<int64_t>(k_block) * kBlock;

    constexpr int32_t tile_quads = static_cast<int32_t>((kBlock * kBlock) / 4);
    for (int32_t quad = threadIdx.x; quad < tile_quads; quad += blockDim.x)
    {
      int32_t linear = quad * 4;
      int64_t row = linear / kBlock;
      int64_t col = linear - row * kBlock;
      int64_t offset = tile_base + row * k + col;
      __nv_fp8x4_e4m3 packed =
          reinterpret_cast<const __nv_fp8x4_e4m3 *>(src + expert_src_base + offset)[0];
      float4 values = static_cast<float4>(packed);
      reinterpret_cast<float4 *>(dst + expert_dst_base + offset)[0] =
          make_float4(values.x * tile_scale, values.y * tile_scale, values.z * tile_scale,
                      values.w * tile_scale);
    }
  }

  __global__ void dequant_selected_experts_blocks_to_f16_kernel(
      const int32_t *__restrict__ expert_ids, const __nv_fp8_e4m3 *__restrict__ src,
      const float *__restrict__ scale, cutlass::half_t *__restrict__ dst, int32_t num_selected,
      int64_t n, int64_t k)
  {
    // One CUDA block owns one 128x128 block-scaled tile for one selected expert.
    int32_t k_block = blockIdx.x;
    int32_t n_block = blockIdx.y;
    int32_t selected_slot = blockIdx.z;
    if (selected_slot >= num_selected)
    {
      return;
    }
    int32_t local_expert = expert_ids[selected_slot];
    int64_t n_blocks = n / kBlock;
    int64_t k_blocks = k / kBlock;
    float tile_scale = scale[static_cast<int64_t>(local_expert) * n_blocks * k_blocks +
                             static_cast<int64_t>(n_block) * k_blocks + k_block];
    int64_t expert_size = n * k;
    int64_t expert_src_base = static_cast<int64_t>(local_expert) * expert_size;
    int64_t expert_dst_base = static_cast<int64_t>(selected_slot) * expert_size;
    int64_t tile_base = static_cast<int64_t>(n_block) * kBlock * k +
                        static_cast<int64_t>(k_block) * kBlock;

    constexpr int32_t tile_quads = static_cast<int32_t>((kBlock * kBlock) / 4);
    for (int32_t quad = threadIdx.x; quad < tile_quads; quad += blockDim.x)
    {
      int32_t linear = quad * 4;
      int64_t row = linear / kBlock;
      int64_t col = linear - row * kBlock;
      int64_t offset = tile_base + row * k + col;
      __nv_fp8x4_e4m3 packed =
          reinterpret_cast<const __nv_fp8x4_e4m3 *>(src + expert_src_base + offset)[0];
      float4 values = static_cast<float4>(packed);
      reinterpret_cast<__half2 *>(dst + expert_dst_base + offset)[0] =
          __floats2half2_rn(values.x * tile_scale, values.y * tile_scale);
      reinterpret_cast<__half2 *>(dst + expert_dst_base + offset + 2)[0] =
          __floats2half2_rn(values.z * tile_scale, values.w * tile_scale);
    }
  }

  // A @ W13^T -> G1
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
    if (expert_count == 0)
    {
      return;
    }
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

  void RunSelectedGroupedGemmF32(const std::vector<int32_t> &experts,
                                 const std::vector<int32_t> &expert_counts,
                                 const std::vector<int32_t> &expert_offsets,
                                 const float *a_base, int a_stride, const float *b_compact,
                                 int b_stride, float *d_base, int d_stride, int n, int k,
                                 cudaStream_t stream)
  {
    int expert_count = static_cast<int>(experts.size());
    if (expert_count == 0)
    {
      return;
    }

    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes(expert_count);
    std::vector<cutlass_grouped_f32::ElementA *> h_ptr_a(expert_count);
    std::vector<cutlass_grouped_f32::ElementB *> h_ptr_b(expert_count);
    std::vector<cutlass_grouped_f32::ElementC *> h_ptr_c(expert_count);
    std::vector<cutlass_grouped_f32::ElementC *> h_ptr_d(expert_count);

    for (int slot = 0; slot < expert_count; ++slot)
    {
      int32_t expert = experts[slot];
      int32_t rows = expert_counts[expert];
      int32_t row_offset = expert_offsets[expert];
      h_problem_sizes[slot] = cutlass::gemm::GemmCoord(rows, n, k);
      h_ptr_a[slot] = const_cast<float *>(a_base + static_cast<int64_t>(row_offset) * a_stride);
      h_ptr_b[slot] = const_cast<float *>(b_compact + static_cast<int64_t>(slot) * b_stride);
      h_ptr_c[slot] = d_base + static_cast<int64_t>(row_offset) * d_stride;
      h_ptr_d[slot] = d_base + static_cast<int64_t>(row_offset) * d_stride;
    }

    AsyncBuffer<cutlass::gemm::GemmCoord> d_problem_sizes(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f32::ElementA *> d_ptr_a(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f32::ElementB *> d_ptr_b(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f32::ElementC *> d_ptr_c(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f32::ElementC *> d_ptr_d(expert_count, stream);

    CHECK_CUDA(cudaMemcpyAsync(d_problem_sizes.get(), h_problem_sizes.data(),
                               expert_count * sizeof(cutlass::gemm::GemmCoord),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_a.get(), h_ptr_a.data(),
                               expert_count * sizeof(cutlass_grouped_f32::ElementA *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_b.get(), h_ptr_b.data(),
                               expert_count * sizeof(cutlass_grouped_f32::ElementB *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_c.get(), h_ptr_c.data(),
                               expert_count * sizeof(cutlass_grouped_f32::ElementC *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_d.get(), h_ptr_d.data(),
                               expert_count * sizeof(cutlass_grouped_f32::ElementC *),
                               cudaMemcpyHostToDevice, stream));

    RunGroupedGemm(d_problem_sizes.get(), h_problem_sizes.data(), d_ptr_a.get(), d_ptr_b.get(),
                   d_ptr_c.get(), d_ptr_d.get(), expert_count, k, n, stream);
  }

  void RunGroupedGemmF16(cutlass::gemm::GemmCoord *d_problem_sizes,
                         cutlass::gemm::GemmCoord *h_problem_sizes,
                         cutlass_grouped_f16::ElementA **ptr_a,
                         cutlass_grouped_f16::ElementB **ptr_b,
                         cutlass_grouped_f16::ElementC **ptr_c,
                         cutlass_grouped_f16::ElementC **ptr_d, int expert_count, int k, int n,
                         cudaStream_t stream)
  {
    if (expert_count == 0)
    {
      return;
    }
    using GemmGrouped = cutlass_grouped_f16::GemmGrouped;
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
    CheckCutlassStatus(gemm_op.initialize(args, workspace.get(), stream), "Grouped F16 GEMM init");
    CheckCutlassStatus(gemm_op.run(stream), "Grouped F16 GEMM run");
  }

  void RunSelectedGroupedGemmF16(const std::vector<int32_t> &experts,
                                 const std::vector<int32_t> &expert_counts,
                                 const std::vector<int32_t> &expert_offsets,
                                 const cutlass::half_t *a_base, int a_stride,
                                 const cutlass::half_t *b_compact, int b_stride, float *d_base,
                                 int d_stride, int n, int k, cudaStream_t stream)
  {
    int expert_count = static_cast<int>(experts.size());
    if (expert_count == 0)
    {
      return;
    }

    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes(expert_count);
    std::vector<cutlass_grouped_f16::ElementA *> h_ptr_a(expert_count);
    std::vector<cutlass_grouped_f16::ElementB *> h_ptr_b(expert_count);
    std::vector<cutlass_grouped_f16::ElementC *> h_ptr_c(expert_count);
    std::vector<cutlass_grouped_f16::ElementC *> h_ptr_d(expert_count);

    for (int slot = 0; slot < expert_count; ++slot)
    {
      int32_t expert = experts[slot];
      int32_t rows = expert_counts[expert];
      int32_t row_offset = expert_offsets[expert];
      h_problem_sizes[slot] = cutlass::gemm::GemmCoord(rows, n, k);
      h_ptr_a[slot] =
          const_cast<cutlass::half_t *>(a_base + static_cast<int64_t>(row_offset) * a_stride);
      h_ptr_b[slot] =
          const_cast<cutlass::half_t *>(b_compact + static_cast<int64_t>(slot) * b_stride);
      h_ptr_c[slot] = d_base + static_cast<int64_t>(row_offset) * d_stride;
      h_ptr_d[slot] = d_base + static_cast<int64_t>(row_offset) * d_stride;
    }

    AsyncBuffer<cutlass::gemm::GemmCoord> d_problem_sizes(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f16::ElementA *> d_ptr_a(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f16::ElementB *> d_ptr_b(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f16::ElementC *> d_ptr_c(expert_count, stream);
    AsyncBuffer<cutlass_grouped_f16::ElementC *> d_ptr_d(expert_count, stream);

    CHECK_CUDA(cudaMemcpyAsync(d_problem_sizes.get(), h_problem_sizes.data(),
                               expert_count * sizeof(cutlass::gemm::GemmCoord),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_a.get(), h_ptr_a.data(),
                               expert_count * sizeof(cutlass_grouped_f16::ElementA *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_b.get(), h_ptr_b.data(),
                               expert_count * sizeof(cutlass_grouped_f16::ElementB *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_c.get(), h_ptr_c.data(),
                               expert_count * sizeof(cutlass_grouped_f16::ElementC *),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_ptr_d.get(), h_ptr_d.data(),
                               expert_count * sizeof(cutlass_grouped_f16::ElementC *),
                               cudaMemcpyHostToDevice, stream));

    RunGroupedGemmF16(d_problem_sizes.get(), h_problem_sizes.data(), d_ptr_a.get(),
                      d_ptr_b.get(), d_ptr_c.get(), d_ptr_d.get(), expert_count, k, n, stream);
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
    bool profile_enabled = ProfileEnabled();
    CpuWallTimer pipeline_timer;
    DeviceOnlyPipelineProfile profile;
    if (profile_enabled)
    {
      pipeline_timer.Start();
      profile.t = t;
      profile.max_rows = max_rows;
    }
    size_t scan_workspace_size = GetExpertScanWorkspaceSize(stream);
    size_t workspace_bytes = GetDeviceOnlyWorkspaceBytes(max_rows, scan_workspace_size);
    static thread_local CachedDeviceBuffer<uint8_t> device_only_workspace_cache;
    WorkspaceArena workspace(device_only_workspace_cache.get(workspace_bytes), workspace_bytes);

    // Step 1. Build the expert-major row layout for all local assignments.
    if (profile_enabled)
    {
      profile.prepare.Start(stream);
    }
    PreparedExpertWorkload prepared =
        PrepareExpertWorkload(topk_idx, t, local_expert_offset, max_rows, scan_workspace_size,
                              &workspace, stream, false);
    if (profile_enabled)
    {
      profile.prepare.Stop(stream);
    }
    // The workspace packs:
    // counts/offsets/reordered rows + A + G1 + C + O for the full local assignment set.
    float *a_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kHiddenSize);
    float *g1_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * 2 * kIntermediateSize);
    float *c_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kIntermediateSize);
    float *o_f32 = workspace.Alloc<float>(static_cast<size_t>(max_rows) * kHiddenSize);

    // Step 2. Gather the routed token rows and dequantize activations to FP32.
    if (profile_enabled)
    {
      profile.gather.Start(stream);
    }
    fused_gather_dequant_f32_kernel<<<(static_cast<int64_t>(max_rows) * kHiddenSize + 255) / 256,
                                      256, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3 *>(hidden_states.data_ptr()), prepared.reordered_tokens,
        static_cast<const float *>(hidden_states_scale.data_ptr()), a_f32, max_rows, scale_t);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.gather.Stop(stream);
    }

    // Step 3. Run the two expert projections with the custom device-only FP8 kernels.
    if (profile_enabled)
    {
      profile.gemm1.Start(stream);
    }
    LaunchGroupedFp8Gemm1<kGemm1TileM, kGemm1TileN, kGemm1TileK>(
        a_f32, prepared.expert_offsets, prepared.expert_counts,
        static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()),
        static_cast<const float *>(gemm1_weights_scale.data_ptr()), g1_f32, stream);
    if (profile_enabled)
    {
      profile.gemm1.Stop(stream);
    }

    // Step 4. Apply SwiGLU between the two projections.
    if (profile_enabled)
    {
      profile.swiglu.Start(stream);
    }
    swiglu_f32_kernel<<<(static_cast<int64_t>(max_rows) * kIntermediateSize + 255) / 256,
                        256, 0, stream>>>(g1_f32, c_f32, prepared.total_rows);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.swiglu.Stop(stream);
      profile.gemm2.Start(stream);
    }

    LaunchGroupedFp8Gemm2<kGemm2TileM, kGemm2TileN, kGemm2TileK>(
        c_f32, prepared.expert_offsets, prepared.expert_counts,
        static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()),
        static_cast<const float *>(gemm2_weights_scale.data_ptr()), o_f32, stream);
    if (profile_enabled)
    {
      profile.gemm2.Stop(stream);
    }

    // Step 5. Scatter the reordered expert rows back into token-major output order.
    if (profile_enabled)
    {
      profile.scatter.Start(stream);
    }
    optimized_scatter_add_kernel<<<(static_cast<int64_t>(max_rows) * kHiddenSize + 255) / 256,
                                   256, 0, stream>>>(
        o_f32, prepared.reordered_tokens, prepared.reordered_local_experts, weights,
        output_fp32, prepared.total_rows, local_expert_offset);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.scatter.Stop(stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      profile.pipeline_wall_ms = pipeline_timer.Stop();
      CHECK_CUDA(cudaMemcpy(&profile.total_rows, prepared.total_rows, sizeof(int32_t),
                            cudaMemcpyDeviceToHost));
      PrintDeviceOnlyPipelineProfile(profile);
    }
  }

  void RunGroupedWorkloadPipeline(
      const ffi::TensorView &hidden_states, const ffi::TensorView &hidden_states_scale,
      const ffi::TensorView &gemm1_weights, const ffi::TensorView &gemm1_weights_scale,
      const ffi::TensorView &gemm2_weights, const ffi::TensorView &gemm2_weights_scale,
      int32_t *topk_idx, float *weights, __nv_bfloat16 *output_bf16, int64_t t,
      int32_t local_expert_offset, cudaStream_t stream)
  {
    // Grouped path: reuse the same workload preparation, compact valid rows, then run
    // non-empty experts through dequantization plus CUTLASS TensorOp grouped GEMM.
    int64_t scale_t = hidden_states_scale.size(1);
    int32_t max_rows = static_cast<int32_t>(t * kTopK);
    bool profile_enabled = ProfileEnabled();
    CpuWallTimer pipeline_timer;
    GroupedWorkloadPipelineProfile profile;
    if (profile_enabled)
    {
      pipeline_timer.Start();
      profile.t = t;
      profile.max_rows = max_rows;
    }
    size_t scan_workspace_size = GetExpertScanWorkspaceSize(stream);
    size_t prep_workspace_bytes = GetGroupedPrepWorkspaceBytes(max_rows, scan_workspace_size);
    static thread_local CachedDeviceBuffer<uint8_t> grouped_prep_workspace_cache;
    WorkspaceArena prep_workspace(grouped_prep_workspace_cache.get(prep_workspace_bytes),
                                  prep_workspace_bytes);

    // Step 1. Build the shared expert-major row layout.
    if (profile_enabled)
    {
      profile.prepare.Start(stream);
    }
    PreparedExpertWorkload prepared =
        PrepareExpertWorkload(topk_idx, t, local_expert_offset, max_rows, scan_workspace_size,
                              &prep_workspace, stream, true);
    if (profile_enabled)
    {
      profile.prepare.Stop(stream);
    }

    // Step 2. Copy only the per-expert row counts to pinned host memory on a side stream.
    // The host classifies experts into the small and tensor-core paths while activation
    // gather/dequantization runs on the main stream.
    CpuWallTimer counts_d2h_timer;
    if (profile_enabled)
    {
      counts_d2h_timer.Start();
    }
    cudaEvent_t workload_ready_event = nullptr;
    cudaStream_t host_copy_stream = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&workload_ready_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaStreamCreateWithFlags(&host_copy_stream, cudaStreamNonBlocking));
    PinnedHostBuffer<int32_t> h_counts_pinned(kLocalExperts);
    CHECK_CUDA(cudaEventRecord(workload_ready_event, stream));
    CHECK_CUDA(cudaStreamWaitEvent(host_copy_stream, workload_ready_event, 0));
    CHECK_CUDA(cudaMemcpyAsync(h_counts_pinned.get(), prepared.expert_counts,
                               kLocalExperts * sizeof(int32_t), cudaMemcpyDeviceToHost,
                               host_copy_stream));

    // Step 3. Wait for the compact expert counts so later stages only touch valid rows.
    CHECK_CUDA(cudaStreamSynchronize(host_copy_stream));
    CHECK_CUDA(cudaEventDestroy(workload_ready_event));
    CHECK_CUDA(cudaStreamDestroy(host_copy_stream));
    CpuWallTimer host_plan_timer;
    if (profile_enabled)
    {
      profile.counts_d2h_ms = counts_d2h_timer.Stop();
      host_plan_timer.Start();
    }

    std::vector<int32_t> h_counts(kLocalExperts);
    for (int32_t expert = 0; expert < kLocalExperts; ++expert)
    {
      h_counts[expert] = h_counts_pinned[expert];
    }
    std::vector<int32_t> h_offsets = BuildExpertOffsetsFromCounts(h_counts);
    int32_t total_reordered = h_offsets[kLocalExperts];
    GroupedExpertPlan grouped_gemm1 = BuildGroupedExpertPlan(h_counts, kGroupedGemm1Threshold);
    GroupedExpertPlan grouped_gemm2 = BuildGroupedExpertPlan(h_counts, kGroupedGemm2Threshold);
    if (profile_enabled)
    {
      profile.host_plan_ms = host_plan_timer.Stop();
      profile.total_rows = total_reordered;
      profile.grouped_gemm1_experts = static_cast<int32_t>(grouped_gemm1.experts.size());
      profile.grouped_gemm2_experts = static_cast<int32_t>(grouped_gemm2.experts.size());
    }

    if (total_reordered == 0)
    {
      if (profile_enabled)
      {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        profile.pipeline_wall_ms = pipeline_timer.Stop();
        PrintGroupedWorkloadPipelineProfile(profile);
      }
      return;
    }

    // Step 4. Gather/dequantize only the valid reordered rows into FP16 for GEMM1. Large
    // workloads have far fewer actual routed rows than the padded t * top_k capacity.
    float *a_f32 = nullptr;
    static thread_local CachedDeviceBuffer<cutlass::half_t> grouped_a_cache;
    cutlass::half_t *a_f16 =
        grouped_a_cache.get(static_cast<size_t>(total_reordered) * kHiddenSize);
    if (profile_enabled)
    {
      profile.gather.Start(stream);
    }
    fused_gather_dequant_f16_kernel<<<
        ((static_cast<int64_t>(total_reordered) * kHiddenSize) / 4 + 255) / 256, 256, 0,
        stream>>>(
        static_cast<const __nv_fp8_e4m3 *>(hidden_states.data_ptr()), prepared.reordered_tokens,
        static_cast<const float *>(hidden_states_scale.data_ptr()), a_f16, total_reordered,
        scale_t);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.gather.Stop(stream);
    }

    size_t compute_workspace_bytes = GetGroupedComputeWorkspaceBytes(
        total_reordered, grouped_gemm1.experts.size(), grouped_gemm2.experts.size());
    static thread_local CachedDeviceBuffer<uint8_t> grouped_compute_workspace_cache;
    WorkspaceArena compute_workspace(grouped_compute_workspace_cache.get(compute_workspace_bytes),
                                     compute_workspace_bytes);
    float *g1_f32 = compute_workspace.Alloc<float>(
        static_cast<size_t>(total_reordered) * 2 * kIntermediateSize);
    float *c_f32 =
        compute_workspace.Alloc<float>(static_cast<size_t>(total_reordered) * kIntermediateSize);
    float *w13_small_tmp = nullptr;
    float *w2_small_tmp = nullptr;
    if constexpr (kGroupedGemm1Threshold > 1)
    {
      w13_small_tmp = compute_workspace.Alloc<float>(2 * kIntermediateSize * kHiddenSize);
    }
    if constexpr (kGroupedGemm2Threshold > 1)
    {
      w2_small_tmp = compute_workspace.Alloc<float>(kHiddenSize * kIntermediateSize);
    }
    cutlass::half_t *w13_grouped_f16 = compute_workspace.Alloc<cutlass::half_t>(
        grouped_gemm1.experts.size() * static_cast<size_t>(2 * kIntermediateSize) * kHiddenSize);
    float *w2_grouped_f32 = compute_workspace.Alloc<float>(
        grouped_gemm2.experts.size() * static_cast<size_t>(kHiddenSize) * kIntermediateSize);
    float *o_grouped_f32 =
        compute_workspace.Alloc<float>(static_cast<size_t>(total_reordered) * kHiddenSize);
    int32_t *d_grouped_experts1 = compute_workspace.Alloc<int32_t>(grouped_gemm1.experts.size());
    int32_t *d_grouped_experts2 = compute_workspace.Alloc<int32_t>(grouped_gemm2.experts.size());

    // Step 5. Execute GEMM1. The SIMT fallback remains compiled for higher thresholds, but the
    // current threshold sends every non-empty expert to compact dequant + CUTLASS TC.
    if (profile_enabled)
    {
      profile.gemm1_small.Start(stream);
    }
    if constexpr (kGroupedGemm1Threshold > 1)
    {
      for (int i = 0; i < kLocalExperts; ++i)
      {
        if (h_counts[i] <= 0)
        {
          continue;
        }
        if (h_counts[i] < kGroupedGemm1Threshold)
        {
          if (profile_enabled)
          {
            ++profile.small_gemm1_experts;
          }
          // Small expert: materialize just one expert's weights and run the SIMT fallback.
          dequant_weights_to_f32_kernel<<<
              (static_cast<int64_t>(2 * kIntermediateSize) * kHiddenSize + 255) / 256, 256, 0,
              stream>>>(static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()) +
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
      }
    }
    if (profile_enabled)
    {
      profile.gemm1_small.Stop(stream);
      profile.grouped_gemm1_dequant.Start(stream);
    }

    if (!grouped_gemm1.experts.empty())
    {
      CHECK_CUDA(cudaMemcpyAsync(d_grouped_experts1, grouped_gemm1.experts.data(),
                                 grouped_gemm1.experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
      dim3 dequant_grid(static_cast<unsigned int>(kHiddenSize / kBlock),
                         static_cast<unsigned int>((2 * kIntermediateSize) / kBlock),
                         static_cast<unsigned int>(grouped_gemm1.experts.size()));
      dequant_selected_experts_blocks_to_f16_kernel<<<dequant_grid, kSelectedDequantThreads, 0,
                                                      stream>>>(
          d_grouped_experts1, static_cast<const __nv_fp8_e4m3 *>(gemm1_weights.data_ptr()),
          static_cast<const float *>(gemm1_weights_scale.data_ptr()), w13_grouped_f16,
          static_cast<int32_t>(grouped_gemm1.experts.size()), 2 * kIntermediateSize,
          kHiddenSize);
      CHECK_CUDA(cudaGetLastError());
    }
    if (profile_enabled)
    {
      profile.grouped_gemm1_dequant.Stop(stream);
      profile.gemm1_grouped.Start(stream);
    }

    if (!grouped_gemm1.experts.empty())
    {
      RunSelectedGroupedGemmF16(grouped_gemm1.experts, h_counts, h_offsets, a_f16, kHiddenSize,
                                w13_grouped_f16, 2 * kIntermediateSize * kHiddenSize, g1_f32,
                                2 * kIntermediateSize, 2 * kIntermediateSize, kHiddenSize, stream);
    }
    if (profile_enabled)
    {
      profile.gemm1_grouped.Stop(stream);
    }

    // Step 6. Apply SwiGLU after GEMM1 completes for all reordered rows.
    if (profile_enabled)
    {
      profile.swiglu.Start(stream);
    }
    swiglu_f32_kernel<<<(static_cast<int64_t>(total_reordered) * kIntermediateSize + 255) / 256,
                        256, 0, stream>>>(g1_f32, c_f32, prepared.total_rows);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.swiglu.Stop(stream);
    }

    // Step 7. Execute GEMM2 with the same grouped-expert policy.
    if (profile_enabled)
    {
      profile.gemm2_small.Start(stream);
    }
    if constexpr (kGroupedGemm2Threshold > 1)
    {
      for (int i = 0; i < kLocalExperts; ++i)
      {
        if (h_counts[i] <= 0)
        {
          continue;
        }
        if (h_counts[i] < kGroupedGemm2Threshold)
        {
          if (profile_enabled)
          {
            ++profile.small_gemm2_experts;
          }
          dequant_weights_to_f32_kernel<<<
              (static_cast<int64_t>(kHiddenSize) * kIntermediateSize + 255) / 256, 256, 0,
              stream>>>(static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()) +
                            static_cast<int64_t>(i) * kHiddenSize * kIntermediateSize,
                        static_cast<const float *>(gemm2_weights_scale.data_ptr()) +
                            static_cast<int64_t>(i) * (kHiddenSize / kBlock) *
                                (kIntermediateSize / kBlock),
                        w2_small_tmp, 1, kHiddenSize, kIntermediateSize);
          CHECK_CUDA(cudaGetLastError());
          RunGemmF32(c_f32 + static_cast<int64_t>(h_offsets[i]) * kIntermediateSize, w2_small_tmp,
                     o_grouped_f32 + static_cast<int64_t>(h_offsets[i]) * kHiddenSize,
                     h_counts[i], kHiddenSize, kIntermediateSize, stream);
        }
      }
    }
    if (profile_enabled)
    {
      profile.gemm2_small.Stop(stream);
      profile.grouped_gemm2_dequant.Start(stream);
    }

    if (!grouped_gemm2.experts.empty())
    {
      CHECK_CUDA(cudaMemcpyAsync(d_grouped_experts2, grouped_gemm2.experts.data(),
                                 grouped_gemm2.experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
      dim3 dequant_grid(static_cast<unsigned int>(kIntermediateSize / kBlock),
                         static_cast<unsigned int>(kHiddenSize / kBlock),
                         static_cast<unsigned int>(grouped_gemm2.experts.size()));
      dequant_selected_experts_blocks_to_f32_kernel<<<dequant_grid, kSelectedDequantThreads, 0,
                                                      stream>>>(
          d_grouped_experts2, static_cast<const __nv_fp8_e4m3 *>(gemm2_weights.data_ptr()),
          static_cast<const float *>(gemm2_weights_scale.data_ptr()), w2_grouped_f32,
          static_cast<int32_t>(grouped_gemm2.experts.size()), kHiddenSize, kIntermediateSize);
      CHECK_CUDA(cudaGetLastError());
    }
    if (profile_enabled)
    {
      profile.grouped_gemm2_dequant.Stop(stream);
      profile.gemm2_grouped.Start(stream);
    }

    if (!grouped_gemm2.experts.empty())
    {
      RunSelectedGroupedGemmF32(grouped_gemm2.experts, h_counts, h_offsets, c_f32,
                                kIntermediateSize, w2_grouped_f32,
                                kHiddenSize * kIntermediateSize, o_grouped_f32, kHiddenSize,
                                kHiddenSize, kIntermediateSize, stream);
    }
    if (profile_enabled)
    {
      profile.gemm2_grouped.Stop(stream);
    }

    // Step 8. Reduce each token's local routed rows and cast directly to the final BF16 output.
    if (profile_enabled)
    {
      profile.scatter.Start(stream);
    }
    final_reduce_cast_grouped_kernel<<<(static_cast<int64_t>(t) * kHiddenSize + 255) / 256, 256,
                                       0, stream>>>(
        o_grouped_f32, prepared.token_route_counts, prepared.token_route_rows,
        prepared.token_route_experts, weights, output_bf16, t);
    CHECK_CUDA(cudaGetLastError());
    if (profile_enabled)
    {
      profile.scatter.Stop(stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      profile.pipeline_wall_ms = pipeline_timer.Stop();
      PrintGroupedWorkloadPipelineProfile(profile);
    }
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
  int64_t output_total = t * kHiddenSize;
  bool use_device_only_path = t <= kHybridDispatchSeqLenThreshold;

  int64_t routing_total = t * kGlobalExperts;
  static thread_local CachedDeviceBuffer<float> s_cache;
  static thread_local CachedDeviceBuffer<float> s_with_bias_cache;
  static thread_local CachedDeviceBuffer<int32_t> topk_idx_cache;
  static thread_local CachedDeviceBuffer<float> weights_cache;
  static thread_local CachedDeviceBuffer<float> output_fp32_cache;
  float *s = s_cache.get(static_cast<size_t>(t) * kGlobalExperts);
  float *s_with_bias = s_with_bias_cache.get(static_cast<size_t>(t) * kGlobalExperts);
  int32_t *topk_idx = topk_idx_cache.get(static_cast<size_t>(t) * kTopK);
  float *weights = weights_cache.get(static_cast<size_t>(t) * kGlobalExperts);
  float *output_fp32 = nullptr;

  if (use_device_only_path)
  {
    output_fp32 = output_fp32_cache.get(static_cast<size_t>(t) * kHiddenSize);
    CHECK_CUDA(cudaMemsetAsync(output_fp32, 0,
                               sizeof(float) * static_cast<size_t>(t) * kHiddenSize, stream));
  }
  CHECK_CUDA(cudaMemsetAsync(weights, 0,
                             sizeof(float) * static_cast<size_t>(t) * kGlobalExperts, stream));

  sigmoid_bias_kernel<<<(routing_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
      static_cast<const float *>(routing_logits.data_ptr()),
      static_cast<const __nv_bfloat16 *>(routing_bias.data_ptr()), s, s_with_bias, t);
  CHECK_CUDA(cudaGetLastError());

  routing_select_kernel<<<t, 1, 0, stream>>>(s, s_with_bias, topk_idx, weights, t,
                                             routed_scaling_factor);
  CHECK_CUDA(cudaGetLastError());

  // Hybrid dispatch: small/medium sequences favor the fully device-side custom path, while
  // large sequences still amortize the host-side planning needed for the split
  // between small-expert SIMT fallback and grouped CUTLASS TensorOp launches.
  if (use_device_only_path)
  {
    RunDeviceOnlyGroupedPipeline(hidden_states, hidden_states_scale, gemm1_weights,
                                 gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                                 topk_idx, weights, output_fp32, t, local_expert_offset, stream);
    cast_output_kernel<<<(output_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
        output_fp32, static_cast<__nv_bfloat16 *>(output_tensor.data_ptr()), output_total);
    CHECK_CUDA(cudaGetLastError());
  }
  else
  {
    RunGroupedWorkloadPipeline(hidden_states, hidden_states_scale, gemm1_weights,
                               gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                               topk_idx, weights,
                               static_cast<__nv_bfloat16 *>(output_tensor.data_ptr()), t,
                               local_expert_offset, stream);
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  return output_tensor;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
