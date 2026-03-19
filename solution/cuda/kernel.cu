#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ffi = tvm::ffi;

namespace {

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
// B200-oriented heuristic tiling: keep stronger row reuse than the 4x64x64 variant while
// still using a deeper K tile than the original baseline.
constexpr int kGemm1TileM = 8;
constexpr int kGemm1TileN = 32;
constexpr int kGemm1TileK = 64;
constexpr int kGemm2TileM = 8;
constexpr int kGemm2TileN = 32;
constexpr int kGemm2TileK = 64;

#define CHECK_CUDA(expr)                                                                       \
  do {                                                                                         \
    cudaError_t err__ = (expr);                                                                \
    if (err__ != cudaSuccess) {                                                                \
      TVM_FFI_THROW(RuntimeError) << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                                   << ": " << cudaGetErrorString(err__);                     \
    }                                                                                          \
  } while (0)

void CheckTensor(const ffi::TensorView& tensor, const char* name, int32_t ndim,
                 ffi::ShapeView expected_shape, DLDataType expected_dtype) {
  if (tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(ValueError) << name << " must be a CUDA tensor";
  }
  if (!tensor.IsContiguous()) {
    TVM_FFI_THROW(ValueError) << name << " must be contiguous";
  }
  if (tensor.ndim() != ndim) {
    TVM_FFI_THROW(ValueError) << name << " must be " << ndim << "D, got " << tensor.ndim();
  }
  for (int32_t i = 0; i < ndim; ++i) {
    if (tensor.size(i) != expected_shape[i]) {
      TVM_FFI_THROW(ValueError) << name << " shape mismatch at dim " << i << ": expected "
                                << expected_shape[i] << ", got " << tensor.size(i);
    }
  }
  DLDataType dtype = tensor.dtype();
  if (dtype.code != expected_dtype.code || dtype.bits != expected_dtype.bits ||
      dtype.lanes != expected_dtype.lanes) {
    TVM_FFI_THROW(TypeError) << name << " dtype mismatch: expected "
                             << ffi::DLDataTypeToString(expected_dtype) << ", got "
                             << ffi::DLDataTypeToString(dtype);
  }
}

void CheckScalarVector(const ffi::TensorView& tensor, const char* name, int64_t size,
                       DLDataType expected_dtype) {
  CheckTensor(tensor, name, 1, ffi::ShapeView({size}), expected_dtype);
}

void CheckSameDevice(const ffi::TensorView& a, const ffi::TensorView& b, const char* a_name,
                     const char* b_name) {
  if (a.device().device_type != b.device().device_type ||
      a.device().device_id != b.device().device_id) {
    TVM_FFI_THROW(ValueError) << a_name << " and " << b_name << " must be on the same device";
  }
}

template <typename T>
class AsyncBuffer {
 public:
  AsyncBuffer() = default;

  AsyncBuffer(size_t count, cudaStream_t stream) : count_(count), stream_(stream) {
    if (count_ == 0) {
      return;
    }
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_, stream_));
  }

  ~AsyncBuffer() {
    if (ptr_ != nullptr) {
      cudaError_t err = cudaFreeAsync(ptr_, stream_);
      (void)err;
    }
  }

  AsyncBuffer(const AsyncBuffer&) = delete;
  AsyncBuffer& operator=(const AsyncBuffer&) = delete;

  AsyncBuffer(AsyncBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_), stream_(other.stream_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
    other.stream_ = nullptr;
  }

  AsyncBuffer& operator=(AsyncBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
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

  T* get() const { return ptr_; }
  size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
  cudaStream_t stream_ = nullptr;
};

__device__ inline float fp8_to_float(__nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}

__device__ inline float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// hidden_states: [T, H], scale: [H/128, T] (transposed layout in the dataset)
__global__ void dequant_hidden_states_kernel(const __nv_fp8_e4m3* hidden_states,
                                             const float* hidden_states_scale, float* a_fp32,
                                             int64_t t) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * kHiddenSize;
  if (idx >= total) {
    return;
  }
  int64_t token = idx / kHiddenSize;
  int64_t hidden = idx % kHiddenSize;
  int64_t block = hidden / kBlock;
  float scale = hidden_states_scale[block * t + token];
  a_fp32[idx] = fp8_to_float(hidden_states[idx]) * scale;
}

// Compute s = sigmoid(logits) and s_with_bias = s + bias.
__global__ void sigmoid_bias_kernel(const float* routing_logits, const __nv_bfloat16* routing_bias,
                                    float* s, float* s_with_bias, int64_t t) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * kGlobalExperts;
  if (idx >= total) {
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
__global__ void routing_select_kernel(const float* s, const float* s_with_bias, int32_t* topk_idx,
                                      float* weights, int64_t t, float routed_scaling_factor) {
  int64_t token = blockIdx.x;
  if (token >= t || threadIdx.x != 0) {
    return;
  }

  const float* s_row = s + token * kGlobalExperts;
  const float* s_with_bias_row = s_with_bias + token * kGlobalExperts;
  int32_t* topk_row = topk_idx + token * kTopK;
  float* weight_row = weights + token * kGlobalExperts;

  float group_scores[kNumGroups];
  for (int group = 0; group < kNumGroups; ++group) {
    float best0 = -INFINITY;
    float best1 = -INFINITY;
    for (int lane = 0; lane < kGroupSize; ++lane) {
      float value = s_with_bias_row[group * kGroupSize + lane];
      if (value > best0) {
        best1 = best0;
        best0 = value;
      } else if (value > best1) {
        best1 = value;
      }
    }
    group_scores[group] = best0 + best1;
  }

  bool keep_group[kNumGroups] = {false};
  for (int pick = 0; pick < kTopKGroups; ++pick) {
    int best_group = -1;
    float best_score = -INFINITY;
    for (int group = 0; group < kNumGroups; ++group) {
      if (!keep_group[group] && group_scores[group] > best_score) {
        best_score = group_scores[group];
        best_group = group;
      }
    }
    keep_group[best_group] = true;
  }

  int32_t selected_idx[kTopK];
  float selected_score[kTopK];
  for (int k = 0; k < kTopK; ++k) {
    selected_idx[k] = -1;
    selected_score[k] = -INFINITY;
  }

  for (int expert = 0; expert < kGlobalExperts; ++expert) {
    if (!keep_group[expert / kGroupSize]) {
      continue;
    }
    float value = s_with_bias_row[expert];
    int insert_at = -1;
    for (int k = 0; k < kTopK; ++k) {
      if (value > selected_score[k]) {
        insert_at = k;
        break;
      }
    }
    if (insert_at == -1) {
      continue;
    }
    for (int k = kTopK - 1; k > insert_at; --k) {
      selected_score[k] = selected_score[k - 1];
      selected_idx[k] = selected_idx[k - 1];
    }
    selected_score[insert_at] = value;
    selected_idx[insert_at] = expert;
  }

  for (int expert = 0; expert < kGlobalExperts; ++expert) {
    weight_row[expert] = 0.0f;
  }

  // Combination weights use s (without bias), matching the PyTorch reference.
  float weight_sum = 0.0f;
  for (int k = 0; k < kTopK; ++k) {
    topk_row[k] = selected_idx[k];
    if (selected_idx[k] >= 0) {
      weight_sum += s_row[selected_idx[k]];
    }
  }
  weight_sum += 1e-20f;

  for (int k = 0; k < kTopK; ++k) {
    int32_t expert = selected_idx[k];
    if (expert >= 0) {
      weight_row[expert] = (s_row[expert] / weight_sum) * routed_scaling_factor;
    }
  }
}

// Gather token rows for one expert's token list.
__global__ void gather_rows_kernel(const float* src, const int32_t* token_idx, float* dst,
                                   int64_t rows, int64_t width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * width;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / width;
  int64_t col = idx % width;
  int32_t token = token_idx[row];
  dst[idx] = src[static_cast<int64_t>(token) * width + col];
}

// GEMM1 specialized for A: [Tk, H], W13: [2I, H] stored as FP8 block-scaled weights.
// We tile along M/N/K, stage A and on-the-fly dequantized W13 into shared memory, and
// accumulate in FP32 without materializing the full dequantized W13 tensor.
__global__ void gemm1_tiled_fused_w13_kernel(const float* __restrict__ a,
                                             const __nv_fp8_e4m3* __restrict__ w13_fp8,
                                             const float* __restrict__ w13_scale,
                                             float* __restrict__ g1, int64_t m) {
  __shared__ float a_tile[kGemm1TileM][kGemm1TileK];
  __shared__ float b_tile[kGemm1TileN][kGemm1TileK];

  int local_col = threadIdx.x;
  int local_row = threadIdx.y;
  int tid = local_row * blockDim.x + local_col;
  int row = blockIdx.y * kGemm1TileM + local_row;
  int col = blockIdx.x * kGemm1TileN + local_col;

  float acc = 0.0f;

  for (int64_t k0 = 0; k0 < kHiddenSize; k0 += kGemm1TileK) {
    for (int idx = tid; idx < kGemm1TileM * kGemm1TileK; idx += blockDim.x * blockDim.y) {
      int tile_row = idx / kGemm1TileK;
      int tile_k = idx % kGemm1TileK;
      int global_row = blockIdx.y * kGemm1TileM + tile_row;
      int64_t global_k = k0 + tile_k;
      if (global_row < m && global_k < kHiddenSize) {
        a_tile[tile_row][tile_k] = a[static_cast<int64_t>(global_row) * kHiddenSize + global_k];
      } else {
        a_tile[tile_row][tile_k] = 0.0f;
      }
    }

    for (int idx = tid; idx < kGemm1TileN * kGemm1TileK; idx += blockDim.x * blockDim.y) {
      int tile_col = idx / kGemm1TileK;
      int tile_k = idx % kGemm1TileK;
      int global_col = blockIdx.x * kGemm1TileN + tile_col;
      int64_t global_k = k0 + tile_k;
      if (global_col < 2 * kIntermediateSize && global_k < kHiddenSize) {
        int64_t weight_idx =
            static_cast<int64_t>(global_col) * kHiddenSize + global_k;
        int64_t scale_idx = static_cast<int64_t>(global_col / kBlock) * (kHiddenSize / kBlock) +
                            (global_k / kBlock);
        b_tile[tile_col][tile_k] = fp8_to_float(w13_fp8[weight_idx]) * w13_scale[scale_idx];
      } else {
        b_tile[tile_col][tile_k] = 0.0f;
      }
    }

    __syncthreads();

    if (row < m && col < 2 * kIntermediateSize) {
      #pragma unroll
      for (int kk = 0; kk < kGemm1TileK; ++kk) {
        acc += a_tile[local_row][kk] * b_tile[local_col][kk];
      }
    }

    __syncthreads();
  }

  if (row < m && col < 2 * kIntermediateSize) {
    g1[static_cast<int64_t>(row) * (2 * kIntermediateSize) + col] = acc;
  }
}

// GEMM2 specialized for C: [Tk, I], W2: [H, I] stored as FP8 block-scaled weights.
// This mirrors GEMM1: tile C and on-the-fly dequantized W2 into shared memory, then
// accumulate O = C @ W2^T in FP32 without materializing the full dequantized W2 tensor.
__global__ void gemm2_tiled_fused_w2_kernel(const float* __restrict__ c,
                                            const __nv_fp8_e4m3* __restrict__ w2_fp8,
                                            const float* __restrict__ w2_scale,
                                            float* __restrict__ o, int64_t m) {
  __shared__ float c_tile[kGemm2TileM][kGemm2TileK];
  __shared__ float b_tile[kGemm2TileN][kGemm2TileK];

  int local_col = threadIdx.x;
  int local_row = threadIdx.y;
  int tid = local_row * blockDim.x + local_col;
  int row = blockIdx.y * kGemm2TileM + local_row;
  int col = blockIdx.x * kGemm2TileN + local_col;

  float acc = 0.0f;

  for (int64_t k0 = 0; k0 < kIntermediateSize; k0 += kGemm2TileK) {
    for (int idx = tid; idx < kGemm2TileM * kGemm2TileK; idx += blockDim.x * blockDim.y) {
      int tile_row = idx / kGemm2TileK;
      int tile_k = idx % kGemm2TileK;
      int global_row = blockIdx.y * kGemm2TileM + tile_row;
      int64_t global_k = k0 + tile_k;
      if (global_row < m && global_k < kIntermediateSize) {
        c_tile[tile_row][tile_k] =
            c[static_cast<int64_t>(global_row) * kIntermediateSize + global_k];
      } else {
        c_tile[tile_row][tile_k] = 0.0f;
      }
    }

    for (int idx = tid; idx < kGemm2TileN * kGemm2TileK; idx += blockDim.x * blockDim.y) {
      int tile_col = idx / kGemm2TileK;
      int tile_k = idx % kGemm2TileK;
      int global_col = blockIdx.x * kGemm2TileN + tile_col;
      int64_t global_k = k0 + tile_k;
      if (global_col < kHiddenSize && global_k < kIntermediateSize) {
        int64_t weight_idx = static_cast<int64_t>(global_col) * kIntermediateSize + global_k;
        int64_t scale_idx = static_cast<int64_t>(global_col / kBlock) *
                                (kIntermediateSize / kBlock) +
                            (global_k / kBlock);
        b_tile[tile_col][tile_k] = fp8_to_float(w2_fp8[weight_idx]) * w2_scale[scale_idx];
      } else {
        b_tile[tile_col][tile_k] = 0.0f;
      }
    }

    __syncthreads();

    if (row < m && col < kHiddenSize) {
      #pragma unroll
      for (int kk = 0; kk < kGemm2TileK; ++kk) {
        acc += c_tile[local_row][kk] * b_tile[local_col][kk];
      }
    }

    __syncthreads();
  }

  if (row < m && col < kHiddenSize) {
    o[static_cast<int64_t>(row) * kHiddenSize + col] = acc;
  }
}

// Split G1 into X1 and X2, apply SiLU(X2), then multiply by X1.
__global__ void swiglu_kernel(const float* g1, float* c, int64_t rows) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * kIntermediateSize;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kIntermediateSize;
  int64_t col = idx % kIntermediateSize;
  float x1 = g1[row * (2 * kIntermediateSize) + col];
  float x2 = g1[row * (2 * kIntermediateSize) + kIntermediateSize + col];
  float silu = x2 / (1.0f + expf(-x2));
  c[idx] = x1 * silu;
}

// Multiply the expert output by the token's routing weight, then scatter-add into output.
__global__ void weighted_scatter_add_kernel(const float* o, const int32_t* token_idx,
                                            const float* weights, float* output, int64_t rows,
                                            int32_t global_expert) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * kHiddenSize;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kHiddenSize;
  int64_t col = idx % kHiddenSize;
  int32_t token = token_idx[row];
  float weight = weights[static_cast<int64_t>(token) * kGlobalExperts + global_expert];
  output[static_cast<int64_t>(token) * kHiddenSize + col] += o[idx] * weight;
}

// Final cast back to the contest output dtype.
__global__ void cast_output_kernel(const float* output_fp32, __nv_bfloat16* output_bf16,
                                   int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output_bf16[idx] = __float2bfloat16(output_fp32[idx]);
}

}  // namespace

ffi::Tensor kernel(const ffi::TensorView& routing_logits, const ffi::TensorView& routing_bias,
                   const ffi::TensorView& hidden_states,
                   const ffi::TensorView& hidden_states_scale,
                   const ffi::TensorView& gemm1_weights,
                   const ffi::TensorView& gemm1_weights_scale,
                   const ffi::TensorView& gemm2_weights,
                   const ffi::TensorView& gemm2_weights_scale, int32_t local_expert_offset,
                   float routed_scaling_factor) {
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

  // Materialize the same float32 intermediates as the PyTorch reference.
  AsyncBuffer<float> a_fp32(static_cast<size_t>(t) * kHiddenSize, stream);
  AsyncBuffer<float> s(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<float> s_with_bias(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<int32_t> topk_idx(static_cast<size_t>(t) * kTopK, stream);
  AsyncBuffer<float> weights(static_cast<size_t>(t) * kGlobalExperts, stream);
  AsyncBuffer<float> output_fp32(static_cast<size_t>(t) * kHiddenSize, stream);

  CHECK_CUDA(cudaMemsetAsync(output_fp32.get(), 0, sizeof(float) * static_cast<size_t>(t) *
                                                  kHiddenSize, stream));

  constexpr int threads_1d = 256;
  int64_t hidden_total = t * kHiddenSize;
  int64_t routing_total = t * kGlobalExperts;

  // 1) FP8 block-scale dequantization
  dequant_hidden_states_kernel<<<(hidden_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                                 stream>>>(
      static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      static_cast<const float*>(hidden_states_scale.data_ptr()), a_fp32.get(), t);
  CHECK_CUDA(cudaGetLastError());

  // 2) No-aux routing
  sigmoid_bias_kernel<<<(routing_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
      static_cast<const float*>(routing_logits.data_ptr()),
      static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()), s.get(), s_with_bias.get(), t);
  CHECK_CUDA(cudaGetLastError());

  routing_select_kernel<<<t, 1, 0, stream>>>(s.get(), s_with_bias.get(), topk_idx.get(),
                                              weights.get(), t, routed_scaling_factor);
  CHECK_CUDA(cudaGetLastError());

  std::vector<int32_t> host_topk(static_cast<size_t>(t) * kTopK);
  CHECK_CUDA(cudaMemcpyAsync(host_topk.data(), topk_idx.get(),
                             sizeof(int32_t) * static_cast<size_t>(t) * kTopK,
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Build the per-local-expert token lists on the host, mirroring the reference's
  // per-expert nonzero/index_select flow.
  std::vector<std::vector<int32_t>> token_lists(kLocalExperts);
  for (int64_t token = 0; token < t; ++token) {
    for (int64_t k = 0; k < kTopK; ++k) {
      int32_t global_expert = host_topk[static_cast<size_t>(token) * kTopK + k];
      int64_t local_expert = static_cast<int64_t>(global_expert) - local_expert_offset;
      if (0 <= local_expert && local_expert < kLocalExperts) {
        token_lists[local_expert].push_back(static_cast<int32_t>(token));
      }
    }
  }

  int64_t max_tk = 0;
  for (const auto& tokens : token_lists) {
    if (static_cast<int64_t>(tokens.size()) > max_tk) {
      max_tk = static_cast<int64_t>(tokens.size());
    }
  }

  AsyncBuffer<int32_t> token_idx_device(max_tk, stream);
  AsyncBuffer<float> a_e(static_cast<size_t>(max_tk) * kHiddenSize, stream);
  AsyncBuffer<float> g1(static_cast<size_t>(max_tk) * (2 * kIntermediateSize), stream);
  AsyncBuffer<float> c(static_cast<size_t>(max_tk) * kIntermediateSize, stream);
  AsyncBuffer<float> o(static_cast<size_t>(max_tk) * kHiddenSize, stream);

  // 3) Local expert compute and accumulation
  for (int64_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    int32_t global_expert = static_cast<int32_t>(local_expert_offset + local_expert);
    if (global_expert < 0 || global_expert >= kGlobalExperts) {
      continue;
    }

    const std::vector<int32_t>& tokens = token_lists[local_expert];
    if (tokens.empty()) {
      continue;
    }

    // Gather inputs for this expert, then run GEMM1 -> SwiGLU -> GEMM2.
    int64_t tk = static_cast<int64_t>(tokens.size());
    CHECK_CUDA(cudaMemcpyAsync(token_idx_device.get(), tokens.data(), sizeof(int32_t) * tk,
                               cudaMemcpyHostToDevice, stream));

    int64_t gather_total = tk * kHiddenSize;
    gather_rows_kernel<<<(gather_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
        a_fp32.get(), token_idx_device.get(), a_e.get(), tk, kHiddenSize);
    CHECK_CUDA(cudaGetLastError());

    const __nv_fp8_e4m3* w13_expert =
        static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()) +
        local_expert * static_cast<int64_t>(2 * kIntermediateSize) * kHiddenSize;
    const float* w13_scale_expert =
        static_cast<const float*>(gemm1_weights_scale.data_ptr()) +
        local_expert * static_cast<int64_t>(num_gemm1_out_blocks) * num_hidden_blocks;
    dim3 gemm1_block(kGemm1TileN, kGemm1TileM);
    dim3 gemm1_grid((2 * kIntermediateSize + kGemm1TileN - 1) / kGemm1TileN,
                    (tk + kGemm1TileM - 1) / kGemm1TileM);
    gemm1_tiled_fused_w13_kernel<<<gemm1_grid, gemm1_block, 0, stream>>>(
        a_e.get(), w13_expert, w13_scale_expert, g1.get(), tk);
    CHECK_CUDA(cudaGetLastError());

    int64_t swiglu_total = tk * kIntermediateSize;
    swiglu_kernel<<<(swiglu_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
        g1.get(), c.get(), tk);
    CHECK_CUDA(cudaGetLastError());

    const __nv_fp8_e4m3* w2_expert =
        static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr()) +
        local_expert * static_cast<int64_t>(kHiddenSize) * kIntermediateSize;
    const float* w2_scale_expert =
        static_cast<const float*>(gemm2_weights_scale.data_ptr()) +
        local_expert * static_cast<int64_t>(num_hidden_blocks) * num_intermediate_blocks;
    dim3 gemm2_block(kGemm2TileN, kGemm2TileM);
    dim3 gemm2_grid((kHiddenSize + kGemm2TileN - 1) / kGemm2TileN,
                    (tk + kGemm2TileM - 1) / kGemm2TileM);
    gemm2_tiled_fused_w2_kernel<<<gemm2_grid, gemm2_block, 0, stream>>>(
        c.get(), w2_expert, w2_scale_expert, o.get(), tk);
    CHECK_CUDA(cudaGetLastError());

    int64_t scatter_total = tk * kHiddenSize;
    weighted_scatter_add_kernel<<<(scatter_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                                  stream>>>(o.get(), token_idx_device.get(), weights.get(),
                                             output_fp32.get(), tk, global_expert);
    CHECK_CUDA(cudaGetLastError());
  }

  int64_t output_total = t * kHiddenSize;
  cast_output_kernel<<<(output_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
      output_fp32.get(), static_cast<__nv_bfloat16*>(output_tensor.data_ptr()), output_total);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  return output_tensor;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
