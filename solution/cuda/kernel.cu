#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "False") != 0 && std::strcmp(value, "FALSE") != 0;
}

struct DebugOptions {
  bool histogram = false;
  bool timing = false;
};

DebugOptions GetDebugOptions() {
  return {
      .histogram = EnvFlagEnabled("MOE_DEBUG_HISTOGRAM"),
      .timing = EnvFlagEnabled("MOE_DEBUG_TIMING"),
  };
}

bool ShouldPrintDebugOnce(bool enabled, bool* printed_flag) {
  if (!enabled || printed_flag == nullptr || *printed_flag) {
    return false;
  }
  *printed_flag = true;
  return true;
}

class ScopedCudaTimer {
 public:
  ScopedCudaTimer(cudaStream_t stream, float* accum_ms) : stream_(stream), accum_ms_(accum_ms) {
    if (accum_ms_ == nullptr) {
      return;
    }
    if (cudaEventCreate(&start_) != cudaSuccess) {
      start_ = nullptr;
      return;
    }
    if (cudaEventCreate(&stop_) != cudaSuccess) {
      cudaEventDestroy(start_);
      start_ = nullptr;
      stop_ = nullptr;
      return;
    }
    if (cudaEventRecord(start_, stream_) != cudaSuccess) {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
      start_ = nullptr;
      stop_ = nullptr;
    }
  }

  ~ScopedCudaTimer() {
    if (start_ == nullptr || stop_ == nullptr || accum_ms_ == nullptr) {
      return;
    }
    if (cudaEventRecord(stop_, stream_) == cudaSuccess &&
        cudaEventSynchronize(stop_) == cudaSuccess) {
      float elapsed_ms = 0.0f;
      if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) == cudaSuccess) {
        *accum_ms_ += elapsed_ms;
      }
    }
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  cudaStream_t stream_ = nullptr;
  float* accum_ms_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class ScopedHostTimer {
 public:
  explicit ScopedHostTimer(double* accum_ms) : accum_ms_(accum_ms) {
    if (accum_ms_ != nullptr) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedHostTimer() {
    if (accum_ms_ == nullptr) {
      return;
    }
    auto end = std::chrono::steady_clock::now();
    *accum_ms_ +=
        std::chrono::duration<double, std::milli>(end - start_).count();
  }

 private:
  double* accum_ms_ = nullptr;
  std::chrono::steady_clock::time_point start_{};
};

struct ExpertBucket {
  size_t begin;
  size_t end;
  int64_t rows_per_expert;
  int64_t min_rows;
  int64_t max_rows;
};

std::vector<ExpertBucket> BuildExpertBuckets(const std::vector<int32_t>& active_experts,
                                             const std::vector<std::vector<int32_t>>& token_lists) {
  std::vector<ExpertBucket> expert_buckets;
  constexpr int64_t kMaxBucketExperts = 8;
  for (size_t begin = 0; begin < active_experts.size();) {
    int64_t min_tk = static_cast<int64_t>(token_lists[active_experts[begin]].size());
    int64_t bucket_tk = min_tk;
    size_t end = begin + 1;
    int64_t allowed_tk =
        (min_tk <= 64) ? min_tk : (min_tk + std::max<int64_t>(8, min_tk / 8));
    while (end < active_experts.size() &&
           static_cast<int64_t>(end - begin) < kMaxBucketExperts) {
      int64_t next_tk = static_cast<int64_t>(token_lists[active_experts[end]].size());
      if (next_tk > allowed_tk) {
        break;
      }
      bucket_tk = next_tk;
      ++end;
    }
    expert_buckets.push_back({begin, end, bucket_tk, min_tk, bucket_tk});
    begin = end;
  }
  return expert_buckets;
}

void PrintTokenHistogramSummary(int64_t seq_len, int32_t local_expert_offset,
                                const std::vector<std::vector<int32_t>>& token_lists) {
  int64_t total_assignments = 0;
  int64_t min_rows = std::numeric_limits<int64_t>::max();
  int64_t max_rows = 0;
  int64_t active_experts = 0;
  std::vector<std::pair<int64_t, int32_t>> counts;
  counts.reserve(kLocalExperts);
  for (int32_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    int64_t rows = static_cast<int64_t>(token_lists[local_expert].size());
    total_assignments += rows;
    min_rows = std::min(min_rows, rows);
    max_rows = std::max(max_rows, rows);
    if (rows > 0) {
      ++active_experts;
    }
    counts.push_back({rows, local_expert});
  }
  if (min_rows == std::numeric_limits<int64_t>::max()) {
    min_rows = 0;
  }
  std::sort(counts.begin(), counts.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });

  std::fprintf(stderr,
               "[MOE_DEBUG] histogram seq_len=%lld local_offset=%d total_local_assignments=%lld "
               "active_local_experts=%lld avg_rows=%.2f min_rows=%lld max_rows=%lld\n",
               static_cast<long long>(seq_len), static_cast<int>(local_expert_offset),
               static_cast<long long>(total_assignments), static_cast<long long>(active_experts),
               static_cast<double>(total_assignments) / static_cast<double>(kLocalExperts),
               static_cast<long long>(min_rows), static_cast<long long>(max_rows));
  std::fprintf(stderr, "[MOE_DEBUG] local_counts");
  for (int32_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    std::fprintf(stderr, " %d:%lld", static_cast<int>(local_expert_offset + local_expert),
                 static_cast<long long>(token_lists[local_expert].size()));
  }
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "[MOE_DEBUG] top_local_experts");
  for (size_t i = 0; i < std::min<size_t>(5, counts.size()) && counts[i].first > 0; ++i) {
    std::fprintf(stderr, " %d:%lld",
                 static_cast<int>(local_expert_offset + counts[i].second),
                 static_cast<long long>(counts[i].first));
  }
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

void PrintBucketSummary(int64_t seq_len, int32_t local_expert_offset,
                        const std::vector<ExpertBucket>& expert_buckets,
                        const std::vector<int32_t>& active_experts,
                        const std::vector<std::vector<int32_t>>& token_lists) {
  int64_t total_real_rows = 0;
  int64_t total_bucket_rows = 0;
  std::fprintf(stderr, "[MOE_DEBUG] bucket_summary seq_len=%lld local_offset=%d buckets=%zu\n",
               static_cast<long long>(seq_len), static_cast<int>(local_expert_offset),
               expert_buckets.size());
  for (size_t bucket_idx = 0; bucket_idx < expert_buckets.size(); ++bucket_idx) {
    const ExpertBucket& bucket = expert_buckets[bucket_idx];
    int64_t bucket_experts = static_cast<int64_t>(bucket.end - bucket.begin);
    int64_t real_rows = 0;
    std::fprintf(stderr,
                 "[MOE_DEBUG] bucket[%zu] experts=%lld rows_per_expert=%lld min_rows=%lld "
                 "max_rows=%lld global_ids=",
                 bucket_idx, static_cast<long long>(bucket_experts),
                 static_cast<long long>(bucket.rows_per_expert),
                 static_cast<long long>(bucket.min_rows),
                 static_cast<long long>(bucket.max_rows));
    for (size_t idx = bucket.begin; idx < bucket.end; ++idx) {
      int32_t local_expert = active_experts[idx];
      int64_t rows = static_cast<int64_t>(token_lists[local_expert].size());
      real_rows += rows;
      std::fprintf(stderr, "%s%d(%lld)", (idx == bucket.begin ? "" : ","),
                   static_cast<int>(local_expert_offset + local_expert),
                   static_cast<long long>(rows));
    }
    int64_t padded_rows = bucket_experts * bucket.rows_per_expert;
    int64_t pad_rows = padded_rows - real_rows;
    total_real_rows += real_rows;
    total_bucket_rows += padded_rows;
    std::fprintf(stderr, " pad_rows=%lld\n", static_cast<long long>(pad_rows));
  }
  int64_t total_pad_rows = total_bucket_rows - total_real_rows;
  double pad_ratio =
      total_real_rows == 0 ? 0.0
                           : static_cast<double>(total_pad_rows) / static_cast<double>(total_real_rows);
  std::fprintf(stderr,
               "[MOE_DEBUG] bucket_totals padded_rows=%lld real_rows=%lld pad_rows=%lld "
               "pad_ratio=%.4f\n",
               static_cast<long long>(total_bucket_rows), static_cast<long long>(total_real_rows),
               static_cast<long long>(total_pad_rows), pad_ratio);
  std::fflush(stderr);
}

struct DebugTimings {
  float dequant_ms = 0.0f;
  float routing_ms = 0.0f;
  float topk_copy_ms = 0.0f;
  double token_list_host_ms = 0.0;
  double bucket_pack_host_ms = 0.0;
  float bucket_upload_ms = 0.0f;
  float gather_ms = 0.0f;
  float gemm1_ms = 0.0f;
  float swiglu_ms = 0.0f;
  float gemm2_ms = 0.0f;
  float scatter_ms = 0.0f;
  float cast_ms = 0.0f;

  void Print(int64_t seq_len, int32_t local_expert_offset) const {
    double total_ms = static_cast<double>(dequant_ms) + routing_ms + topk_copy_ms +
                      token_list_host_ms + bucket_pack_host_ms + bucket_upload_ms + gather_ms +
                      gemm1_ms + swiglu_ms + gemm2_ms + scatter_ms + cast_ms;
    std::fprintf(stderr,
                 "[MOE_DEBUG] timing seq_len=%lld local_offset=%d total_ms=%.3f\n",
                 static_cast<long long>(seq_len), static_cast<int>(local_expert_offset), total_ms);
    std::fprintf(stderr,
                 "[MOE_DEBUG] stages dequant=%.3f routing=%.3f topk_copy=%.3f "
                 "token_list_host=%.3f bucket_pack_host=%.3f bucket_upload=%.3f "
                 "gather=%.3f gemm1=%.3f swiglu=%.3f gemm2=%.3f scatter=%.3f cast=%.3f\n",
                 dequant_ms, routing_ms, topk_copy_ms, token_list_host_ms, bucket_pack_host_ms,
                 bucket_upload_ms, gather_ms, gemm1_ms, swiglu_ms, gemm2_ms, scatter_ms,
                 cast_ms);
    std::fprintf(stderr,
                 "[MOE_DEBUG] note timing mode synchronizes per stage and perturbs performance\n");
    std::fflush(stderr);
  }
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
__global__ void gather_rows_vec4_kernel(const float4* src, const int32_t* token_idx, float4* dst,
                                        int64_t rows, int64_t vec_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * vec_width;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / vec_width;
  int64_t col = idx % vec_width;
  int32_t token = token_idx[row];
  if (token < 0) {
    dst[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  } else {
    dst[idx] = src[static_cast<int64_t>(token) * vec_width + col];
  }
}

// GEMM1 specialized for A: [Tk, H], W13: [2I, H] stored as FP8 block-scaled weights.
// We tile along M/N/K, stage A and on-the-fly dequantized W13 into shared memory, and
// accumulate in FP32 without materializing the full dequantized W13 tensor.
__global__ void gemm1_tiled_fused_w13_grouped_kernel(
    const float* __restrict__ a, const int32_t* __restrict__ local_expert_ids,
    const __nv_fp8_e4m3* __restrict__ gemm1_weights, const float* __restrict__ gemm1_weights_scale,
    float* __restrict__ g1, int64_t rows_per_expert) {
  __shared__ float a_tile[kGemm1TileM][kGemm1TileK];
  // Store B as [K][N+1] in shared memory. The +1 padding breaks 32-bank aliasing for both:
  // 1) load phase: warp writes fixed tile_col, varying tile_k
  // 2) compute phase: warp reads fixed kk, varying local_col
  __shared__ float b_tile[kGemm1TileK][kGemm1TileN + 1];

  int batch_expert = blockIdx.z;
  int32_t local_expert = local_expert_ids[batch_expert];
  const float* a_expert = a + static_cast<int64_t>(batch_expert) * rows_per_expert * kHiddenSize;
  float* g1_expert =
      g1 + static_cast<int64_t>(batch_expert) * rows_per_expert * (2 * kIntermediateSize);
  const __nv_fp8_e4m3* w13_fp8 =
      gemm1_weights + static_cast<int64_t>(local_expert) * (2 * kIntermediateSize) * kHiddenSize;
  const float* w13_scale = gemm1_weights_scale +
                           static_cast<int64_t>(local_expert) * ((2 * kIntermediateSize) / kBlock) *
                               (kHiddenSize / kBlock);

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
      if (global_row < rows_per_expert && global_k < kHiddenSize) {
        a_tile[tile_row][tile_k] =
            a_expert[static_cast<int64_t>(global_row) * kHiddenSize + global_k];
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
        b_tile[tile_k][tile_col] = fp8_to_float(w13_fp8[weight_idx]) * w13_scale[scale_idx];
      } else {
        b_tile[tile_k][tile_col] = 0.0f;
      }
    }

    __syncthreads();

    if (row < rows_per_expert && col < 2 * kIntermediateSize) {
      #pragma unroll
      for (int kk = 0; kk < kGemm1TileK; ++kk) {
        acc += a_tile[local_row][kk] * b_tile[kk][local_col];
      }
    }

    __syncthreads();
  }

  if (row < rows_per_expert && col < 2 * kIntermediateSize) {
    g1_expert[static_cast<int64_t>(row) * (2 * kIntermediateSize) + col] = acc;
  }
}

// GEMM2 specialized for C: [Tk, I], W2: [H, I] stored as FP8 block-scaled weights.
// This mirrors GEMM1: tile C and on-the-fly dequantized W2 into shared memory, then
// accumulate O = C @ W2^T in FP32 without materializing the full dequantized W2 tensor.
__global__ void gemm2_tiled_fused_w2_grouped_kernel(
    const float* __restrict__ c, const int32_t* __restrict__ local_expert_ids,
    const __nv_fp8_e4m3* __restrict__ gemm2_weights, const float* __restrict__ gemm2_weights_scale,
    float* __restrict__ o, int64_t rows_per_expert) {
  __shared__ float c_tile[kGemm2TileM][kGemm2TileK];
  __shared__ float b_tile[kGemm2TileK][kGemm2TileN + 1];

  int batch_expert = blockIdx.z;
  int32_t local_expert = local_expert_ids[batch_expert];
  const float* c_expert =
      c + static_cast<int64_t>(batch_expert) * rows_per_expert * kIntermediateSize;
  float* o_expert = o + static_cast<int64_t>(batch_expert) * rows_per_expert * kHiddenSize;
  const __nv_fp8_e4m3* w2_fp8 =
      gemm2_weights + static_cast<int64_t>(local_expert) * kHiddenSize * kIntermediateSize;
  const float* w2_scale = gemm2_weights_scale +
                          static_cast<int64_t>(local_expert) * (kHiddenSize / kBlock) *
                              (kIntermediateSize / kBlock);

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
      if (global_row < rows_per_expert && global_k < kIntermediateSize) {
        c_tile[tile_row][tile_k] =
            c_expert[static_cast<int64_t>(global_row) * kIntermediateSize + global_k];
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
        b_tile[tile_k][tile_col] = fp8_to_float(w2_fp8[weight_idx]) * w2_scale[scale_idx];
      } else {
        b_tile[tile_k][tile_col] = 0.0f;
      }
    }

    __syncthreads();

    if (row < rows_per_expert && col < kHiddenSize) {
      #pragma unroll
      for (int kk = 0; kk < kGemm2TileK; ++kk) {
        acc += c_tile[local_row][kk] * b_tile[kk][local_col];
      }
    }

    __syncthreads();
  }

  if (row < rows_per_expert && col < kHiddenSize) {
    o_expert[static_cast<int64_t>(row) * kHiddenSize + col] = acc;
  }
}

// Split G1 into X1 and X2, apply SiLU(X2), then multiply by X1.
__global__ void swiglu_grouped_kernel(const float* g1, float* c, int64_t rows_per_expert,
                                      int64_t num_experts) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows_per_expert * num_experts * kIntermediateSize;
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
__global__ void weighted_scatter_add_vec4_kernel(const float4* o, const int32_t* token_idx,
                                                 const float* weights, float4* output,
                                                 int64_t rows, int32_t global_expert,
                                                 int64_t vec_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * vec_width;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / vec_width;
  int64_t col = idx % vec_width;
  int32_t token = token_idx[row];
  if (token < 0) {
    return;
  }
  float weight = weights[static_cast<int64_t>(token) * kGlobalExperts + global_expert];
  float4 value = o[idx];
  float4 out = output[static_cast<int64_t>(token) * vec_width + col];
  out.x += value.x * weight;
  out.y += value.y * weight;
  out.z += value.z * weight;
  out.w += value.w * weight;
  output[static_cast<int64_t>(token) * vec_width + col] = out;
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
  DebugOptions debug = GetDebugOptions();
  DebugTimings timings;
  static bool histogram_printed = false;
  static bool timing_printed = false;

  auto time_cuda = [&](float* accum_ms, auto&& fn) {
    if (debug.timing) {
      ScopedCudaTimer timer(stream, accum_ms);
      fn();
    } else {
      fn();
    }
  };
  auto time_host = [&](double* accum_ms, auto&& fn) {
    if (debug.timing) {
      ScopedHostTimer timer(accum_ms);
      fn();
    } else {
      fn();
    }
  };

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
  constexpr int64_t kHiddenVecWidth = kHiddenSize / 4;

  // 1) FP8 block-scale dequantization
  time_cuda(&timings.dequant_ms, [&] {
    dequant_hidden_states_kernel<<<(hidden_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                                   stream>>>(
        static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()), a_fp32.get(), t);
    CHECK_CUDA(cudaGetLastError());
  });

  // 2) No-aux routing
  time_cuda(&timings.routing_ms, [&] {
    sigmoid_bias_kernel<<<(routing_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                          stream>>>(
        static_cast<const float*>(routing_logits.data_ptr()),
        static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()), s.get(),
        s_with_bias.get(), t);
    CHECK_CUDA(cudaGetLastError());

    routing_select_kernel<<<t, 1, 0, stream>>>(s.get(), s_with_bias.get(), topk_idx.get(),
                                                weights.get(), t, routed_scaling_factor);
    CHECK_CUDA(cudaGetLastError());
  });

  std::vector<int32_t> host_topk(static_cast<size_t>(t) * kTopK);
  time_cuda(&timings.topk_copy_ms, [&] {
    CHECK_CUDA(cudaMemcpyAsync(host_topk.data(), topk_idx.get(),
                               sizeof(int32_t) * static_cast<size_t>(t) * kTopK,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  });

  // Build the per-local-expert token lists on the host, mirroring the reference's
  // per-expert nonzero/index_select flow.
  std::vector<std::vector<int32_t>> token_lists(kLocalExperts);
  time_host(&timings.token_list_host_ms, [&] {
    for (int64_t token = 0; token < t; ++token) {
      for (int64_t k = 0; k < kTopK; ++k) {
        int32_t global_expert = host_topk[static_cast<size_t>(token) * kTopK + k];
        int64_t local_expert = static_cast<int64_t>(global_expert) - local_expert_offset;
        if (0 <= local_expert && local_expert < kLocalExperts) {
          token_lists[local_expert].push_back(static_cast<int32_t>(token));
        }
      }
    }
  });

  std::vector<int32_t> active_experts;
  for (int32_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    if (!token_lists[local_expert].empty()) {
      active_experts.push_back(local_expert);
    }
  }

  std::sort(active_experts.begin(), active_experts.end(),
            [&token_lists](int32_t lhs, int32_t rhs) {
              return token_lists[lhs].size() < token_lists[rhs].size();
            });
  std::vector<ExpertBucket> expert_buckets = BuildExpertBuckets(active_experts, token_lists);
  if (ShouldPrintDebugOnce(debug.histogram, &histogram_printed)) {
    PrintTokenHistogramSummary(t, local_expert_offset, token_lists);
    PrintBucketSummary(t, local_expert_offset, expert_buckets, active_experts, token_lists);
  }

  int64_t max_bucket_rows = 0;
  int64_t max_bucket_experts = 0;
  for (const ExpertBucket& bucket : expert_buckets) {
    int64_t bucket_experts = static_cast<int64_t>(bucket.end - bucket.begin);
    max_bucket_rows =
        std::max(max_bucket_rows, bucket_experts * bucket.rows_per_expert);
    max_bucket_experts = std::max(max_bucket_experts, bucket_experts);
  }

  AsyncBuffer<int32_t> bucket_token_idx_device(max_bucket_rows, stream);
  AsyncBuffer<int32_t> bucket_expert_ids_device(max_bucket_experts, stream);
  AsyncBuffer<float> a_e(static_cast<size_t>(max_bucket_rows) * kHiddenSize, stream);
  AsyncBuffer<float> g1(static_cast<size_t>(max_bucket_rows) * (2 * kIntermediateSize), stream);
  AsyncBuffer<float> c(static_cast<size_t>(max_bucket_rows) * kIntermediateSize, stream);
  AsyncBuffer<float> o(static_cast<size_t>(max_bucket_rows) * kHiddenSize, stream);

  // 3) Local expert compute and accumulation, grouped by bounded-padding buckets.
  for (const ExpertBucket& bucket : expert_buckets) {
    int64_t bucket_tk = bucket.rows_per_expert;
    int64_t bucket_experts = static_cast<int64_t>(bucket.end - bucket.begin);
    int64_t bucket_rows = bucket_experts * bucket_tk;
    std::vector<int32_t> host_bucket_expert_ids;
    std::vector<int32_t> host_bucket_tokens;
    time_host(&timings.bucket_pack_host_ms, [&] {
      host_bucket_expert_ids.resize(static_cast<size_t>(bucket_experts));
      host_bucket_tokens.assign(static_cast<size_t>(bucket_rows), -1);
      for (int64_t i = 0; i < bucket_experts; ++i) {
        int32_t local_expert = active_experts[bucket.begin + static_cast<size_t>(i)];
        host_bucket_expert_ids[static_cast<size_t>(i)] = local_expert;
        const std::vector<int32_t>& tokens = token_lists[local_expert];
        std::copy(tokens.begin(), tokens.end(),
                  host_bucket_tokens.begin() + static_cast<size_t>(i * bucket_tk));
      }
    });

    time_cuda(&timings.bucket_upload_ms, [&] {
      CHECK_CUDA(cudaMemcpyAsync(bucket_expert_ids_device.get(), host_bucket_expert_ids.data(),
                                 sizeof(int32_t) * bucket_experts, cudaMemcpyHostToDevice,
                                 stream));
      CHECK_CUDA(cudaMemcpyAsync(bucket_token_idx_device.get(), host_bucket_tokens.data(),
                                 sizeof(int32_t) * bucket_rows, cudaMemcpyHostToDevice, stream));
    });

    int64_t gather_total = bucket_rows * kHiddenVecWidth;
    time_cuda(&timings.gather_ms, [&] {
      gather_rows_vec4_kernel<<<(gather_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                                stream>>>(
          reinterpret_cast<const float4*>(a_fp32.get()), bucket_token_idx_device.get(),
          reinterpret_cast<float4*>(a_e.get()), bucket_rows, kHiddenVecWidth);
      CHECK_CUDA(cudaGetLastError());
    });

    time_cuda(&timings.gemm1_ms, [&] {
      dim3 gemm1_block(kGemm1TileN, kGemm1TileM);
      dim3 gemm1_grid((2 * kIntermediateSize + kGemm1TileN - 1) / kGemm1TileN,
                      (bucket_tk + kGemm1TileM - 1) / kGemm1TileM, bucket_experts);
      gemm1_tiled_fused_w13_grouped_kernel<<<gemm1_grid, gemm1_block, 0, stream>>>(
          a_e.get(), bucket_expert_ids_device.get(),
          static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
          static_cast<const float*>(gemm1_weights_scale.data_ptr()), g1.get(), bucket_tk);
      CHECK_CUDA(cudaGetLastError());
    });

    int64_t swiglu_total = bucket_rows * kIntermediateSize;
    time_cuda(&timings.swiglu_ms, [&] {
      swiglu_grouped_kernel<<<(swiglu_total + threads_1d - 1) / threads_1d, threads_1d, 0,
                              stream>>>(g1.get(), c.get(), bucket_tk, bucket_experts);
      CHECK_CUDA(cudaGetLastError());
    });

    time_cuda(&timings.gemm2_ms, [&] {
      dim3 gemm2_block(kGemm2TileN, kGemm2TileM);
      dim3 gemm2_grid((kHiddenSize + kGemm2TileN - 1) / kGemm2TileN,
                      (bucket_tk + kGemm2TileM - 1) / kGemm2TileM, bucket_experts);
      gemm2_tiled_fused_w2_grouped_kernel<<<gemm2_grid, gemm2_block, 0, stream>>>(
          c.get(), bucket_expert_ids_device.get(),
          static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr()),
          static_cast<const float*>(gemm2_weights_scale.data_ptr()), o.get(), bucket_tk);
      CHECK_CUDA(cudaGetLastError());
    });

    int64_t scatter_total = bucket_tk * kHiddenVecWidth;
    time_cuda(&timings.scatter_ms, [&] {
      for (int64_t i = 0; i < bucket_experts; ++i) {
        int32_t global_expert =
            local_expert_offset + host_bucket_expert_ids[static_cast<size_t>(i)];
        const float4* o_expert = reinterpret_cast<const float4*>(o.get()) +
                                 static_cast<int64_t>(i) * bucket_tk * kHiddenVecWidth;
        const int32_t* token_idx_expert =
            bucket_token_idx_device.get() + static_cast<int64_t>(i) * bucket_tk;
        weighted_scatter_add_vec4_kernel<<<(scatter_total + threads_1d - 1) / threads_1d,
                                           threads_1d, 0, stream>>>(
            o_expert, token_idx_expert, weights.get(),
            reinterpret_cast<float4*>(output_fp32.get()), bucket_tk, global_expert,
            kHiddenVecWidth);
        CHECK_CUDA(cudaGetLastError());
      }
    });
  }

  int64_t output_total = t * kHiddenSize;
  time_cuda(&timings.cast_ms, [&] {
    cast_output_kernel<<<(output_total + threads_1d - 1) / threads_1d, threads_1d, 0, stream>>>(
        output_fp32.get(), static_cast<__nv_bfloat16*>(output_tensor.data_ptr()), output_total);
    CHECK_CUDA(cudaGetLastError());
  });
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (ShouldPrintDebugOnce(debug.timing, &timing_printed)) {
    timings.Print(t, local_expert_offset);
  }

  return output_tensor;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
