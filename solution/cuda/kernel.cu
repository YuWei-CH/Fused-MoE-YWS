#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "tile_config.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
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

namespace {

constexpr int64_t kHiddenSize = 7168;
constexpr int64_t kIntermediateSize = 2048;
constexpr int64_t kBlock = 128;
constexpr int64_t kGlobalExperts = 256;
constexpr int64_t kLocalExperts = 32;
constexpr int64_t kTopK = 8;
constexpr int64_t kNumGroups = 8;
constexpr int64_t kTopKGroups = 4;
constexpr int64_t kGroupSize = kGlobalExperts / kNumGroups;

namespace cutlass_grouped {
  using ElementA = float; // TF32
  using ElementB = float; // TF32
  using ElementC = float; // CRITICAL: Keep as float for precision
  using ElementAccumulator = float;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<
      typename cutlass::gemm::kernel::DefaultGemmGrouped<
          ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
          ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,
          ElementC, cutlass::layout::RowMajor,
          ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<128, 128, 32>,
          cutlass::gemm::GemmShape<64, 64, 32>,
          cutlass::gemm::GemmShape<16, 8, 8>,
          cutlass::epilogue::thread::LinearCombination<ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator>,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
          4
      >::GemmKernel>;
}

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t err__ = (call);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      TVM_FFI_THROW(RuntimeError) << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                                  << cudaGetErrorString(err__);                                    \
    }                                                                                              \
  } while (0)

inline void CheckCutlassStatus(cutlass::Status status, const char* msg) {
  if (status != cutlass::Status::kSuccess) {
    TVM_FFI_THROW(RuntimeError) << "CUTLASS error: " << msg << " ("                                \
                                << cutlassGetStatusString(status) << ")";                          \
  }
}

template <typename T> struct AsyncBuffer {
  T* ptr = nullptr;
  size_t count = 0;
  cudaStream_t stream;
  AsyncBuffer(size_t n, cudaStream_t s) : count(n), stream(s) {
    if (n > 0) CHECK_CUDA(cudaMallocAsync(&ptr, n * sizeof(T), stream));
  }
  ~AsyncBuffer() { if (ptr) cudaFreeAsync(ptr, stream); }
  T* get() const { return ptr; }
};

__device__ inline float fp8_e4m3_to_float(uint8_t x) {
  int s = (x >> 7);
  int e = (x >> 3) & 0xf;
  int m = x & 0x7;
  if (e == 0) return (s ? -1.0f : 1.0f) * (m * 0.0009765625f);
  if (e == 15 && m == 7) return NAN;
  return (s ? -1.0f : 1.0f) * ldexpf((float)(8 + m), e - 10);
}

__global__ void sigmoid_bias_kernel(const float* logits, const __nv_bfloat16* bias, float* s,
                                    float* s_with_bias, int64_t t) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= t * kGlobalExperts) return;
  int64_t e = idx % kGlobalExperts;
  float sig = 1.0f / (1.0f + expf(-logits[idx]));
  s[idx] = sig;
  s_with_bias[idx] = sig + __bfloat162float(bias[e]);
}

__global__ void routing_select_kernel(const float* s, const float* s_with_bias,
                                       int32_t* topk_idx, float* weight_row_base,
                                       int64_t t, double routed_scaling_factor) {
  int64_t token = blockIdx.x;
  if (token >= t) return;
  const float* swb_r = s_with_bias + token * kGlobalExperts;
  const float* s_r = s + token * kGlobalExperts;
  float g_scores[kNumGroups];
  for (int g = 0; g < kNumGroups; ++g) {
    float b0 = -1e38f, b1 = -1e38f;
    for (int i = 0; i < kGroupSize; ++i) {
      float v = swb_r[g * kGroupSize + i];
      if (v > b0) { b1 = b0; b0 = v; } else if (v > b1) b1 = v;
    }
    g_scores[g] = b0 + b1;
  }
  uint8_t keep[kNumGroups]; for (int g = 0; g < kNumGroups; ++g) keep[g] = 0;
  for (int p = 0; p < kTopKGroups; ++p) {
    int bg = -1; float bs = -1e38f;
    for (int g = 0; g < kNumGroups; ++g) if (!keep[g] && g_scores[g] > bs) { bs = g_scores[g]; bg = g; }
    if (bg != -1) keep[bg] = 1;
  }
  int32_t sel_i[kTopK]; float sel_s[kTopK];
  for (int k = 0; k < kTopK; ++k) { sel_i[k] = -1; sel_s[k] = -1e38f; }
  for (int e = 0; e < kGlobalExperts; ++e) {
    if (!keep[e / kGroupSize]) continue;
    float v = swb_r[e];
    int ins = -1;
    for (int k = 0; k < kTopK; ++k) if (v > sel_s[k]) { ins = k; break; }
    if (ins != -1) {
      for (int k = kTopK - 1; k > ins; --k) { sel_s[k] = sel_s[k - 1]; sel_i[k] = sel_i[k - 1]; }
      sel_s[ins] = v; sel_i[ins] = (int32_t)e;
    }
  }
  double w_sum = 0.0;
  for (int k = 0; k < kTopK; ++k) if (sel_i[k] >= 0) w_sum += (double)s_r[sel_i[k]];
  w_sum += 1e-20;
  float* w_row = weight_row_base + token * kGlobalExperts;
  for (int e = 0; e < kGlobalExperts; ++e) w_row[e] = 0.0f;
  int32_t* tk_r = topk_idx + token * kTopK;
  for (int k = 0; k < kTopK; ++k) {
    int e = sel_i[k]; tk_r[k] = e;
    if (e >= 0) w_row[e] = (float)(((double)s_r[e] / w_sum) * routed_scaling_factor);
  }
}

__global__ void count_expert_tokens_kernel(const int32_t* topk_idx, int32_t* expert_counts, int64_t t, int32_t local_expert_offset) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= t * kTopK) return;
  int32_t ge = topk_idx[idx];
  int32_t le = ge - local_expert_offset;
  if (le >= 0 && le < kLocalExperts) atomicAdd(&expert_counts[le], 1);
}

__global__ void reorder_tokens_kernel(const int32_t* topk_idx, const int32_t* expert_offsets, int32_t* expert_current_counts, int32_t* reordered_tokens, int32_t* reordered_le_idx, int64_t t, int32_t local_expert_offset) {
  int64_t token_id = blockIdx.x;
  if (token_id >= t) return;
  for (int k = 0; k < kTopK; ++k) {
    int32_t ge = topk_idx[token_id * kTopK + k];
    int32_t le = ge - local_expert_offset;
    if (le >= 0 && le < kLocalExperts) {
      int32_t slot = atomicAdd(&expert_current_counts[le], 1);
      int32_t pos = expert_offsets[le] + slot;
      reordered_tokens[pos] = (int32_t)token_id;
      reordered_le_idx[pos] = le;
    }
  }
}

// SIMT Dequant (Float)
__global__ void dequant_weights_to_f32_kernel(const uint8_t* src, const float* scale, float* dst, int64_t num_experts, int64_t n, int64_t k) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t expert_size = n * k;
  if (idx >= num_experts * expert_size) return;
  int64_t ex = idx / expert_size; int64_t off = idx % expert_size;
  int64_t row = off / k; int64_t col = off % k;
  int64_t n_blks = n / kBlock; int64_t k_blks = k / kBlock;
  dst[idx] = fp8_e4m3_to_float(src[idx]) * scale[ex * n_blks * k_blks + (row / kBlock) * k_blks + (col / kBlock)];
}

__global__ void fused_gather_dequant_f32_kernel(const uint8_t* src, const int32_t* token_idx, const float* scale_src, float* dst, int64_t m, int64_t scale_t) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * kHiddenSize) return;
  int64_t row = idx / kHiddenSize; int64_t col = idx % kHiddenSize;
  int32_t token = token_idx[row];
  if (token < 0) { dst[idx] = 0.0f; return; }
  float scale = scale_src[(int64_t)(col / kBlock) * scale_t + token];
  dst[idx] = fp8_e4m3_to_float(src[(int64_t)token * kHiddenSize + col]) * scale;
}

__global__ void swiglu_f32_kernel(const float* g1, float* c, int64_t total_rows) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_rows * kIntermediateSize) return;
  int64_t row = idx / kIntermediateSize; int64_t col = idx % kIntermediateSize;
  float x1 = g1[row * 2 * kIntermediateSize + col];
  float x2 = g1[row * 2 * kIntermediateSize + kIntermediateSize + col];
  c[idx] = x1 * (x2 / (1.0f + expf(-x2)));
}

__global__ void optimized_scatter_add_kernel(const float* o, const int32_t* token_idx, const int32_t* le_idx, const float* weights, float* output, int64_t total_reordered, int32_t local_expert_offset) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_reordered * kHiddenSize) return;
  int64_t row = idx / kHiddenSize;
  int32_t token = token_idx[row];
  int32_t le = le_idx[row];
  float w = weights[(int64_t)token * kGlobalExperts + (le + local_expert_offset)];
  atomicAdd(&output[(int64_t)token * kHiddenSize + (idx % kHiddenSize)], o[idx] * w);
}

__global__ void cast_output_kernel(const float* src, __nv_bfloat16* dst, int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) dst[idx] = __float2bfloat16(src[idx]);
}

// SIMT Helper
void RunGemmF32(const float* a, const float* b, float* d, int m, int n, int k, cudaStream_t stream) {
  using ElementA = float; using LayoutA = cutlass::layout::RowMajor;
  using ElementB = float; using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float; using LayoutC = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80>;
  Gemm gemm_op;
  typename Gemm::Arguments args({m, n, k}, {(ElementA*)a, k}, {(ElementB*)b, k}, {(ElementC*)d, n}, {(ElementC*)d, n}, {1.0f, 0.0f});
  CheckCutlassStatus(gemm_op.initialize(args, nullptr, stream), "GEMM F32");
  CheckCutlassStatus(gemm_op.run(stream), "GEMM F32 run");
}

// Grouped GEMM Helper
void RunGroupedGemm(cutlass::gemm::GemmCoord* d_problem_sizes, 
                    cutlass::gemm::GemmCoord* h_problem_sizes,
                    cutlass_grouped::ElementA** ptr_A, 
                    cutlass_grouped::ElementB** ptr_B, 
                    cutlass_grouped::ElementC** ptr_C, 
                    cutlass_grouped::ElementC** ptr_D,
                    int expert_count, int k, int n, cudaStream_t stream) {
  using GemmGrouped = cutlass_grouped::GemmGrouped;
  using LongIndex = typename GemmGrouped::LayoutA::Stride::LongIndex;
  
  // LDB = K for ColumnMajor B matrix (transposed weight)
  std::vector<LongIndex> lda(expert_count, k), ldb(expert_count, k), ldc(expert_count, n), ldd(expert_count, n);
  AsyncBuffer<LongIndex> d_lda(expert_count, stream), d_ldb(expert_count, stream), d_ldc(expert_count, stream), d_ldd(expert_count, stream);
  CHECK_CUDA(cudaMemcpyAsync(d_lda.get(), lda.data(), expert_count * sizeof(LongIndex), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_ldb.get(), ldb.data(), expert_count * sizeof(LongIndex), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_ldc.get(), ldc.data(), expert_count * sizeof(LongIndex), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_ldd.get(), ldd.data(), expert_count * sizeof(LongIndex), cudaMemcpyHostToDevice, stream));

  typename GemmGrouped::Arguments args(
      d_problem_sizes,
      expert_count,
      1024,
      { (float)1.0f, (float)0.0f },
      ptr_A, ptr_B, ptr_C, ptr_D,
      d_lda.get(), d_ldb.get(), d_ldc.get(), d_ldd.get(),
      h_problem_sizes
  );

  GemmGrouped gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  AsyncBuffer<uint8_t> workspace(workspace_size, stream);
  CheckCutlassStatus(gemm_op.initialize(args, workspace.get(), stream), "Grouped GEMM Init");
  CheckCutlassStatus(gemm_op.run(stream), "Grouped GEMM Run");
}

} // namespace

tvm::ffi::Tensor kernel(tvm::ffi::Tensor routing_logits, tvm::ffi::Tensor routing_bias, tvm::ffi::Tensor hidden_states,
                   tvm::ffi::Tensor hidden_states_scale, tvm::ffi::Tensor gemm1_weights, tvm::ffi::Tensor gemm1_weights_scale,
                   tvm::ffi::Tensor gemm2_weights, tvm::ffi::Tensor gemm2_weights_scale, int64_t local_expert_offset,
                   double routed_scaling_factor) {
  auto dev_obj = hidden_states.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(3, dev_obj.device_id);
  int64_t t = hidden_states.shape()[0];
  int64_t scale_t = hidden_states_scale.shape()[1];
  DLDataType bf16_dt{4, 16, 1};
  tvm::ffi::Tensor output_tensor = tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, tvm::ffi::Shape({t, kHiddenSize}), bf16_dt, dev_obj);
  
  AsyncBuffer<float> s(t * kGlobalExperts, stream), s_wb(t * kGlobalExperts, stream), w(t * kGlobalExperts, stream), o_f32(t * kHiddenSize, stream);
  AsyncBuffer<int32_t> tk_idx(t * kTopK, stream);
  CHECK_CUDA(cudaMemsetAsync(o_f32.get(), 0, t * kHiddenSize * sizeof(float), stream));
  
  sigmoid_bias_kernel<<<(t*kGlobalExperts+255)/256, 256, 0, stream>>>((float*)routing_logits.data_ptr(), (__nv_bfloat16*)routing_bias.data_ptr(), s.get(), s_wb.get(), t);
  routing_select_kernel<<<t, 1, 0, stream>>>(s.get(), s_wb.get(), tk_idx.get(), w.get(), t, routed_scaling_factor);
  
  AsyncBuffer<int32_t> d_expert_counts(kLocalExperts, stream), d_expert_offsets(kLocalExperts + 1, stream), d_reordered_tokens(t * kTopK, stream), d_reordered_le(t * kTopK, stream), d_tmp_counts(kLocalExperts, stream);
  CHECK_CUDA(cudaMemsetAsync(d_expert_counts.get(), 0, kLocalExperts * sizeof(int32_t), stream));
  CHECK_CUDA(cudaMemsetAsync(d_tmp_counts.get(), 0, kLocalExperts * sizeof(int32_t), stream));
  count_expert_tokens_kernel<<<(t * kTopK + 255) / 256, 256, 0, stream>>>(tk_idx.get(), d_expert_counts.get(), t, (int32_t)local_expert_offset);
  
  thrust::device_ptr<int32_t> dev_counts_ptr(d_expert_counts.get());
  thrust::device_ptr<int32_t> dev_offsets_ptr(d_expert_offsets.get());
  thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_counts_ptr, dev_counts_ptr + kLocalExperts + 1, dev_offsets_ptr);
  
  reorder_tokens_kernel<<<t, 1, 0, stream>>>(tk_idx.get(), d_expert_offsets.get(), d_tmp_counts.get(), d_reordered_tokens.get(), d_reordered_le.get(), t, (int32_t)local_expert_offset);

  int32_t h_total_reordered;
  CHECK_CUDA(cudaMemcpyAsync(&h_total_reordered, d_expert_offsets.get() + kLocalExperts, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  std::vector<int32_t> h_counts(kLocalExperts), h_offsets(kLocalExperts + 1);
  CHECK_CUDA(cudaMemcpyAsync(h_counts.data(), d_expert_counts.get(), kLocalExperts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(h_offsets.data(), d_expert_offsets.get(), (kLocalExperts+1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (h_total_reordered == 0) { cast_output_kernel<<<(t*kHiddenSize+255)/256, 256, 0, stream>>>(o_f32.get(), (__nv_bfloat16*)output_tensor.data_ptr(), t*kHiddenSize); CHECK_CUDA(cudaStreamSynchronize(stream)); return output_tensor; }

  // Allocation
  AsyncBuffer<float> g1_f32(h_total_reordered * 2 * kIntermediateSize, stream);
  AsyncBuffer<float> c_f32(h_total_reordered * kIntermediateSize, stream);
  
  AsyncBuffer<float> a_f32(h_total_reordered * kHiddenSize, stream);
  AsyncBuffer<float> w13_f32(kLocalExperts * 2 * kIntermediateSize * kHiddenSize, stream);
  AsyncBuffer<float> w2_f32(kLocalExperts * kHiddenSize * kIntermediateSize, stream);
  
  AsyncBuffer<float> o_grouped_f32(h_total_reordered * kHiddenSize, stream); // CRITICAL FIX: Sized correctly for 8x token expansion

  // Dequantization
  fused_gather_dequant_f32_kernel<<<(h_total_reordered*kHiddenSize+255)/256, 256, 0, stream>>>((uint8_t*)hidden_states.data_ptr(), d_reordered_tokens.get(), (float*)hidden_states_scale.data_ptr(), a_f32.get(), h_total_reordered, scale_t);
  dequant_weights_to_f32_kernel<<<(w13_f32.count+255)/256, 256, 0, stream>>>((uint8_t*)gemm1_weights.data_ptr(), (float*)gemm1_weights_scale.data_ptr(), w13_f32.get(), kLocalExperts, 2*kIntermediateSize, kHiddenSize);
  dequant_weights_to_f32_kernel<<<(w2_f32.count+255)/256, 256, 0, stream>>>((uint8_t*)gemm2_weights.data_ptr(), (float*)gemm2_weights_scale.data_ptr(), w2_f32.get(), kLocalExperts, kHiddenSize, kIntermediateSize);

  // HYBRID DISPATCH: GEMM1
  const int THRESHOLD1 = tile_config::kLargeGemm1TensorCoreThreshold;
  std::vector<cutlass::gemm::GemmCoord> prob1;
  std::vector<cutlass_grouped::ElementA*> ptr_A1;
  std::vector<cutlass_grouped::ElementB*> ptr_B1;
  std::vector<cutlass_grouped::ElementC*> ptr_C1, ptr_D1;

  for(int i=0; i<kLocalExperts; ++i) {
    if(h_counts[i] <= 0) continue;
    if (h_counts[i] < THRESHOLD1) {
        // SIMT path for small buckets
        RunGemmF32(a_f32.get() + (int64_t)h_offsets[i] * kHiddenSize, 
                   w13_f32.get() + (int64_t)i * 2 * kIntermediateSize * kHiddenSize,
                   g1_f32.get() + (int64_t)h_offsets[i] * 2 * kIntermediateSize,
                   h_counts[i], 2 * kIntermediateSize, kHiddenSize, stream);
    } else {
        // Grouped GEMM path for large buckets
        prob1.push_back({(int)h_counts[i], (int)(2*kIntermediateSize), (int)kHiddenSize});
        ptr_A1.push_back(a_f32.get() + (int64_t)h_offsets[i] * kHiddenSize);
        ptr_B1.push_back(w13_f32.get() + (int64_t)i * 2 * kIntermediateSize * kHiddenSize);
        ptr_C1.push_back(g1_f32.get() + (int64_t)h_offsets[i] * 2 * kIntermediateSize);
        ptr_D1.push_back(g1_f32.get() + (int64_t)h_offsets[i] * 2 * kIntermediateSize);
    }
  }

  if (!prob1.empty()) {
      AsyncBuffer<cutlass::gemm::GemmCoord> d_prob1(prob1.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementA*> d_ptrA1(ptr_A1.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementB*> d_ptrB1(ptr_B1.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementC*> d_ptrC1(ptr_C1.size(), stream), d_ptrD1(ptr_D1.size(), stream);

      CHECK_CUDA(cudaMemcpyAsync(d_prob1.get(), prob1.data(), prob1.size() * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrA1.get(), ptr_A1.data(), ptr_A1.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrB1.get(), ptr_B1.data(), ptr_B1.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrC1.get(), ptr_C1.data(), ptr_C1.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrD1.get(), ptr_D1.data(), ptr_D1.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));

      RunGroupedGemm(d_prob1.get(), prob1.data(), d_ptrA1.get(), d_ptrB1.get(), d_ptrC1.get(), d_ptrD1.get(), prob1.size(), kHiddenSize, 2*kIntermediateSize, stream);
  }

  swiglu_f32_kernel<<<(h_total_reordered * kIntermediateSize + 255) / 256, 256, 0, stream>>>(g1_f32.get(), c_f32.get(), h_total_reordered);

  // HYBRID DISPATCH: GEMM2
  const int THRESHOLD2 = tile_config::kLargeGemm2TensorCoreThreshold;
  std::vector<cutlass::gemm::GemmCoord> prob2;
  std::vector<cutlass_grouped::ElementA*> ptr_C2;
  std::vector<cutlass_grouped::ElementB*> ptr_B2;
  std::vector<cutlass_grouped::ElementC*> ptr_D2;

  for(int i=0; i<kLocalExperts; ++i) {
    if(h_counts[i] <= 0) continue;
    if (h_counts[i] < THRESHOLD2) {
        RunGemmF32(c_f32.get() + (int64_t)h_offsets[i] * kIntermediateSize, 
                   w2_f32.get() + (int64_t)i * kHiddenSize * kIntermediateSize,
                   o_grouped_f32.get() + (int64_t)h_offsets[i] * kHiddenSize,
                   h_counts[i], kHiddenSize, kIntermediateSize, stream);
    } else {
        prob2.push_back({(int)h_counts[i], (int)kHiddenSize, (int)kIntermediateSize});
        ptr_C2.push_back(c_f32.get() + (int64_t)h_offsets[i] * kIntermediateSize);
        ptr_B2.push_back(w2_f32.get() + (int64_t)i * kHiddenSize * kIntermediateSize);
        ptr_D2.push_back(o_grouped_f32.get() + (int64_t)h_offsets[i] * kHiddenSize);
    }
  }

  if (!prob2.empty()) {
      AsyncBuffer<cutlass::gemm::GemmCoord> d_prob2(prob2.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementA*> d_ptrC2(ptr_C2.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementB*> d_ptrB2(ptr_B2.size(), stream);
      AsyncBuffer<cutlass_grouped::ElementC*> d_ptrD2(ptr_D2.size(), stream);

      CHECK_CUDA(cudaMemcpyAsync(d_prob2.get(), prob2.data(), prob2.size() * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrC2.get(), ptr_C2.data(), ptr_C2.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrB2.get(), ptr_B2.data(), ptr_B2.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_ptrD2.get(), ptr_D2.data(), ptr_D2.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));

      RunGroupedGemm(d_prob2.get(), prob2.data(), d_ptrC2.get(), d_ptrB2.get(), d_ptrD2.get(), d_ptrD2.get(), prob2.size(), kIntermediateSize, kHiddenSize, stream);
  }

  // Final scatter-add to o_f32 (which is sized t * H)
  optimized_scatter_add_kernel<<<(h_total_reordered * kHiddenSize + 255) / 256, 256, 0, stream>>>(o_grouped_f32.get(), d_reordered_tokens.get(), d_reordered_le.get(), w.get(), o_f32.get(), h_total_reordered, (int32_t)local_expert_offset);

  cast_output_kernel<<<(t*kHiddenSize+255)/256, 256, 0, stream>>>(o_f32.get(), (__nv_bfloat16*)output_tensor.data_ptr(), t*kHiddenSize);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return output_tensor;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
