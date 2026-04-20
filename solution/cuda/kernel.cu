#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "tile_config.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
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
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x & 0x78) >> 3;
  int mant = x & 0x07;
  if (exp == 0) return sign * (float)mant * 0.001953125f;
  float res = sign * (1.0f + (float)mant * 0.125f);
  int shift = exp - 7;
  if (shift > 0) res *= (float)(1 << shift);
  else if (shift < 0) res /= (float)(1 << (-shift));
  return res;
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

__global__ void reorder_tokens_kernel(const int32_t* topk_idx, const int32_t* expert_offsets, int32_t* expert_current_counts, int32_t* reordered_tokens, int64_t t, int32_t local_expert_offset) {
  int64_t token_id = blockIdx.x;
  if (token_id >= t) return;
  for (int k = 0; k < kTopK; ++k) {
    int32_t ge = topk_idx[token_id * kTopK + k];
    int32_t le = ge - local_expert_offset;
    if (le >= 0 && le < kLocalExperts) {
      int32_t slot = atomicAdd(&expert_current_counts[le], 1);
      reordered_tokens[expert_offsets[le] + slot] = (int32_t)token_id;
    }
  }
}

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

__global__ void optimized_scatter_add_kernel(const float* o, const int32_t* token_idx, const float* weights, float* output, int64_t rows, int32_t global_expert) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * kHiddenSize) return;
  int64_t row = idx / kHiddenSize;
  int32_t token = token_idx[row];
  if (token < 0) return;
  float w = weights[(int64_t)token * kGlobalExperts + global_expert];
  atomicAdd(&output[(int64_t)token * kHiddenSize + (idx % kHiddenSize)], o[idx] * w);
}

__global__ void cast_output_kernel(const float* src, __nv_bfloat16* dst, int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) dst[idx] = __float2bfloat16(src[idx]);
}

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
  
  // GPU Bucketing
  AsyncBuffer<int32_t> d_expert_counts(kLocalExperts, stream), d_expert_offsets(kLocalExperts + 1, stream), d_reordered_tokens(t * kTopK, stream), d_tmp_counts(kLocalExperts, stream);
  CHECK_CUDA(cudaMemsetAsync(d_expert_counts.get(), 0, kLocalExperts * sizeof(int32_t), stream));
  CHECK_CUDA(cudaMemsetAsync(d_tmp_counts.get(), 0, kLocalExperts * sizeof(int32_t), stream));
  
  count_expert_tokens_kernel<<<(t * kTopK + 255) / 256, 256, 0, stream>>>(tk_idx.get(), d_expert_counts.get(), t, (int32_t)local_expert_offset);
  
  std::vector<int32_t> h_expert_counts(kLocalExperts);
  CHECK_CUDA(cudaMemcpyAsync(h_expert_counts.data(), d_expert_counts.get(), kLocalExperts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  
  std::vector<int32_t> h_expert_offsets(kLocalExperts + 1, 0);
  for (int i = 0; i < kLocalExperts; ++i) h_expert_offsets[i+1] = h_expert_offsets[i] + h_expert_counts[i];
  CHECK_CUDA(cudaMemcpyAsync(d_expert_offsets.get(), h_expert_offsets.data(), (kLocalExperts + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  
  reorder_tokens_kernel<<<t, 1, 0, stream>>>(tk_idx.get(), d_expert_offsets.get(), d_tmp_counts.get(), d_reordered_tokens.get(), t, (int32_t)local_expert_offset);

  AsyncBuffer<float> w13_f32(kLocalExperts * 2 * kIntermediateSize * kHiddenSize, stream);
  AsyncBuffer<float> w2_f32(kLocalExperts * kHiddenSize * kIntermediateSize, stream);
  dequant_weights_to_f32_kernel<<<(w13_f32.count+255)/256, 256, 0, stream>>>((uint8_t*)gemm1_weights.data_ptr(), (float*)gemm1_weights_scale.data_ptr(), w13_f32.get(), kLocalExperts, 2*kIntermediateSize, kHiddenSize);
  dequant_weights_to_f32_kernel<<<(w2_f32.count+255)/256, 256, 0, stream>>>((uint8_t*)gemm2_weights.data_ptr(), (float*)gemm2_weights_scale.data_ptr(), w2_f32.get(), kLocalExperts, kHiddenSize, kIntermediateSize);

  int64_t total_reordered = h_expert_offsets[kLocalExperts];
  if (total_reordered == 0) { cast_output_kernel<<<(t*kHiddenSize+255)/256, 256, 0, stream>>>(o_f32.get(), (__nv_bfloat16*)output_tensor.data_ptr(), t*kHiddenSize); CHECK_CUDA(cudaStreamSynchronize(stream)); return output_tensor; }

  AsyncBuffer<float> a_f32(total_reordered * kHiddenSize, stream);
  AsyncBuffer<float> g1(total_reordered * 2 * kIntermediateSize, stream), c(total_reordered * kIntermediateSize, stream), o(total_reordered * kHiddenSize, stream);

  for (int32_t le = 0; le < kLocalExperts; ++le) {
    int32_t m = h_expert_counts[le];
    if (m <= 0) continue;
    int32_t offset = h_expert_offsets[le];
    fused_gather_dequant_f32_kernel<<<(m*kHiddenSize+255)/256, 256, 0, stream>>>((uint8_t*)hidden_states.data_ptr(), d_reordered_tokens.get() + offset, (float*)hidden_states_scale.data_ptr(), a_f32.get() + (int64_t)offset * kHiddenSize, m, scale_t);
    RunGemmF32(a_f32.get() + (int64_t)offset * kHiddenSize, w13_f32.get() + (int64_t)le * 2 * kIntermediateSize * kHiddenSize, g1.get() + (int64_t)offset * 2 * kIntermediateSize, m, 2*kIntermediateSize, kHiddenSize, stream);
    swiglu_f32_kernel<<<((m*kIntermediateSize)+255)/256, 256, 0, stream>>>(g1.get() + (int64_t)offset * 2 * kIntermediateSize, c.get() + (int64_t)offset * kIntermediateSize, m);
    RunGemmF32(c.get() + (int64_t)offset * kIntermediateSize, w2_f32.get() + (int64_t)le * kHiddenSize * kIntermediateSize, o.get() + (int64_t)offset * kHiddenSize, m, kHiddenSize, kIntermediateSize, stream);
    optimized_scatter_add_kernel<<<((m*kHiddenSize)+255)/256, 256, 0, stream>>>(o.get() + (int64_t)offset * kHiddenSize, d_reordered_tokens.get() + offset, w.get(), o_f32.get(), m, (int32_t)(le + local_expert_offset));
  }
  cast_output_kernel<<<(t*kHiddenSize+255)/256, 256, 0, stream>>>(o_f32.get(), (__nv_bfloat16*)output_tensor.data_ptr(), t*kHiddenSize);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return output_tensor;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
