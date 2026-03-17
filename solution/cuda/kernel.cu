/*
 * Correctness-first CUDA implementation for the MLSys26 MoE task.
 *
 * This is a direct translation of moe_reference.py using ATen CUDA ops inside
 * a Torch extension. It is intentionally naive and unfused.
 */

#include <torch/extension.h>

#include <limits>
#include <vector>

namespace idx = torch::indexing;

namespace
{

  void check_cuda_tensor(const torch::Tensor &tensor, const char *name)
  {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  }

  void check_same_device(const torch::Tensor &lhs, const torch::Tensor &rhs,
                         const char *lhs_name, const char *rhs_name)
  {
    TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name,
                " must be on the same device");
  }

} // namespace

torch::Tensor kernel(torch::Tensor routing_logits,
                     torch::Tensor routing_bias,
                     torch::Tensor hidden_states,
                     torch::Tensor hidden_states_scale,
                     torch::Tensor gemm1_weights,
                     torch::Tensor gemm1_weights_scale,
                     torch::Tensor gemm2_weights,
                     torch::Tensor gemm2_weights_scale,
                     int64_t local_expert_offset,
                     double routed_scaling_factor)
{
  // Check is CUDA tensors and same device.
  check_cuda_tensor(routing_logits, "routing_logits");
  check_cuda_tensor(routing_bias, "routing_bias");
  check_cuda_tensor(hidden_states, "hidden_states");
  check_cuda_tensor(hidden_states_scale, "hidden_states_scale");
  check_cuda_tensor(gemm1_weights, "gemm1_weights");
  check_cuda_tensor(gemm1_weights_scale, "gemm1_weights_scale");
  check_cuda_tensor(gemm2_weights, "gemm2_weights");
  check_cuda_tensor(gemm2_weights_scale, "gemm2_weights_scale");

  check_same_device(routing_logits, hidden_states, "routing_logits", "hidden_states");
  check_same_device(routing_logits, routing_bias, "routing_logits", "routing_bias");
  check_same_device(routing_logits, hidden_states_scale, "routing_logits", "hidden_states_scale");
  check_same_device(routing_logits, gemm1_weights, "routing_logits", "gemm1_weights");
  check_same_device(routing_logits, gemm1_weights_scale, "routing_logits", "gemm1_weights_scale");
  check_same_device(routing_logits, gemm2_weights, "routing_logits", "gemm2_weights");
  check_same_device(routing_logits, gemm2_weights_scale, "routing_logits", "gemm2_weights_scale");

  // Fixed parameters from reference example
  constexpr int64_t kHiddenSize = 7168;
  constexpr int64_t kIntermediateSize = 2048;
  constexpr int64_t kBlock = 128;
  constexpr int64_t kGlobalExperts = 256;
  constexpr int64_t kTopK = 8;
  constexpr int64_t kNumGroups = 8;
  constexpr int64_t kTopKGroups = 4;

  const int64_t e_local = gemm1_weights.size(0);
  const int64_t t = routing_logits.size(0);
  const int64_t e_global = routing_logits.size(1);

  TORCH_CHECK(kHiddenSize == hidden_states.size(1), "hidden_size must be 7168");
  TORCH_CHECK(kIntermediateSize == gemm2_weights.size(2), "intermediate_size must be 2048");
  TORCH_CHECK(e_global == kGlobalExperts, "num_experts must be 256");
  TORCH_CHECK(e_local == 32, "num_local_experts must be 32");

  const int64_t num_hidden_blocks = kHiddenSize / kBlock;
  const int64_t num_intermediate_blocks = kIntermediateSize / kBlock;
  const int64_t num_gemm1_out_blocks = (2 * kIntermediateSize) / kBlock;

  TORCH_CHECK(hidden_states.sizes() == torch::IntArrayRef({t, kHiddenSize}),
              "hidden_states shape mismatch");
  TORCH_CHECK(hidden_states_scale.sizes() == torch::IntArrayRef({num_hidden_blocks, t}),
              "hidden_states_scale shape mismatch");
  TORCH_CHECK(gemm1_weights.sizes() == torch::IntArrayRef({e_local, 2 * kIntermediateSize, kHiddenSize}),
              "gemm1_weights shape mismatch");
  TORCH_CHECK(gemm1_weights_scale.sizes() ==
                  torch::IntArrayRef({e_local, num_gemm1_out_blocks, num_hidden_blocks}),
              "gemm1_weights_scale shape mismatch");
  TORCH_CHECK(gemm2_weights.sizes() == torch::IntArrayRef({e_local, kHiddenSize, kIntermediateSize}),
              "gemm2_weights shape mismatch");
  TORCH_CHECK(gemm2_weights_scale.sizes() ==
                  torch::IntArrayRef({e_local, num_hidden_blocks, num_intermediate_blocks}),
              "gemm2_weights_scale shape mismatch");
  TORCH_CHECK(routing_bias.numel() == e_global, "routing_bias shape mismatch");

  // 1) FP8 block-scale dequantization
  // hidden_states: [T, H], scale: [H/128, T] (transposed layout)
  auto a_fp32 = hidden_states.to(torch::kFloat32);
  auto a_scale = hidden_states_scale.to(torch::kFloat32); // [H/128, T]
  auto a_scale_th = a_scale.permute({1, 0}).contiguous(); // # to [T, H/128]
  auto a_scale_expanded =
      a_scale_th.unsqueeze(-1).repeat({1, 1, kBlock}).reshape({t, kHiddenSize}).contiguous(); // [T, H]
  auto a = a_fp32 * a_scale_expanded;                                                         // [T, H] float32

  // W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
  auto w13_fp32 = gemm1_weights.to(torch::kFloat32);
  auto s13 = gemm1_weights_scale.to(torch::kFloat32);
  auto s13_expanded = torch::repeat_interleave(s13, kBlock, 1);     // [E, 2I, H/128]
  s13_expanded = torch::repeat_interleave(s13_expanded, kBlock, 2); // [E, 2I, H]
  auto w13 = w13_fp32 * s13_expanded;                               // [E, 2I, H] float32

  // W2: [E_local, H, I], scale: [E_local, H/128, I/128]
  auto w2_fp32 = gemm2_weights.to(torch::kFloat32);
  auto s2 = gemm2_weights_scale.to(torch::kFloat32);
  auto s2_expanded = torch::repeat_interleave(s2, kBlock, 1);     // [E, H, I/128]
  s2_expanded = torch::repeat_interleave(s2_expanded, kBlock, 2); // [E, H, I]
  auto w2 = w2_fp32 * s2_expanded;                                // [E, H, I] float32

  // 2) No-aux routing

  auto logits = routing_logits.to(torch::kFloat32);
  auto bias = routing_bias.to(torch::kFloat32).reshape({e_global});

  // Sigmoid
  auto s = 1.0f / (1.0f + torch::exp(-logits)); // [T, E]
  auto s_with_bias = s + bias;                  // [T, E] (broadcast)

  // Grouping
  const int64_t group_size = e_global / kNumGroups;                  // 32
  auto s_wb_grouped = s_with_bias.view({t, kNumGroups, group_size}); // [T, 8, 32]

  // Group scores = sum of top-2 values within each group
  auto top2_vals = std::get<0>(torch::topk(s_wb_grouped, 2, 2, true, false)); // [T, 8, 2]
  auto group_scores = top2_vals.sum(2);                                       // [T, 8]

  // Select topk_group groups → group mask
  auto group_idx = std::get<1>(torch::topk(group_scores, kTopKGroups, 1, true, false)); // [T, 4]
  auto group_mask = torch::zeros_like(group_scores);                                    // [T, 8]
  group_mask.scatter_(1, group_idx, 1.0);
  auto score_mask =
      group_mask.unsqueeze(2).expand({t, kNumGroups, group_size}).reshape({t, e_global}); // [T, E]

  // Global top-k (within kept groups), based on s_with_bias
  auto scores_pruned = s_with_bias.masked_fill(
      score_mask.eq(0), std::numeric_limits<float>::lowest());                    // [T, E]
  auto topk_idx = std::get<1>(torch::topk(scores_pruned, kTopK, 1, true, false)); // [T, 8]

  // Combination weights: use s (without bias) for normalization
  auto m = torch::zeros_like(s); // [T, E]
  m.scatter_(1, topk_idx, 1.0);  // 0/1 mask
  auto weights = s * m;          // [T, E]
  auto weights_sum = weights.sum(1, true) + 1e-20;
  weights = (weights / weights_sum) * static_cast<float>(routed_scaling_factor); // [T, E]

  // 3) Local expert compute and accumulation
  auto output = torch::zeros({t, kHiddenSize},
                             torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));

  // For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
  for (int64_t le = 0; le < e_local; ++le)
  {
    const int64_t ge = local_expert_offset + le;
    if (ge < 0 || ge >= e_global)
    {
      continue;
    }

    // Tokens that selected this global expert ge in their top-k
    auto sel_mask = topk_idx.eq(ge).any(1);
    if (!sel_mask.any().item<bool>())
    {
      continue;
    }

    // Gather inputs and weights for this expert
    auto token_idx = torch::nonzero(sel_mask).squeeze(1); // [Tk, H]
    auto a_e = a.index_select(0, token_idx);              // [2I, H]
    auto w13_e = w13.index({le});                         // [H, I]
    auto w2_e = w2.index({le});

    // GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
    auto g1 = torch::matmul(a_e, w13_e.t()); // [Tk, 2I]

    // SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
    auto x1 = g1.index({idx::Slice(), idx::Slice(0, kIntermediateSize)});                     // [Tk, I]
    auto x2 = g1.index({idx::Slice(), idx::Slice(kIntermediateSize, 2 * kIntermediateSize)}); // [Tk, I]
    auto silu_x2 = x2 / (1.0f + torch::exp(-x2));                                             // [Tk, I]
    auto c = silu_x2 * x1;                                                                    // [Tk, I]

    // GEMM2: [Tk, I] @ [I, H] = [Tk, H]
    auto o = torch::matmul(c, w2_e.t());

    // Accumulate with per-token routing weights for this expert
    auto w_tok = weights.index_select(0, token_idx).select(1, ge); // [Tk]
    output.index_add_(0, token_idx, o * w_tok.unsqueeze(1));       // [Tk,H] * [Tk,1]
  }

  return output.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("kernel", &kernel, "Naive MoE CUDA forward");
}
