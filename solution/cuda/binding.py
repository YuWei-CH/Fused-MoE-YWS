"""TVM FFI binding that lazily builds and dispatches the naive MoE CUDA extension."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from tvm.ffi import register_func


THIS_DIR = Path(__file__).resolve().parent
KERNEL_PATH = THIS_DIR / "kernel.cu"

# hash-based JIT extension loading (For test now)
@lru_cache(maxsize=1)
def _load_extension():
    ### Read .cu file, compute hash, and create module
    source_hash = hashlib.sha1(KERNEL_PATH.read_bytes()).hexdigest()[:12]
    module_name = f"flashinfer_moe_cuda_{source_hash}"
    return load(
        name=module_name,
        sources=[str(KERNEL_PATH)], ### Load the .cu file as a source
        extra_cuda_cflags=["-O0"], ### NO optimization, TODO: ["-O3", "--use_fast_math"]
        verbose=False,
    )


# Registers this Python function into a global function registry
@register_func("flashinfer.kernel")
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    return _load_extension().kernel(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
    )
