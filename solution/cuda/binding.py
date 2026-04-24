"""TVM FFI binding that lazily builds and dispatches the naive MoE CUDA extension."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from tvm.ffi import register_func


THIS_DIR = Path(__file__).resolve().parent
KERNEL_PATH = THIS_DIR / "kernel.cu"
SOURCE_PATHS = tuple(
    path
    for path in sorted(THIS_DIR.rglob("*"))
    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() not in ("", "0", "false", "off", "no")


def _cuda_cflags(profile_enabled: bool) -> list[str]:
    flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++21",
        "-lineinfo",
        # "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_100,code=sm_100",
    ]
    if profile_enabled:
        flags.append("-DFUSED_MOE_PROFILE=1")
    return flags

# Hash-based JIT extension loading.
@lru_cache(maxsize=2)
def _load_extension(profile_enabled: bool):
    hasher = hashlib.sha1()
    hasher.update(f"profile={int(profile_enabled)}".encode())
    for path in SOURCE_PATHS:
        hasher.update(str(path.relative_to(THIS_DIR)).encode())
        hasher.update(path.read_bytes())
    source_hash = hasher.hexdigest()[:12]
    module_name = f"flashinfer_moe_cuda_{source_hash}"
    return load(
        name=module_name,
        sources=[str(KERNEL_PATH)],
        extra_cuda_cflags=_cuda_cflags(profile_enabled),
        verbose=False,
    )


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
    return _load_extension(_env_flag("FUSED_MOE_PROFILE")).kernel(
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
