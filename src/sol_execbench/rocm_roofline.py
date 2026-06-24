# SPDX-License-Identifier: Apache-2.0
"""AMD ROCm roofline + Speed-of-Light bounds for SOL-ExecBench.

NVIDIA's SOL-Score grades a kernel against an analytically-derived B200
speed-of-light bound produced by SOLAR (NVIDIA-only). On AMD that bound is
meaningless. This module supplies **MI300X / MI355X** roofline bounds so an
AMD SOL-Score can be computed:

    t_sol = max(flops / peak_flops(dtype),  bytes_moved / peak_bw)   [seconds]

i.e. the larger of the compute-bound and memory-bound ideal times. The anchored
score reuses :func:`sol_execbench.sol_score.sol_score`.

Peak numbers are vendor-published dense peaks (no sparsity):
- MI300X (CDNA3, gfx942): HBM3 5.3 TB/s; matrix BF16/FP16 1307 TFLOP/s,
  FP8 2615 TFLOP/s, FP32 163 TFLOP/s, FP64 163 TFLOP/s.
- MI355X (CDNA4, gfx950): HBM3e 8.0 TB/s; matrix BF16/FP16 2516.6 TFLOP/s,
  FP8 5033.2 TFLOP/s, FP4 10066.3 TFLOP/s, FP32 157.3 TFLOP/s, FP64 78.6
  TFLOP/s (AMD datasheet, dense / no sparsity).
"""
from __future__ import annotations

from dataclasses import dataclass

from sol_execbench.sol_score import sol_score

_TFLOP = 1.0e12


@dataclass(frozen=True)
class GpuSpec:
    name: str
    hbm_bw_bytes_per_s: float          # peak memory bandwidth (B/s)
    peak_flops: dict[str, float]       # dtype -> peak dense FLOP/s


GPU_SPECS: dict[str, GpuSpec] = {
    "MI300X": GpuSpec(
        name="AMD Instinct MI300X (CDNA3, gfx942)",
        hbm_bw_bytes_per_s=5.3e12,
        peak_flops={
            "float16": 1307.4 * _TFLOP, "bfloat16": 1307.4 * _TFLOP,
            "float8_e4m3fn": 2614.9 * _TFLOP, "float8_e5m2": 2614.9 * _TFLOP,
            "float32": 163.4 * _TFLOP, "float64": 163.4 * _TFLOP,
        },
    ),
    "MI355X": GpuSpec(
        name="AMD Instinct MI355X (CDNA4, gfx950)",
        hbm_bw_bytes_per_s=8.0e12,
        peak_flops={
            "float16": 2516.6 * _TFLOP, "bfloat16": 2516.6 * _TFLOP,
            "float8_e4m3fn": 5033.2 * _TFLOP, "float8_e5m2": 5033.2 * _TFLOP,
            "float4_e2m1fn_x2": 10066.3 * _TFLOP,
            "float32": 157.3 * _TFLOP, "float64": 78.6 * _TFLOP,
        },
    ),
}

_DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "bool": 1,
    "int32": 4, "int64": 8, "float4_e2m1fn_x2": 1,  # packed 2x4-bit per byte
}


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype, 2)


def roofline_time_s(flops: float, bytes_moved: float, compute_dtype: str,
                    gpu: str = "MI300X") -> float:
    """Ideal (speed-of-light) execution time in seconds for the given GPU."""
    spec = GPU_SPECS[gpu]
    peak = spec.peak_flops.get(compute_dtype, spec.peak_flops.get("bfloat16"))
    t_compute = (flops / peak) if (flops and peak) else 0.0
    t_memory = (bytes_moved / spec.hbm_bw_bytes_per_s) if bytes_moved else 0.0
    return max(t_compute, t_memory)


def amd_sol_score(measured_ms: float, baseline_ms: float, flops: float,
                  bytes_moved: float, compute_dtype: str, gpu: str = "MI300X") -> dict:
    """Compute the AMD SOL-Score for a measured kernel.

    Returns t_sol (ms), the bound regime (compute/memory), and the anchored
    SOL-Score in [0, 1] (1.0 == at speed-of-light).
    """
    spec = GPU_SPECS[gpu]
    peak = spec.peak_flops.get(compute_dtype, spec.peak_flops.get("bfloat16"))
    t_compute = (flops / peak) if (flops and peak) else 0.0
    t_memory = (bytes_moved / spec.hbm_bw_bytes_per_s) if bytes_moved else 0.0
    t_sol_s = max(t_compute, t_memory)
    regime = "compute" if t_compute >= t_memory else "memory"
    # SOL is a hard ceiling: a kernel cannot beat it. Clamp to [0, 1]; a measured
    # time below t_sol means the bound is under-estimated for this op.
    score = min(1.0, max(0.0, sol_score(measured_ms, baseline_ms, t_sol_s * 1e3)))
    achieved = (t_sol_s * 1e3 / measured_ms * 100.0) if measured_ms > 0 else 0.0
    return {
        "gpu": gpu, "t_sol_ms": t_sol_s * 1e3, "regime": regime,
        "measured_ms": measured_ms, "baseline_ms": baseline_ms,
        "achieved_pct_of_sol": min(100.0, achieved),
        "sol_score": score,
    }


# --- FLOPs / bytes estimators for common op classes -------------------------

def gemm_flops_bytes(m: int, n: int, k: int, dtype: str = "bfloat16",
                     out_dtype: str | None = None) -> tuple[float, float]:
    """Dense GEMM (M,K)x(K,N): 2*M*N*K flops; bytes = A+B read + C write."""
    ib = dtype_bytes(dtype)
    ob = dtype_bytes(out_dtype or dtype)
    flops = 2.0 * m * n * k
    bytes_moved = (m * k + k * n) * ib + (m * n) * ob
    return flops, bytes_moved


def elementwise_bytes(num_elems: int, n_read: int, n_write: int,
                      dtype: str = "bfloat16") -> tuple[float, float]:
    """Memory-bound elementwise/norm: ~0 useful matmul flops; bytes dominate."""
    b = dtype_bytes(dtype)
    return 0.0, float(num_elems) * (n_read + n_write) * b
