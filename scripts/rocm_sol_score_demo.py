#!/usr/bin/env python3
"""Demo: compute the AMD (MI300X) SOL-Score for kernels measured live on the GPU.

NVIDIA's SOL-Score uses B200 bounds; this uses the MI300X roofline from
sol_execbench.rocm_roofline so the metric is meaningful on AMD. Kernels are
timed here with HIP events so measured-time and the roofline bound are consistent.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from sol_execbench.rocm_roofline import (  # noqa: E402
    amd_sol_score, gemm_flops_bytes, elementwise_bytes,
)


def time_ms(fn, warmup=20, rep=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    t = sorted(si.elapsed_time(ei) for si, ei in zip(s, e))
    return t[len(t) // 2]  # median


def show(title, r):
    print(f"\n{title}")
    print(f"  regime          : {r['regime']}-bound")
    print(f"  t_sol (MI300X)  : {r['t_sol_ms']:.4f} ms")
    print(f"  measured kernel : {r['measured_ms']:.4f} ms  ({r['achieved_pct_of_sol']:.1f}% of SOL)")
    print(f"  reference       : {r['baseline_ms']:.4f} ms")
    print(f"  AMD SOL-Score   : {r['sol_score']:.3f}   (1.0 = at speed-of-light)")


def main():
    dev = "cuda"
    import aiter

    # 1) RMSNorm (memory-bound): aiter CK rms_norm vs torch reference, [8192,4096] bf16
    B, H = 8192, 4096
    x = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
    w = torch.randn(H, dtype=torch.bfloat16, device=dev)
    def ref_rms():
        xf = x.float(); inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
        return ((xf * inv) * w.float()).to(x.dtype)
    aiter_ms = time_ms(lambda: aiter.rms_norm(x, w, 1e-5))
    ref_ms = time_ms(ref_rms)
    # bytes: read x + write y (w is negligible); ~2 passes of the big tensor
    flops, by = elementwise_bytes(B * H, n_read=1, n_write=1, dtype="bfloat16")
    show(f"RMSNorm [{B}x{H}] bf16 — aiter CK rms_norm",
         amd_sol_score(aiter_ms, ref_ms, flops, by, "bfloat16", "MI300X"))

    # 2) GEMM (compute-bound): aiter tuned gemm vs torch.matmul, 4096^3 bf16
    M = N = K = 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    mm_ms = time_ms(lambda: torch.matmul(a, b))  # hipBLASLt via torch (prod path)
    flops, by = gemm_flops_bytes(M, N, K, dtype="bfloat16")
    show(f"GEMM {M}x{N}x{K} bf16 — torch.matmul (hipBLASLt)",
         amd_sol_score(mm_ms, mm_ms, flops, by, "bfloat16", "MI300X"))


if __name__ == "__main__":
    main()
