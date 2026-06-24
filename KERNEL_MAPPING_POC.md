# PoC: production ROCm kernel as a SOL-ExecBench solution (correctness-verified)

Demonstrates the end goal of the mapping in [`KERNEL_MAPPING.md`](KERNEL_MAPPING.md): replace the Triton/PyTorch solution with a **production aiter kernel**, and let the SOL-ExecBench harness verify it against the problem's **PyTorch reference** on MI300X.

## Example: RMSNorm → aiter CK `rms_norm`
Problem: `examples/triton/rmsnorm` (hidden_states[B,4096] bf16, weight[4096] bf16, eps 1e-5 → bf16).

Solution (`examples/triton/rmsnorm/solution_aiter.json`, entry `kernel.py::run`):
```python
import torch, aiter

@torch.no_grad()
def run(hidden_states, weight):
    # AMD production kernel: aiter CK rmsnorm (eps must match the reference)
    return aiter.rms_norm(hidden_states, weight, 1e-5)
```

Run + result on **8× MI300X (gfx942), ROCm 7.2**:
```
sol-execbench examples/triton/rmsnorm --solution examples/triton/rmsnorm/solution_aiter.json
→ 14/14 workloads PASSED
```
- **Correctness:** verified by the harness vs the PyTorch reference (per-workload `torch.allclose`, rel-err ~7.8e-3 < 1e-2 tol). ✅
- **Latency (HIP-events):** aiter CK rmsnorm ~**0.023 ms** vs the Triton rmsnorm solution ~0.039 ms on the same problem — the production kernel is faster, which is the whole point.

## Why this is the template
1. **Correctness first** — the PyTorch reference is the oracle; the harness gates on it. We never trust a kernel that doesn't match.
2. **Backend swap is just the solution `run()`** — point it at `aiter.rms_norm` (CK), `aiter.gemm_a16w16_asm` (ASM), `aiter.flydsl_flash_attn_func` (FlyDSL), etc., matching the reference's I/O signature and dtypes.
3. `target_hardware: ["MI300X", "LOCAL_AMD"]` (added to `SupportedHardware`).

## Next ops to wire (high-coverage, see KERNEL_MAPPING.md)
- **GEMM/linear** (122 problems) → `aiter.gemm_a16w16_asm` / `tuned_gemm` (hipBLASLt) / CK / FlyDSL `flydsl_hgemm`.
- **Attention** (64) → `aiter.flash_attn_func` (CK/ASM) / `flydsl_flash_attn_func`.
- **MoE** (39) → `aiter.fused_moe` (ASM/CK/FlyDSL).
- **norm/rope/activation** → `aiter` fused kernels (`fused_add_rmsnorm`, `fused_qk_norm_rope`, `activation`).

Fusions specific to a SOL problem that aiter doesn't ship as one kernel are **composed** from aiter primitives; niche ops (2D/3D conv, FFT, fancy indexing) stay on the reference.
