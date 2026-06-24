# SOL-ExecBench op → production ROCm kernel mapping (MI300X)

**Goal:** run SOL-ExecBench's L1/L2 ops on AMD with **real production kernels** (not Triton/PyTorch, which aren't used in prod), using each problem's **PyTorch reference purely as the correctness oracle**. This maps the 176 L1+L2 ops onto kernels available in **aiter** (umbrella prod lib) and its backends **ASM / CK / ck_tile / FlyDSL(+Atom)**, plus Composable Kernel directly.

> "atom" = the FlyDSL **`Atom`** tile primitive used inside aiter's FlyDSL kernels (`aiter/ops/flydsl/kernels/*`), not a separate library.

## Demand: what the 176 L1+L2 problems need (multi-label)

| Category | L1 | L2 | total |
|---|--:|--:|--:|
| gemm / linear (proj, bmm, einsum) | 56 | 66 | **122** |
| activation (silu/gelu/swiglu/glu/…) | 27 | 52 | **79** |
| norm (rms/layer/group/instance) | 26 | 35 | **61** |
| attention-fwd | 14 | 35 | **49** |
| softmax / dropout | 16 | 33 | **49** |
| elementwise / residual / add | 25 | 19 | **44** |
| rope | 23 | 19 | **42** |
| moe (route/dispatch/experts/topk) | 13 | 26 | **39** |
| conv / causal-conv (1d/2d/3d) | 11 | 14 | **25** |
| attention-bwd | 3 | 12 | **15** |

## Supply → coverage matrix

| Category | aiter prod kernel(s) | Backends available | MI300X coverage |
|---|---|---|---|
| **gemm / linear** | `gemm_a16w16`, `gemm_a8w8(_blockscale/_bpreshuffle)`, `gemm_a4w4`, `batched_gemm_*`, `tuned_gemm`(hipBLASLt/rocBLAS), `deepgemm`, `wvSplitK` skinny | **ASM, CK, ck_tile, FlyDSL, opus, triton, hipBLASLt** | ✅ **full** (best-covered; ASM + CK + FlyDSL all present) |
| **attention-fwd** (MHA/GQA/varlen) | `mha_fwd`, `mha_varlen_fwd`, `fmha_v3_fwd` (asm), `flydsl_flash_attn_func`, `flash_attn_func` | **CK (ck_tile FMHA), ASM (fmha_v3), FlyDSL, triton** | ✅ **full** |
| **attention-bwd** | `mha_bwd`, `mha_varlen_bwd`, `fmha_v3_bwd` | **CK, ASM, triton** | ✅ full |
| **paged-attn / MLA** | `pa_fwd_asm`, `paged_attention_v1/v2`, `mla_decode/prefill_asm_fwd` | **ASM, HIP, triton** | ✅ (decode/prefill) |
| **moe** | `fused_moe`, `fused_moe_bf16_asm`, ck_tile `gemm_moe_2stages`, FlyDSL `moe_kernels`, `moe_sorting`, `asm_topksoftmax` | **ASM, CK, FlyDSL, triton** | ✅ **full** |
| **norm** (rms/layer/group) | `rmsnorm`, `norm`, `groupnorm`, `fused_add_rmsnorm`, `gated_rmsnorm_*`, `fused_qk_rmsnorm` | **HIP/CK + fused** | ✅ rms/layer/group; ◐ instance/GRN |
| **rope** | `rope`, `pos_encoding`, `fused_qk_norm_rope_*` (FlyDSL), `fused_qk_norm_mrope` | **HIP, FlyDSL** | ✅ (1D/multi-axis; 3D vision = compose) |
| **activation** (silu/gelu/swiglu) | `activation.py` (silu/gelu/swiglu/…) | **HIP, triton** | ✅ (usually fused into gemm/ff) |
| **elementwise / residual** | fused variants: `fused_add_rmsnorm`, `fused_gemm_*_mul_add` | **HIP, triton** | ◐ compose from primitives |
| **conv** | `causal_conv1d` ✅; 2D/3D conv via CK conv or torch | **HIP (causal-1d), CK (conv)** | ◐ 1D yes; 2D/3D weak |
| **softmax / dropout** | inside attention; `asm_topksoftmax`; standalone softmax (triton) | **ASM, triton** | ◐ usually fused |
| **embedding/indexing, FFT/rfft, mask-gen, instance-norm** | — | — | ✗ niche → keep torch/reference |

### Backend inventory (where the kernels live)
- **ASM** (`aiter/csrc/py_itfs_cu/asm_*.cu`, exposed via `aiter/ops/*`): `asm_gemm_a16w16/a4w4/a8w8_blockscale_bpreshuffle`, `asm_mha_varlen_fwd/bwd`, `asm_pa` (paged-attn), `asm_mla`, `asm_topk_per_row_*`, `asm_topksoftmax`. gfx942/gfx950, bf16/fp16/fp8/fp4. (2810 prebuilt `*.hsaco`.)
- **CK / ck_tile** (`/opt/rocm/include/ck*`, `aiter/3rdparty/composable_kernel`, `aiter/csrc/ck*`): GEMM (incl. epilogue fusion), grouped/batched GEMM, fused-MoE GEMM (`ck_tile_gemm_moe_2stages`), FMHA, normalization, reduction/softmax.
- **FlyDSL (+Atom)** (`aiter/ops/flydsl/`): `flydsl_hgemm`/splitk, `flydsl_flash_attn_func`, `linear_attention` prefill/decode, `moe_kernels`, `qk_norm_rope_quant`, `fused_compress_attn` — `Atom` is its tile primitive.
- **opus** (`aiter/ops/opus/`): high-perf `gemm_a16w16_opus` (gfx942/gfx950).

## Bottom line
- **The dominant LLM compute — GEMM (122), attention (64), MoE (39), norm (61), RoPE (42), activation (79) — has strong aiter coverage with ASM/CK/FlyDSL backends on MI300X.** These are exactly the ops that matter for prod.
- **L2 fusions** are the main effort: SOL problems are often *specific* fused chains; aiter ships some exact fusions (`fused_add_rmsnorm`, `fused_qk_norm_rope`, gated FF) but many fusions must be **composed** from aiter primitives.
- **Niche/vision ops** (2D/3D conv, FFT/rfft, fancy indexing, instance-norm, mask-gen) have no prod kernel → leave as torch/reference.

## Method (correctness-first)
For each problem: keep the PyTorch **reference** as the oracle; provide a **solution** whose `run()` calls the aiter/CK/ASM/FlyDSL kernel with matching I/O; the harness verifies correctness (`torch.allclose`, per-workload tolerances) on MI300X, then times the prod kernel. See `KERNEL_MAPPING_POC.md` for a worked example.
