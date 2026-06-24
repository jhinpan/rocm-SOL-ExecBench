# aiter kernel mapping driver

Auto-wires SOL-ExecBench problems to **production aiter ROCm kernels** and verifies them against each problem's PyTorch reference on MI300X (correctness oracle), reporting speedup.

- `scripts/aiter_kernel_map.py` — registry of **signature-based recognizers** (`recognize_rmsnorm`, `recognize_geglu`, `recognize_swiglu`, …). Each inspects a problem's inputs/outputs/reference and, on a match, returns `kernel.py` source whose `run()` calls the aiter kernel. Extend by adding recognizers.
- `scripts/run_aiter_mapping.py` — driver: maps each problem → generates `solution_aiter_auto.json` → runs `sol-execbench … --benchmark-reference --json` → records correctness + median speedup → `results/aiter_mapping_results.json`.

## Run
```bash
python scripts/run_aiter_mapping.py          # examples + L1 + L2
python scripts/run_aiter_mapping.py '<glob>/definition.json'
```

## Verified results (8× MI300X, ROCm 7.2)
| Problem | aiter kernel | backend | correctness | speedup vs reference |
|---|---|---|---|---|
| `L1/085_geglu_activation` | `gelu_tanh_and_mul` | HIP | 16/16 ✅ | **1.46×** |
| `examples/cuda_cpp/rmsnorm` | `rms_norm` | CK | 14/14 ✅ | **3.25×** |
| `examples/triton/rmsnorm` | `rms_norm` | CK | 14/14 ✅ | **3.30×** |

The driver **gates on correctness** — a result is a win only when *every* workload row PASSED. A mismatch (e.g. wrong GELU variant), a solution crash, empty/garbled harness output, a missing `sol-execbench`, or a timeout all report FAIL and are excluded, never counted as a win.

## Recognizer guards
Recognizers are deliberately conservative (a false negative only costs coverage; a false positive would hand a fused problem a single-op kernel). Each requires an exact input/output arity, requires the op's core signature (e.g. RMSNorm needs `rsqrt`, `mean`, and an explicit `x**2`/`pow(2)`), and rejects any reference containing fused/composite tokens (`attention`, `softmax`, `matmul`, `rope`, `conv`, `residual`, `variance`/layernorm, `dropout`, `embedding`, …). GEGLU vs SwiGLU are disambiguated by the gate activation (`gelu` vs `silu`). Verified across the full L1+L2+FlashInfer dataset: zero false positives.

Lint: `ruff check scripts/aiter_kernel_map.py scripts/run_aiter_mapping.py` passes clean.

## Scope
Recognizers here cover **clean single ops**. The SOL dataset is dominated by **fused/composite** chains (e.g. attention+rope+qk_norm, conv+groupnorm+silu+residual), so single-op dataset coverage is small by design — those are handled by composing aiter primitives / aiter fused kernels (`fused_add_rmsnorm`, `fused_qk_norm_rope`, `silu_and_mul`, gated FF) in the **L2-fusion** work. This PR establishes the extensible framework + verified speedups; coverage grows by adding recognizers. See `KERNEL_MAPPING.md` for the full op→kernel map.

## L2 fusion recognizers
Beyond single ops, the registry composes aiter primitives for fused chains:
- `recognize_gated_mlp_silu` — `linear(x, gate_up) → silu_and_mul → linear(_, down)`: maps the gate to aiter **`silu_and_mul`** (fused) + hipBLASLt linears.
- `recognize_post_norm_residual` — `residual + RMSNorm(x)`: maps to aiter CK **`rms_norm`** (2D-flattened) + add.

Verified fusion wins (8× MI300X, correctness-gated):
| problem | composed kernel | correct | speedup |
|---|---|---|---|
| `L1/074_fused_gated_mlp_silu` | silu_and_mul + hipBLASLt linear | 16/16 ✅ | 1.09× |
| `examples/triton/olmo3_post_norm` | aiter CK rms_norm + add | 3/3 ✅ | 2.27× |

The gate excludes cases where aiter's rmsnorm differs from the reference on some workloads (`L1/033` 12/16, `nemotron_rms_norm` 0/3) — never counted as wins. This is by design: composition correctness is verified per-workload against the PyTorch reference.
