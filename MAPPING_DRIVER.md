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

The driver **gates on correctness** — a mismatch (e.g. wrong GELU variant) is reported as FAIL and excluded, never counted as a win.

## Scope
Recognizers here cover **clean single ops**. The SOL dataset is dominated by **fused/composite** chains (e.g. attention+rope+qk_norm, conv+groupnorm+silu+residual), so single-op dataset coverage is small by design — those are handled by composing aiter primitives / aiter fused kernels (`fused_add_rmsnorm`, `fused_qk_norm_rope`, `silu_and_mul`, gated FF) in the **L2-fusion** work. This PR establishes the extensible framework + verified speedups; coverage grows by adding recognizers. See `KERNEL_MAPPING.md` for the full op→kernel map.
