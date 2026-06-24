# rocm-SOL-ExecBench — ROCm/AMD port (PyTorch + Triton path)

A fork of [NVIDIA/SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) that runs the **PyTorch and Triton** evaluation path on **AMD Instinct (ROCm)** GPUs. Verified on **8× MI300X (gfx942), ROCm 7.2, PyTorch 2.9.1+rocm7.2**.

## Status — what works

| Capability | ROCm status |
|---|---|
| PyTorch solutions (correctness + timing) | ✅ works |
| Triton solutions (correctness + timing) | ✅ works (Triton's ROCm/HIP backend) |
| CUDA/HIP-event timing | ✅ works (`torch.cuda.Event` → `hipEvent`) |
| Correctness vs reference (tolerances) | ✅ works |
| Reward-hack checks | ✅ works (vendor-agnostic) |
| CUTLASS / cuDNN / CuTe DSL / cuTile / CUDA C++ solutions | ❌ NVIDIA-only (need nvcc/CUDA) — out of scope |
| CUPTI timing (`methodology="cupti"`) | ⚠️ unavailable on ROCm → **auto-downgraded to `cuda_events`** |
| `nvidia-smi` clock locking (`--lock-clocks`) | ⚠️ NVIDIA-only → leave off (default) on AMD |
| **SOL-Score** (vs B200 speed-of-light) | ⚠️ `t_sol`/`t_b` are **B200-derived data**; the *score on AMD is not comparable* to NVIDIA's. Correctness + measured latency are valid; the SOL number is not. |

### Verified examples on MI300X
All five PyTorch/Triton examples pass (`bash scripts/run_rocm_examples.sh`):

| Example | Lang | Result |
|---|---|---|
| `examples/pytorch/linear_backward` | PyTorch | 3/3 workloads passed |
| `examples/pytorch/gemma3_swiglu` | PyTorch | 3/3 workloads passed |
| `examples/triton/rmsnorm` | Triton | 14/14 workloads passed |
| `examples/triton/nemotron_rms_norm` | Triton | 3/3 workloads passed |
| `examples/triton/olmo3_post_norm` | Triton | 3/3 workloads passed |

## Full-dataset runnability on MI300X (235 problems)

The SOL-ExecBench dataset (`nvidia/SOL-ExecBench`) ships `definition.json` + `reference.py` + `workload.jsonl` per problem but **no solutions** (it's a generation benchmark). To measure how much of it runs on AMD, we generate a PyTorch **reference-as-solution** for every problem (`scripts/make_reference_solutions.py`) and run the whole set through the ported harness (`scripts/run_rocm_dataset.py`). This validates that each problem's inputs/workloads/reference execute + time on MI300X (correctness trivially passes; speedup ≈ 1×).

| Subset | Problems | **Runs on MI300X** | Notes |
|---|--:|--:|---|
| **L1** single-op | 94 | **94 (100%)** | 1 needed >240 s (passes at 600 s) |
| **L2** fusion | 82 | **81 (99%)** | `L2/033` partial (13/16 workloads); 3 needed 600 s |
| **Quant** | 33 | **0** | FP8/FP4 (`float8_e4m3fn` / `float4_e2m1fn_x2`) — these problems' quant op/numeric-format path is NVIDIA-specific and not ported here (the underlying FP8 dtype is otherwise supported on MI300X/CDNA3) |
| **FlashInfer-Bench** | 26 | **0** | `Definition` has an empty-string field → fails pydantic `min_length` at load (data/loader issue, not GPU; would fail on NVIDIA too) |
| **Total** | 235 | **175** | |

**Takeaway:** the **standard L1+L2 SOL-ExecBench (176 problems) is ~99% runnable on MI300X (175/176)** via its PyTorch reference. The non-runnable 60 are entirely FP8/FP4 numeric formats (Quant) or a strict-validation data quirk (FlashInfer-Bench) — **not** an MI300X capability gap. Reproduce: `python scripts/make_reference_solutions.py && python scripts/run_rocm_dataset.py` (writes `results/rocm_dataset_runnability.json`).

> This measures *runnability*, not kernel quality. A real benchmark score needs generated solutions (e.g., via an LLM) evaluated against these references — and a meaningful **SOL-Score requires MI300X (CDNA3) roofline bounds**, which NVIDIA's SOLAR does not produce (see caveats).

## What was changed for ROCm
- **`core/bench/timing.py`** — guarded the module-level `from cupti import cupti` (CUPTI is NVIDIA-only, absent on ROCm) behind a `try/except` (`_HAS_CUPTI`); changed `time_runnable` default `methodology` to `cuda_events`; auto-downgrade `cupti → cuda_events` when CUPTI is missing. The CUDA-event path runs unchanged on ROCm via PyTorch's HIP shim.
- **`pyproject.toml`** — relaxed `requires-python` to `>=3.10` (ROCm images commonly ship 3.10).
- Added **`requirements-rocm.txt`** (portable deps only — do **not** install the NVIDIA stack) and **`scripts/run_rocm_examples.sh`**.

Everything else (CUDA C++/CUTLASS/cuDNN/cuTile compile path, `nvidia-smi` clock lock, B200 clock presets, SOL bound generation) is left intact but **not used** on the PyTorch/Triton ROCm path.

## Install on a ROCm box (do NOT use `uv sync`)
`uv sync` pins `torch==2.9.0+cu130` and pulls the NVIDIA CUDA-13 wheel stack (cupti-python, cuda-tile, nvidia-cutlass-dsl, nvidia-cudnn-frontend), which would clobber the ROCm PyTorch. Instead, use the ROCm PyTorch already in the image:

```bash
# in a ROCm PyTorch container (torch already installed, e.g. 2.9.x+rocm)
pip install --no-deps -e .
pip install -r requirements-rocm.txt
```

## Run
```bash
# single problem
sol-execbench examples/triton/rmsnorm --solution examples/triton/rmsnorm/solution_triton.json --verbose
# all ROCm-portable examples
bash scripts/run_rocm_examples.sh
```

## Caveats / not yet ported
- **SOL-Score is meaningless on AMD** until MI300X (CDNA3) speed-of-light bounds are supplied — NVIDIA's SOLAR bound generator is not available; `t_sol`/`t_b` in the dataset are B200 values. Use correctness + measured latency; treat any SOL number as N/A on AMD.
- **CUPTI-accurate kernel timing** is replaced by CUDA/HIP-event timing (includes a little CPU-launch overhead vs CUPTI's device-only timestamps). A true ROCm equivalent would reimplement the CUPTI path against **roctracer / rocprofiler-sdk**.
- FP8 / NVFP4 problems and the CUDA-only solution languages are out of scope for this port.
- `--lock-clocks` requires `nvidia-smi`; on AMD leave it off (a `rocm-smi`-based equivalent could be added).
