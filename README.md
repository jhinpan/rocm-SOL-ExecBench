# rocm-SOL-ExecBench

<div align="center" id="top">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**A ROCm / AMD Instinct port of [NVIDIA SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)** — run the GPU-kernel benchmark on **AMD MI300X (gfx942)** and grade kernels against **AMD speed-of-light**, with the PyTorch reference as the correctness oracle.

</div>

> This is a community ROCm fork. Upstream SOL-ExecBench (NVIDIA) targets NVIDIA Blackwell (B200) with CUDA DSLs and a B200 SOL-Score. This fork makes the **PyTorch + Triton** path run on **AMD ROCm**, maps benchmark ops to **production AMD kernels (aiter: ASM / CK / FlyDSL)**, and adds an **MI300X/MI355X roofline SOL-Score**. All credit for the original framework, dataset, and methodology goes to NVIDIA — see [Upstream](#upstream).

## What this fork adds

| Capability | Where | Status (8× MI300X, ROCm 7.2) |
|---|---|---|
| Run the **PyTorch/Triton path on ROCm** (CUPTI→HIP-event timing, no `uv`/CUDA stack) | [`ROCM_PORT.md`](ROCM_PORT.md) | ✅ 5/5 examples pass |
| **Full-dataset runnability** (PyTorch reference per problem) | [`ROCM_PORT.md`](ROCM_PORT.md) | ✅ **175/176** standard L1+L2 run |
| **Op → production aiter kernel map** (aiter/CK/ASM/FlyDSL) | [`KERNEL_MAPPING.md`](KERNEL_MAPPING.md) | gemm/attention/moe/norm/rope/activation covered |
| **Mapping driver** — wires aiter kernels as solutions, correctness-gated, reports speedup | [`MAPPING_DRIVER.md`](MAPPING_DRIVER.md) | ✅ rmsnorm 3.3×, geglu 1.45×, gated-MLP 1.1×, post-norm 2.3× |
| **AMD SOL-Score** — MI300X/MI355X roofline (NVIDIA's is B200-bound) | [`ROCM_ROOFLINE.md`](ROCM_ROOFLINE.md) | ✅ aiter rmsnorm 61% of memory SOL; hipBLASLt GEMM 47% of BF16 peak |
| `--benchmark-reference` flag (solution-vs-reference speedup) | `src/sol_execbench/cli/main.py` | ✅ |

**Philosophy:** Triton/PyTorch are not what runs in production — the PyTorch reference is kept only as the **correctness oracle**, and real **AMD production kernels (aiter)** are the solutions under test.

## ROCm quickstart (AMD)

Run inside a ROCm PyTorch container (torch already installed; **do not** use `uv` or install the NVIDIA CUDA wheels):

```bash
pip install --no-deps -e .
pip install -r requirements-rocm.txt

# (one-time) download the benchmark dataset
python scripts/download_solexecbench.py

# 1) examples — production-kernel solutions, correctness-verified vs reference
bash scripts/run_rocm_examples.sh

# 2) full-dataset runnability on MI300X (PyTorch reference per problem)
python scripts/make_reference_solutions.py && python scripts/run_rocm_dataset.py

# 3) map ops to aiter kernels, report correctness + speedup
python scripts/run_aiter_mapping.py

# 4) AMD (MI300X) SOL-Score demo / self-test
python scripts/rocm_sol_score_demo.py             # live on GPU
python scripts/rocm_sol_score_demo.py --self-test # CPU-only invariants

# evaluate one solution
sol-execbench <problem_dir> --solution sol.json --benchmark-reference
```

Supported solution languages on ROCm: **PyTorch, Triton**, and **Python wrappers over aiter** (ASM/CK/FlyDSL kernels). CUDA-only DSLs (CUTLASS, cuDNN, CuTe, cuTile, CUDA C++) and the FP8/FP4 (Quant) problems are out of scope for this port (the Quant problems' NVIDIA-specific quant path is not ported; MI300X itself supports FP8).

## Caveats
- **SOL-Score:** upstream grades against a **B200** roofline (SOLAR is NVIDIA-only). This fork adds an **MI300X/MI355X** roofline (`src/sol_execbench/rocm_roofline.py`); AMD SOL-Scores are **not comparable** to NVIDIA's leaderboard.
- **CUPTI** device-timing is replaced by HIP-event timing on ROCm; `nvidia-smi` clock-lock is a no-op on AMD.
- Of the 235 dataset problems, **60 are not runnable** via this path: **33 Quant** (FP8/FP4 numeric formats; 19 detected as `float8_e4m3fn`/`float4_e2m1fn_x2`) and **26 FlashInfer-Bench** (a strict-validation loader quirk, not a GPU gap), plus **1 L2** partial. The standard L1+L2 set (176 problems) is **175/176** runnable — see [`ROCM_PORT.md`](ROCM_PORT.md).

<a name="upstream"></a>
## Upstream (NVIDIA SOL-ExecBench)

The original framework, dataset (`nvidia/SOL-ExecBench`), SOL-Score methodology, and the NVIDIA Docker/CUDA path are unchanged and still present (`scripts/run_docker.sh`, `docker/`, the CUDA DSL examples). Original project: <https://github.com/NVIDIA/SOL-ExecBench> · [dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) · [technical report](https://arxiv.org/abs/2603.19173). License: Apache-2.0 (see `LICENSE`).
