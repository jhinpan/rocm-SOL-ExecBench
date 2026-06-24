# AMD MI300X roofline / SOL-Score

NVIDIA's SOL-Score grades a kernel against a **B200** speed-of-light bound (from SOLAR, NVIDIA-only), so it's meaningless on AMD. This adds an **MI300X / MI355X roofline** so an AMD SOL-Score can be computed.

`src/sol_execbench/rocm_roofline.py`:
- `GPU_SPECS` — MI300X (CDNA3): HBM3 5.3 TB/s; BF16/FP16 1307, FP8 2615, FP32/FP64 163 TFLOP/s. MI355X (CDNA4): 8 TB/s; BF16/FP16 2516.6, FP8 5033.2, FP4 10066.3, FP32 157.3, FP64 78.6 TFLOP/s (AMD datasheet, dense / no sparsity).
- `roofline_time_s(flops, bytes, dtype, gpu)` → `t_sol = max(flops/peak_flops, bytes/peak_bw)`.
- `amd_sol_score(measured_ms, baseline_ms, flops, bytes, dtype, gpu)` → `{t_sol_ms, regime, achieved_pct_of_sol, sol_score}` (reuses `sol_score.sol_score`, clamped to [0,1]).
- `gemm_flops_bytes(M,N,K,…)`, `elementwise_bytes(n, n_read, n_write, …)` estimators.

## Live demo on MI300X (`scripts/rocm_sol_score_demo.py`, HIP-event timed)
| Op | regime | t_sol | measured | % of SOL | SOL-Score |
|---|---|---|---|---|---|
| RMSNorm [8192×4096] bf16 — aiter CK `rms_norm` | memory | 0.0253 ms | ~0.041 ms | ~61% | ~0.95 |
| GEMM 4096³ bf16 — hipBLASLt `matmul` | compute | 0.1051 ms | ~0.225 ms | ~47% | 0.500 |

→ A real AMD speed-of-light metric: aiter rmsnorm reaches ~60% of HBM SOL; hipBLASLt GEMM ~47% of BF16 compute peak. The demo asserts the physical invariants live (regime, `measured >= t_sol`, score in `[0,1]`, achieved `<= 100%`).

`python scripts/rocm_sol_score_demo.py --self-test` runs the same invariant checks on CPU (no GPU/torch needed).

## Caveats
The GEMM row scores exactly 0.500 because the demo uses `torch.matmul` as *both* candidate and baseline (`T_k == T_b`), where `sol_score` is 0.5 by construction; supply a slower reference to get a discriminating score.

Peaks are vendor dense peaks (no sparsity). Per-problem FLOPs/bytes must be supplied (estimators given for gemm + elementwise; extend per op). This replaces the B200-bound SOL-Score with an MI300X-grounded one; it is not comparable to NVIDIA's leaderboard.
