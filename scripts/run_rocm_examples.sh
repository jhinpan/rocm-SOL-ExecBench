#!/usr/bin/env bash
# Run the PyTorch/Triton SOL-ExecBench examples on AMD ROCm (MI300X/gfx942).
# CUPTI timing is auto-downgraded to CUDA/HIP-events; clock-lock is off by default.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="${RUN:-sol-execbench}"
command -v "$RUN" >/dev/null 2>&1 || RUN="python3 -m sol_execbench.cli.main"
EXAMPLES=(
  "examples/pytorch/linear_backward:solution_python.json"
  "examples/pytorch/gemma3_swiglu:solution_python.json"
  "examples/triton/rmsnorm:solution_triton.json"
  "examples/triton/nemotron_rms_norm:solution_triton.json"
  "examples/triton/olmo3_post_norm:solution_triton.json"
)
pass=0; total=0
for e in "${EXAMPLES[@]}"; do
  d="${e%%:*}"; s="${e##*:}"; total=$((total+1))
  out=$(timeout 300 $RUN "$d" --solution "$d/$s" 2>&1 | grep -E "workloads passed|Error|Traceback" | tail -1)
  echo "[$d] $out"
  echo "$out" | grep -qE "^[0-9]+/[0-9]+ workloads passed" && pass=$((pass+1))
done
echo "=== ROCm examples: $pass/$total problems all-workloads-passed ==="
