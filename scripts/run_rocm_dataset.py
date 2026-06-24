#!/usr/bin/env python3
"""Run every SOL-ExecBench problem with its PyTorch reference solution on AMD ROCm
and tally how many RUN (all workloads pass) on MI300X. Categorizes failures
(FP8/FP4 dtype vs runtime/other). Writes results/rocm_dataset_runnability.json."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
bench = root / "data" / "benchmark"
probs = sorted(bench.glob("*/*/definition.json"))
res = {}
t0 = time.time()
for i, defjson in enumerate(probs, 1):
    pdir = defjson.parent
    sub = pdir.parent.name
    name = pdir.name
    d = json.load(open(defjson))
    specs = list(d.get("inputs", {}).values()) + list(d.get("outputs", {}).values())
    dtypes = {s.get("dtype", "?") for s in specs}
    is_fp8fp4 = any(("float8" in t or "float4" in t) for t in dtypes)
    sol = pdir / "solution_reference.json"
    try:
        p = subprocess.run(
            ["sol-execbench", str(pdir), "--solution", str(sol)],
            capture_output=True, text=True, timeout=240,
            cwd=str(root),
        )
        out = (p.stdout or "") + (p.stderr or "")
        passed = (p.returncode == 0) and ("workloads passed" in out)
        # extract X/Y
        frac = next((ln.strip() for ln in out.splitlines() if "workloads passed" in ln), "")
        status = "RUN_OK" if passed else ("FP8FP4" if is_fp8fp4 else "FAIL")
        reason = "" if passed else (frac or out.strip().splitlines()[-1][:160] if out.strip() else "no output")
    except subprocess.TimeoutExpired:
        status, frac, reason = ("TIMEOUT", "", "240s timeout")
    res[f"{sub}/{name}"] = {"subset": sub, "fp8fp4": is_fp8fp4, "status": status, "frac": frac, "reason": reason[:160]}
    if i % 20 == 0:
        print(f"  ...{i}/{len(probs)} ({time.time()-t0:.0f}s)", flush=True)

# summarize
from collections import Counter
bysub = {}
for k, v in res.items():
    s = v["subset"]; bysub.setdefault(s, Counter())[v["status"]] += 1
tot = Counter(v["status"] for v in res.values())
summary = {"total": len(res), "by_status": dict(tot),
           "by_subset": {s: dict(c) for s, c in bysub.items()},
           "fp8fp4_problems": sum(1 for v in res.values() if v["fp8fp4"])}
outdir = root / "results"; outdir.mkdir(exist_ok=True)
json.dump({"summary": summary, "per_problem": res}, open(outdir / "rocm_dataset_runnability.json", "w"), indent=2)
print("SUMMARY:", json.dumps(summary, indent=2))
print("DATASET_RUN_DONE")
