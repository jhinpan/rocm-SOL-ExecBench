#!/usr/bin/env python3
"""Generate a PyTorch 'reference solution' for every SOL-ExecBench problem so the
benchmark harness can be exercised on AMD ROCm (the dataset ships definition +
reference + workloads but NO solutions). The solution simply reuses the problem's
own PyTorch reference run() -> correctness trivially passes; this measures whether
each problem (inputs/workloads/reference) RUNS + times on MI300X."""
from __future__ import annotations
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "data" / "benchmark"
n = 0
for defjson in sorted(root.glob("*/*/definition.json")):
    d = json.load(open(defjson))
    ref = d["reference"]                      # python string defining top-level run(...)
    sol = {
        "name": f"{d['name']}_ref_pytorch",
        "definition": d["name"],
        "author": "rocm-port-reference",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "kernel.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": False,
        },
        "sources": [{"path": "kernel.py", "content": ref}],
    }
    out = defjson.parent / "solution_reference.json"
    json.dump(sol, open(out, "w"), indent=2)
    n += 1
print(f"wrote {n} reference solutions")
