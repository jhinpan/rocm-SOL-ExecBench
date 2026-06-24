#!/usr/bin/env python3
"""Mapping driver: for every problem that a recognizer in aiter_kernel_map maps
to a production aiter kernel, generate the solution, run it through SOL-ExecBench
on MI300X with --benchmark-reference, and report correctness + speedup vs the
PyTorch reference. Writes results/aiter_mapping_results.json.

Usage: python scripts/run_aiter_mapping.py [problem_dir_glob ...]
Default scans examples/ and data/benchmark/L1,L2.
"""
from __future__ import annotations
import glob, json, statistics, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_kernel_map import map_problem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def find_problems(patterns):
    dirs = []
    for pat in patterns:
        for defjson in glob.glob(pat, recursive=True):
            dirs.append(Path(defjson).parent)
    return sorted(set(dirs))


def run_one(pdir: Path) -> dict | None:
    defjson = pdir / "definition.json"
    if not defjson.exists():
        return None
    d = json.load(open(defjson))
    m = map_problem(d)
    if not m:
        return None
    # write solution
    sol = {
        "name": f"{d['name']}_aiter", "definition": d["name"], "author": "aiter-mapping-driver",
        "spec": {"languages": ["pytorch"], "target_hardware": ["MI300X", "LOCAL_AMD"],
                 "entry_point": "kernel.py::run", "dependencies": ["torch", "aiter"],
                 "destination_passing_style": False},
        "sources": [{"path": "kernel.py", "content": m["kernel"]}],
    }
    solpath = pdir / "solution_aiter_auto.json"
    json.dump(sol, open(solpath, "w"), indent=2)
    try:
        p = subprocess.run(
            ["sol-execbench", str(pdir), "--solution", str(solpath), "--benchmark-reference", "--json"],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        traces = [json.loads(ln) for ln in (p.stdout or "").splitlines() if ln.strip().startswith("{")]
        evals = [t.get("evaluation") for t in traces if t.get("evaluation")]
        passed = sum(1 for e in evals if e.get("status") == "PASSED")
        total = len(evals)
        sp = [e["performance"]["speedup_factor"] for e in evals
              if e.get("status") == "PASSED" and e.get("performance", {}).get("speedup_factor")]
        med_speedup = round(statistics.median(sp), 2) if sp else None
        all_pass = total > 0 and passed == total
    except subprocess.TimeoutExpired:
        passed = total = 0; med_speedup = None; all_pass = False
    return {"problem": f"{pdir.parent.name}/{pdir.name}", "recognizer": m["recognizer"],
            "aiter_fn": m["aiter_fn"], "backend": m["backend"],
            "passed": passed, "total": total, "all_pass": all_pass, "median_speedup": med_speedup}


def main():
    pats = sys.argv[1:] or [
        str(ROOT / "examples/**/definition.json"),
        str(ROOT / "data/benchmark/L1/*/definition.json"),
        str(ROOT / "data/benchmark/L2/*/definition.json"),
    ]
    results = []
    for pdir in find_problems(pats):
        r = run_one(pdir)
        if r:
            sp = f"{r['median_speedup']}x" if r["median_speedup"] else "n/a"
            print(f"[{'PASS' if r['all_pass'] else 'FAIL'}] {r['problem']:<55} {r['aiter_fn']:<14} "
                  f"{r['passed']}/{r['total']}  speedup={sp}", flush=True)
            results.append(r)
    mapped = len(results)
    correct = sum(1 for r in results if r["all_pass"])
    summary = {"mapped": mapped, "correct": correct,
               "speedups": {r["problem"]: r["median_speedup"] for r in results if r["all_pass"] and r["median_speedup"]}}
    outdir = ROOT / "results"; outdir.mkdir(exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(outdir / "aiter_mapping_results.json", "w"), indent=2)
    print(f"\n=== mapped {mapped} problems to aiter kernels; {correct} correct on MI300X ===")
    for k, v in summary["speedups"].items():
        print(f"   {k}: {v}x vs PyTorch reference")


if __name__ == "__main__":
    main()
