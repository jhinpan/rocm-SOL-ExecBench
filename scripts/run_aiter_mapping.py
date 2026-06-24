#!/usr/bin/env python3
"""Mapping driver: for every problem that a recognizer in aiter_kernel_map maps
to a production aiter kernel, generate the solution, run it through SOL-ExecBench
on MI300X with --benchmark-reference, and report correctness + speedup vs the
PyTorch reference. Writes results/aiter_mapping_results.json.

The harness is the correctness oracle: a result is only counted as a win when
EVERY workload row PASSED (all_pass). A mismatch, a crash, an empty/garbled
harness output, or a timeout all yield all_pass=False and are never reported as
a speedup.

Usage: python scripts/run_aiter_mapping.py [problem_dir_glob ...]
Default scans examples/ and data/benchmark/L1,L2.
"""
from __future__ import annotations

import glob
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_kernel_map import map_problem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 300


def find_problems(patterns):
    dirs = []
    for pat in patterns:
        for defjson in glob.glob(pat, recursive=True):
            dirs.append(Path(defjson).parent)
    return sorted(set(dirs))


def _parse_traces(stdout: str):
    """Parse one JSON object per line, skipping non-JSON / malformed lines."""
    traces = []
    for ln in (stdout or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            traces.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return traces


def _fail(pdir, m, reason):
    return {
        "problem": f"{pdir.parent.name}/{pdir.name}", "recognizer": m["recognizer"],
        "aiter_fn": m["aiter_fn"], "backend": m["backend"],
        "passed": 0, "total": 0, "all_pass": False, "median_speedup": None,
        "error": reason,
    }


def run_one(pdir: Path) -> dict | None:
    defjson = pdir / "definition.json"
    if not defjson.exists():
        return None
    try:
        with open(defjson) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Unreadable definition: nothing to map, skip silently.
        return None
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
    with open(solpath, "w") as f:
        json.dump(sol, f, indent=2)
    try:
        p = subprocess.run(
            ["sol-execbench", str(pdir), "--solution", str(solpath),
             "--benchmark-reference", "--json"],
            capture_output=True, text=True, timeout=TIMEOUT_S, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return _fail(pdir, m, f"timeout after {TIMEOUT_S}s")
    except FileNotFoundError:
        return _fail(pdir, m, "sol-execbench not found on PATH")

    traces = _parse_traces(p.stdout)
    evals = [t.get("evaluation") for t in traces if t.get("evaluation")]
    if not evals:
        # No parseable evaluations: surface the harness's stderr tail to help debug.
        err = (p.stderr or "").strip().splitlines()
        tail = err[-1] if err else f"no evaluations (rc={p.returncode})"
        return _fail(pdir, m, tail)

    passed = sum(1 for e in evals if e.get("status") == "PASSED")
    total = len(evals)
    sp = [e["performance"]["speedup_factor"] for e in evals
          if e.get("status") == "PASSED" and e.get("performance", {}).get("speedup_factor")]
    med_speedup = round(statistics.median(sp), 2) if sp else None
    all_pass = total > 0 and passed == total
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
        if not r:
            continue
        sp = f"{r['median_speedup']}x" if r.get("median_speedup") else "n/a"
        status = "PASS" if r.get("all_pass") else "FAIL"
        note = "" if r.get("all_pass") else f"  ({r.get('error', 'mismatch')})"
        print(f"[{status}] {r['problem']:<55} {r.get('aiter_fn', '?'):<14} "
              f"{r.get('passed', 0)}/{r.get('total', 0)}  speedup={sp}{note}", flush=True)
        results.append(r)
    mapped = len(results)
    correct = sum(1 for r in results if r.get("all_pass"))
    summary = {"mapped": mapped, "correct": correct,
               "speedups": {r["problem"]: r["median_speedup"]
                            for r in results if r.get("all_pass") and r.get("median_speedup")}}
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "aiter_mapping_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n=== mapped {mapped} problems to aiter kernels; {correct} correct on MI300X ===")
    for k, v in summary["speedups"].items():
        print(f"   {k}: {v}x vs PyTorch reference")


if __name__ == "__main__":
    main()
