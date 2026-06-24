#!/usr/bin/env python3
"""Registry mapping SOL-ExecBench problems to PRODUCTION aiter ROCm kernels.

Each recognizer inspects a problem Definition (inputs/outputs/reference) and, if
it matches a known single-op pattern, returns the `kernel.py` source for a
solution whose run() calls the corresponding aiter kernel (ASM/CK/FlyDSL). The
PyTorch reference remains the correctness oracle — the harness gates on it.

Fused/composite chains (most of L2) are handled separately (see L2-fusion work);
this registry covers cleanly-mappable single ops. Extend by adding recognizers.
"""
from __future__ import annotations
import re
from typing import Optional


def _eps(ref: str, default: float = 1e-5) -> float:
    m = re.search(r"(?:eps|EPS|epsilon)\s*=\s*([0-9.eE+-]+)", ref)
    try:
        return float(m.group(1)) if m else default
    except Exception:
        return default


def recognize_rmsnorm(d: dict) -> Optional[dict]:
    ref = d["reference"]; ins = list(d["inputs"]); outs = list(d["outputs"])
    r = ref.lower()
    if len(outs) != 1:
        return None
    if not ("rsqrt" in r and "mean" in r and ("pow(2)" in r or "**2" in r or "* x" in r or "x * x" in r)):
        return None
    if "var" in r or "+ bias" in r or "residual" in r or "softmax" in r:
        return None
    # need exactly an x-like and a weight-like input
    if len(ins) != 2:
        return None
    x, w = ins[0], ins[1]
    eps = _eps(ref)
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}, {w}):\n"
        f"    # aiter CK rmsnorm (production ROCm kernel)\n"
        f"    return aiter.rms_norm({x}, {w}, {eps})\n"
    )
    return {"kernel": src, "aiter_fn": "rms_norm", "backend": "CK"}


def recognize_geglu(d: dict) -> Optional[dict]:
    ref = d["reference"]; ins = list(d["inputs"]); outs = list(d["outputs"])
    r = ref.lower()
    if len(ins) != 1 or len(outs) != 1:
        return None
    if "gelu" not in r or not ("chunk" in r or "split" in r or "[..., :" in ref or "* x_linear" in r or "geglu" in r):
        return None
    x = ins[0]
    fn = "gelu_tanh_and_mul" if "tanh" in r or "approximate='tanh'" in ref else "gelu_and_mul"
    # aiter activations are destination-passing: fn(out, input); out last dim = input/2
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}):\n"
        f"    out = torch.empty(*{x}.shape[:-1], {x}.shape[-1] // 2, dtype={x}.dtype, device={x}.device)\n"
        f"    aiter.{fn}(out, {x})\n"
        f"    return out\n"
    )
    return {"kernel": src, "aiter_fn": fn, "backend": "HIP"}


def recognize_swiglu(d: dict) -> Optional[dict]:
    ref = d["reference"]; ins = list(d["inputs"]); outs = list(d["outputs"])
    r = ref.lower()
    if len(ins) != 1 or len(outs) != 1:
        return None
    if "silu" not in r and "swish" not in r:
        return None
    if not ("chunk" in r or "split" in r or "[..., :" in ref):
        return None
    x = ins[0]
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}):\n"
        f"    out = torch.empty(*{x}.shape[:-1], {x}.shape[-1] // 2, dtype={x}.dtype, device={x}.device)\n"
        f"    aiter.silu_and_mul(out, {x})\n"
        f"    return out\n"
    )
    return {"kernel": src, "aiter_fn": "silu_and_mul", "backend": "HIP"}


RECOGNIZERS = [recognize_rmsnorm, recognize_geglu, recognize_swiglu]


def map_problem(definition: dict) -> Optional[dict]:
    for rec in RECOGNIZERS:
        out = rec(definition)
        if out:
            out["recognizer"] = rec.__name__
            return out
    return None
