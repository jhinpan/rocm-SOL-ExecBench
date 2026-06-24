#!/usr/bin/env python3
"""Registry mapping SOL-ExecBench problems to PRODUCTION aiter ROCm kernels.

Each recognizer inspects a problem Definition (inputs/outputs/reference) and, if
it matches a known single-op pattern, returns the `kernel.py` source for a
solution whose run() calls the corresponding aiter kernel (ASM/CK/FlyDSL). The
PyTorch reference remains the correctness oracle -- the harness gates on it.

Fused/composite chains (most of L2) are handled separately (see L2-fusion work);
this registry covers cleanly-mappable single ops. Extend by adding recognizers.

Recognizers are deliberately conservative: when in doubt they return None. A
false negative only costs coverage, but a false positive would silently hand a
fused/composite problem a single-op kernel. The harness would catch the
mismatch as a correctness FAIL, but we still guard up front so the driver does
not waste a benchmarking run (and so the intent stays auditable).
"""
from __future__ import annotations

import re
from typing import Optional

# Tokens that indicate a problem is fused/composite (i.e. not a clean single op).
# Any of these in the reference disqualifies every single-op recognizer below.
_FUSED_TOKENS = (
    "softmax",
    "attention",
    "matmul",
    "@ ",
    "conv",
    "scaled_dot_product",
    "rope",
    "rotary",
    "dropout",
    "embedding",
)


def _fields(d: dict):
    """Extract (reference, input-names, output-names) defensively.

    inputs/outputs in a Definition are name->spec dicts; we only need the
    ordered argument names, which list(dict) yields (insertion order, py3.7+).
    """
    ref = d.get("reference") or ""
    ins = list(d.get("inputs") or {})
    outs = list(d.get("outputs") or {})
    return ref, ins, outs


def _has_fused_token(r: str) -> bool:
    return any(tok in r for tok in _FUSED_TOKENS)


def _eps(ref: str, default: float = 1e-5) -> float:
    # Only accept a numeric literal (optionally float/scientific). A symbolic
    # rhs (e.g. `eps = config.rms_eps`) leaves the default in place.
    m = re.search(r"(?:eps|EPS|epsilon)\s*=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", ref)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def recognize_rmsnorm(d: dict) -> Optional[dict]:
    ref, ins, outs = _fields(d)
    r = ref.lower()
    if len(ins) != 2 or len(outs) != 1:
        return None
    if _has_fused_token(r):
        return None
    # Core RMSNorm signature: rsqrt(mean(x**2) + eps). Require an explicit
    # squaring of the input (pow(2) / **2 / x*x); a bare `* x` is too loose.
    if "rsqrt" not in r or "mean" not in r:
        return None
    if not ("pow(2)" in r or "**2" in r or "x * x" in r or "x*x" in r):
        return None
    # Reject anything that is more than a plain RMSNorm: variance-based
    # layernorm, bias, residual add, or a different normalization.
    if any(tok in r for tok in ("var", "+ bias", "residual", "layernorm", "layer_norm", "group")):
        return None
    x, w = ins[0], ins[1]
    eps = _eps(ref)
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}, {w}):\n"
        "    # aiter CK rmsnorm (production ROCm kernel)\n"
        f"    return aiter.rms_norm({x}, {w}, {eps})\n"
    )
    return {"kernel": src, "aiter_fn": "rms_norm", "backend": "CK"}


def _is_glu_split(r: str, ref: str) -> bool:
    """True if the reference splits its single input into two halves."""
    return "chunk(2" in r or ".chunk(2" in ref or "split" in r or "[..., :" in ref or "[:, :" in ref


def recognize_geglu(d: dict) -> Optional[dict]:
    ref, ins, outs = _fields(d)
    r = ref.lower()
    if len(ins) != 1 or len(outs) != 1:
        return None
    if _has_fused_token(r):
        return None
    if "gelu" not in r:
        return None
    # Must be a gated unit: split into two halves and multiply.
    if not (_is_glu_split(r, ref) or "geglu" in r):
        return None
    if "*" not in ref:
        return None
    # A GEGLU is GELU-gated only; presence of silu/swish means it is not geglu.
    if "silu" in r or "swish" in r:
        return None
    x = ins[0]
    fn = "gelu_tanh_and_mul" if ("tanh" in r or "approximate='tanh'" in ref or 'approximate="tanh"' in ref) else "gelu_and_mul"
    # aiter activations are destination-passing: fn(out, input); out last dim = input/2
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}):\n"
        f"    out = torch.empty(*{x}.shape[:-1], {x}.shape[-1] // 2, dtype={x}.dtype, device={x}.device)\n"
        f"    aiter.{fn}(out, {x})\n"
        "    return out\n"
    )
    return {"kernel": src, "aiter_fn": fn, "backend": "HIP"}


def recognize_swiglu(d: dict) -> Optional[dict]:
    ref, ins, outs = _fields(d)
    r = ref.lower()
    if len(ins) != 1 or len(outs) != 1:
        return None
    if _has_fused_token(r):
        return None
    if "silu" not in r and "swish" not in r:
        return None
    if not (_is_glu_split(r, ref) or "swiglu" in r):
        return None
    if "*" not in ref:
        return None
    # A SwiGLU is SiLU-gated only; a gelu term means it is some other GLU.
    if "gelu" in r:
        return None
    x = ins[0]
    src = (
        "import torch\nimport aiter\n\n"
        "@torch.no_grad()\n"
        f"def run({x}):\n"
        f"    out = torch.empty(*{x}.shape[:-1], {x}.shape[-1] // 2, dtype={x}.dtype, device={x}.device)\n"
        f"    aiter.silu_and_mul(out, {x})\n"
        "    return out\n"
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
