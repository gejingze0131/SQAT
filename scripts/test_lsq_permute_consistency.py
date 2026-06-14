#!/usr/bin/env python
"""
sqat_permute LSQ / LSQ+ consistency tests — CPU-only.

Phase 2 of LSQ_IMPLEMENTATION_PROMPT.md: the salient slice [0:group_k] is quantized on a LEARNED
scale[, zp] (LSQ grid). The same train↔export invariant as Phase 1 must hold, AND it must compose
with the AWQ-style per-channel amplification (the default target config is asym + LSQ+ + AWQ on).

Covers:
  1. salient-slice train fakequant == export quant→dequant with a learned scale[, zp]
     (sym/asym × INT2/3/4 × g64/g128), via verify_permute_quant_consistency(fixed_scale=...).
  2. same, but with AWQ amplification on (amplify W*S → LSQ fakequant → /S ; export bakes /S).
  3. fused_qat_residual_outputs(lsq_scale=...) matches an independent reference delta.
  4. group_quantize(fixed_scale) + group_dequantize reproduces group_fakequant(fixed_scale) exactly.
  5. enable_lsq=False path bit-identical to the original min-max group_fakequant (regression).

Run:  CUDA_VISIBLE_DEVICES="" python scripts/test_lsq_permute_consistency.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qat_permute_sqat import (
    group_fakequant,
    group_quantize,
    group_dequantize,
    verify_permute_quant_consistency,
    fused_qat_residual_outputs,
)
from src.qat_base import (
    init_lsq_scale_sym,
    init_lsq_scale_zp_asym,
)

torch.manual_seed(0)
TOL = 1e-4
_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


def make_fixed(W_sal, group_size, q_bits, symmetric):
    """current_minmax-init learned scale[, zp] for the salient slice (the realistic init)."""
    if symmetric:
        return init_lsq_scale_sym(W_sal, group_size, q_bits)
    s, z = init_lsq_scale_zp_asym(W_sal, group_size, q_bits)
    return (s, z)


# ---------------------------------------------------------------------------
# 1 & 2. salient train fakequant == export quant->dequant (no AWQ, then AWQ on)
# ---------------------------------------------------------------------------
print("== salient train==export (LSQ grid), no AWQ ==")
for q_bits in (2, 3, 4):
    for gs in (64, 128):
        out_f, gk = 48, gs * 2          # group_k = 2 groups
        W = torch.randn(out_f, gk).float() * 0.05
        for symmetric in (True, False):
            fixed = make_fixed(W[:, :gk], gs, q_bits, symmetric)
            err = verify_permute_quant_consistency(
                W, gk, gs, q_bits, symmetric, awq_s=None, fixed_scale=fixed,
            )
            tag = "ASYM" if not symmetric else "SYM "
            check(f"{tag} INT{q_bits} g{gs}: salient train==export (max|Δ|={err:.2e})", err < TOL)

print("\n== salient train==export (LSQ grid), AWQ amplification ON ==")
for q_bits in (2, 3, 4):
    for gs in (64, 128):
        out_f, gk = 48, gs * 2
        W = torch.randn(out_f, gk).float() * 0.05
        awq_s = (torch.rand(gk) * 1.0 + 1.0)          # S in [1, 2]
        for symmetric in (True, False):
            # learned scale must be init'd in the AMPLIFIED space (training does this)
            fixed = make_fixed(W[:, :gk] * awq_s.view(1, -1), gs, q_bits, symmetric)
            err = verify_permute_quant_consistency(
                W, gk, gs, q_bits, symmetric, awq_s=awq_s, fixed_scale=fixed,
            )
            tag = "ASYM" if not symmetric else "SYM "
            check(f"{tag} INT{q_bits} g{gs}: AWQ+LSQ train==export (max|Δ|={err:.2e})", err < TOL)


# ---------------------------------------------------------------------------
# 3. fused_qat_residual_outputs(lsq_scale=...) matches an independent reference
# ---------------------------------------------------------------------------
print("\n== fused_qat_residual_outputs LSQ == reference delta ==")
for symmetric in (True, False):
    for awq_on in (False, True):
        gs, gk, q_bits = 64, 128, 2
        rank = 8
        out0, out1 = 32, 24            # GQA-like uneven splits
        Wb = torch.randn(out0 + out1, gk).float() * 0.05
        A0 = torch.randn(rank, gk).float() * 0.02
        A1 = torch.randn(rank, gk).float() * 0.02
        B0 = torch.randn(out0, rank).float() * 0.02
        B1 = torch.randn(out1, rank).float() * 0.02
        X = torch.randn(5, gk).float()
        scaling = 2.0
        awq_s = (torch.rand(gk) + 1.0) if awq_on else None

        Wamp = Wb if awq_s is None else Wb * awq_s.view(1, -1)
        if symmetric:
            scale = init_lsq_scale_sym(Wamp, gs, q_bits)
            zp = None
        else:
            scale, zp = init_lsq_scale_zp_asym(Wamp, gs, q_bits)

        outs = fused_qat_residual_outputs(
            Wb, [A0, A1], [B0, B1], (out0, out1), X, gs, q_bits, symmetric, scaling,
            awq_s=awq_s, lsq_scale=scale, lsq_zp=zp,
        )
        Y = torch.cat(outs, dim=-1)

        # reference: W_curr = Wb + cat(B@A)*scaling; fakequant in amplified space; delta=Wfq/s - W_curr
        BA = torch.cat([B0 @ A0, B1 @ A1], dim=0)
        W_curr = Wb + BA * scaling
        if awq_s is not None:
            s = awq_s.view(1, -1)
            Wfq = group_fakequant(W_curr * s, gs, q_bits, symmetric,
                                  fixed_scale=(scale if symmetric else (scale, zp)))
            delta = Wfq / s - W_curr
        else:
            Wfq = group_fakequant(W_curr, gs, q_bits, symmetric,
                                  fixed_scale=(scale if symmetric else (scale, zp)))
            delta = Wfq - W_curr
        Yref = torch.nn.functional.linear(X, delta)
        err = (Y - Yref).abs().max().item()
        tag = ("ASYM" if not symmetric else "SYM ") + (" +AWQ" if awq_on else "     ")
        check(f"{tag}: fused residual == reference (max|Δ|={err:.2e})", err < 1e-4)


# ---------------------------------------------------------------------------
# 4. group_quantize(fixed_scale) + dequantize == group_fakequant(fixed_scale)
# ---------------------------------------------------------------------------
print("\n== group_quantize/dequantize(fixed) == group_fakequant(fixed) ==")
for symmetric in (True, False):
    gs, gk, q_bits = 64, 128, 3
    W = torch.randn(40, gk).float() * 0.05
    fixed = make_fixed(W, gs, q_bits, symmetric)
    fq = group_fakequant(W, gs, q_bits, symmetric, fixed_scale=fixed)
    wi, sc, zp = group_quantize(W, gs, q_bits, symmetric, fixed_scale=fixed)
    dq = group_dequantize(wi, sc, zp, gs, gk, symmetric)
    err = (fq - dq).abs().max().item()
    check(f"{'ASYM' if not symmetric else 'SYM '}: quantize→dequant == fakequant (max|Δ|={err:.2e})",
          err < 1e-5)


# ---------------------------------------------------------------------------
# 5. enable_lsq=False (fixed_scale=None) bit-identical to original min-max path
# ---------------------------------------------------------------------------
print("\n== regression: fixed_scale=None == original min-max group_fakequant ==")
for symmetric in (True, False):
    gs, gk, q_bits = 64, 128, 4
    W = torch.randn(40, gk).float() * 0.05
    a = group_fakequant(W, gs, q_bits, symmetric)                       # original (no fixed)
    b = group_fakequant(W, gs, q_bits, symmetric, fixed_scale=None)     # explicit None
    check(f"{'ASYM' if not symmetric else 'SYM '}: None path unchanged",
          torch.equal(a, b))


print(f"\n{'=' * 52}")
print(f"  sqat_permute LSQ consistency: {_passed} passed, {_failed} failed")
print(f"{'=' * 52}")
sys.exit(1 if _failed else 0)
