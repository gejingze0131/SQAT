#!/usr/bin/env python
"""
LSQ / LSQ+ numerical consistency tests — CPU-only, no bitsandbytes / real model needed.

The single most important invariant (see LSQ_IMPLEMENTATION_PROMPT.md §1, §3.4): the training
fakequant grid MUST be bit-identical to the export quantize->dequant grid when both use the SAME
learned scale[, zp]. Historically a train(min/max) vs export(pos/neg) mismatch silently killed QAT.

For a random W and a random POSITIVE scale (+ random zp for asym), this asserts:

    groupwise_lsq_*_fakequant(W, s[, z]) == dequant(export_quant(W, s[, z]))

with max|Δ| < 1e-5, across sym/asym × INT3/INT4 × group_size {64, 128}.

It also checks:
  - init_lsq_* produces a grid equal to the original min-max fakequant at step 0 (no jump).
  - grad_scale forward is identity (only scales the backward gradient).

Run:  CUDA_VISIBLE_DEVICES="" python scripts/test_lsq_consistency.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qat_base import (
    grad_scale,
    groupwise_lsq_symmetric_fakequant,
    groupwise_lsq_asym_fakequant,
    init_lsq_scale_sym,
    init_lsq_scale_zp_asym,
    lsq_quantize_export_sym,
    lsq_quantize_export_asym,
    groupwise_symmetric_fakequant,
)
from src.export import dequantize_symmetric, dequantize_asymmetric

torch.manual_seed(0)
TOL = 1e-5

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. grad_scale forward identity
# ---------------------------------------------------------------------------
print("== grad_scale ==")
x = torch.randn(10, requires_grad=True)
y = grad_scale(x, 0.123)
check("grad_scale forward == identity", (y - x).abs().max().item() < 1e-6,
      f"max|Δ|={(y - x).abs().max().item():.2e}")
y.sum().backward()
check("grad_scale backward scales grad by g",
      torch.allclose(x.grad, torch.full_like(x.grad, 0.123)),
      f"grad={x.grad[:3].tolist()}")


# ---------------------------------------------------------------------------
# 2. train fakequant == export quant->dequant (THE invariant)
# ---------------------------------------------------------------------------
print("\n== train fakequant == export quant->dequant ==")
for q_bits in (3, 4):
    for gs in (64, 128):
        out_f, in_f = 32, 320          # in_f not a multiple of gs (320 % 128 = 64) → exercises pad
        W = torch.randn(out_f, in_f).float()

        # --- symmetric LSQ (learn scale only) ---
        ng = (in_f + gs - 1) // gs
        scale = torch.rand(out_f, ng).float() * 0.05 + 1e-3      # random positive
        fq = groupwise_lsq_symmetric_fakequant(W, scale, gs, q_bits)
        W_int = lsq_quantize_export_sym(W, scale, gs, q_bits)
        deq = dequantize_symmetric(W_int, scale, gs, in_f)
        err = (fq - deq).abs().max().item()
        check(f"SYM  INT{q_bits} g{gs}: train==export (max|Δ|={err:.2e})", err < TOL)

        # --- asymmetric LSQ+ (learn scale + zp) ---
        scale_a = torch.rand(out_f, ng).float() * 0.05 + 1e-3
        Qp = 2 ** q_bits - 1
        zp = torch.randint(0, Qp + 1, (out_f, ng)).float()
        fq_a = groupwise_lsq_asym_fakequant(W, scale_a, zp, gs, q_bits)
        W_int_a, z_int = lsq_quantize_export_asym(W, scale_a, zp, gs, q_bits)
        # export dequant uses (q - z_int) * scale — same as the fakequant dequant.
        deq_a = dequantize_asymmetric(W_int_a, scale_a, z_int, gs, in_f)
        err_a = (fq_a - deq_a).abs().max().item()
        check(f"ASYM INT{q_bits} g{gs}: train==export (max|Δ|={err_a:.2e})", err_a < TOL)


# ---------------------------------------------------------------------------
# 3. init_lsq_* current_minmax matches the original min-max grid at step 0
# ---------------------------------------------------------------------------
print("\n== current_minmax init matches original min-max fakequant ==")
for q_bits in (3, 4):
    for gs in (64, 128):
        out_f, in_f = 16, 256
        W = torch.randn(out_f, in_f).float()

        # SYM: LSQ uses Qn=-2^(b-1), Qp=2^(b-1)-1 — a STRICT superset of the old [-q_max, q_max]
        # grid (q_max=2^(b-1)-1). The scale init is identical (amax/Qp), so quantized values match
        # EXCEPT where the old grid would clamp a value to -q_max that LSQ can place at -q_max-1.
        # With current_minmax the per-group amax row anchors |w|max exactly at +Qp, so the minimum
        # in-group value maps to >= -Qp on BOTH grids → identical. Assert equality.
        s0 = init_lsq_scale_sym(W, gs, q_bits)
        lsq_fq = groupwise_lsq_symmetric_fakequant(W, s0, gs, q_bits)
        mm_fq = groupwise_symmetric_fakequant(W, gs, float(2 ** (q_bits - 1) - 1))
        err = (lsq_fq - mm_fq).abs().max().item()
        check(f"SYM  INT{q_bits} g{gs}: LSQ init == min-max (max|Δ|={err:.2e})", err < 1e-5)

        # ASYM: LSQ+ init uses the affine min/max formula directly. This is NOT bit-identical to the
        # repo's pos/neg `asymmetric_scale_zero_from_pos_neg` (different zp anchoring), so we only
        # sanity-check that the init reconstructs W well (round-trip error bounded by 1 LSB).
        s0a, z0a = init_lsq_scale_zp_asym(W, gs, q_bits)
        rec = groupwise_lsq_asym_fakequant(W, s0a, z0a, gs, q_bits)
        # max per-group LSB
        lsb = s0a.max().item()
        rt = (rec - W).abs().max().item()
        check(f"ASYM INT{q_bits} g{gs}: LSQ+ init round-trip <= ~1 LSB "
              f"(max|Δ|={rt:.3e}, lsb={lsb:.3e})", rt <= lsb * 1.5 + 1e-6)


# ---------------------------------------------------------------------------
# 4. STE backward reaches W and scale (gradients non-trivial)
# ---------------------------------------------------------------------------
print("\n== STE backward flows to W and scale[, zp] ==")
W = torch.randn(8, 128, requires_grad=True)
scale = (torch.rand(8, 2) * 0.05 + 1e-3).requires_grad_(True)
out = groupwise_lsq_symmetric_fakequant(W, scale, 64, 4).sum()
out.backward()
check("SYM: W.grad nonzero", W.grad is not None and W.grad.abs().sum() > 0)
check("SYM: scale.grad nonzero", scale.grad is not None and scale.grad.abs().sum() > 0)

W2 = torch.randn(8, 128, requires_grad=True)
scale2 = (torch.rand(8, 2) * 0.05 + 1e-3).requires_grad_(True)
zp2 = torch.randint(0, 16, (8, 2)).float().requires_grad_(True)
out2 = groupwise_lsq_asym_fakequant(W2, scale2, zp2, 64, 4).sum()
out2.backward()
check("ASYM: W.grad nonzero", W2.grad is not None and W2.grad.abs().sum() > 0)
check("ASYM: scale.grad nonzero", scale2.grad is not None and scale2.grad.abs().sum() > 0)
check("ASYM: zp.grad nonzero", zp2.grad is not None and zp2.grad.abs().sum() > 0)


# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"  LSQ consistency: {_passed} passed, {_failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if _failed else 0)
