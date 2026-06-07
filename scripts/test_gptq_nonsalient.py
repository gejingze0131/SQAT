#!/usr/bin/env python
"""
Numerical sanity tests for the GPTQ non-salient export path (Improvement 1).

Checks, on synthetic weights + calibration activations (CPU, no model needed):
  1. group_quantize layout: gptq_quantize_layer returns (W_int, scale, zp) that group_dequantize
     reconstructs.
  2. Salient slice [0:group_k] is BIT-IDENTICAL to the canonical training grid (group_fakequant) —
     GPTQ must not disturb the QAT-protected columns.
  3. The OBS objective trace(ΔW H ΔWᵀ) (output error on the calibration set) is LOWER for GPTQ than
     for plain RTN — i.e. the non-salient columns are genuinely better, and they also absorb the
     salient slice's quant error.
  4. o_proj case (group_k=0, fully GPTQ) — still beats RTN, no fixed slice.
  5. Works for both asymmetric and symmetric grids and at INT3/INT4.

Run:  python scripts/test_gptq_nonsalient.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.qat_permute_sqat import (
    gptq_quantize_layer,
    group_quantize,
    group_dequantize,
    group_fakequant,
)


def _obs_objective(W_deq, W, H):
    """trace(ΔW H ΔWᵀ) = output error on the calibration set whose Hessian is H."""
    dW = (W_deq - W).float()
    return torch.einsum("oi,ij,oj->", dW, H, dW).item()


def _make_problem(out_f=64, in_f=256, n_tokens=2048, n_outliers=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g)                      # ~Gaussian weights
    X = torch.randn(n_tokens, in_f, generator=g)
    # activation outliers: a handful of channels with large variance (these become "salient")
    scale = torch.ones(in_f)
    idx = torch.randperm(in_f, generator=g)[:n_outliers]
    scale[idx] = torch.linspace(8.0, 20.0, n_outliers)
    X = X * scale
    H = X.t() @ X
    return W, H, idx


def _check_case(symmetric, q_bits, group_size, group_k, tag):
    W, H, _ = _make_problem()
    out_f, in_f = W.shape

    # ---- RTN baseline (current export path) ----
    wi_r, sc_r, zp_r = group_quantize(W, group_size, q_bits, symmetric)
    W_rtn = group_dequantize(wi_r, sc_r, zp_r, group_size, in_f, symmetric)

    # ---- GPTQ ----
    wi_g, sc_g, zp_g = gptq_quantize_layer(W, H, group_k, group_size, q_bits, symmetric)
    assert wi_g.shape == (out_f, in_f)
    assert sc_g.shape == (out_f, in_f // group_size)
    W_gptq = group_dequantize(wi_g, sc_g, zp_g, group_size, in_f, symmetric)

    # (2) salient slice identical to the canonical training grid
    if group_k > 0:
        canon = group_fakequant(W[:, :group_k].float(), group_size, q_bits, symmetric)
        sal_err = (W_gptq[:, :group_k] - canon).abs().max().item()
        assert sal_err < 1e-5, f"[{tag}] salient slice disturbed by GPTQ: max|Δ|={sal_err:.2e}"
    else:
        sal_err = 0.0

    # (3) OBS objective lower than RTN
    obj_rtn = _obs_objective(W_rtn, W, H)
    obj_gptq = _obs_objective(W_gptq, W, H)
    ratio = obj_gptq / max(obj_rtn, 1e-12)
    assert obj_gptq < obj_rtn, (
        f"[{tag}] GPTQ obj {obj_gptq:.4e} not < RTN obj {obj_rtn:.4e}"
    )

    print(f"[OK] {tag:42s} salient max|Δ|={sal_err:.1e}  "
          f"OBS: RTN={obj_rtn:.3e} GPTQ={obj_gptq:.3e}  (GPTQ/RTN={ratio:.3f})")


def _check_awq_case(symmetric, q_bits, group_size, group_k, tag):
    """GPTQ + AWQ-scale: the salient slice (after /S bake-back) must equal the amplified canonical
    training grid quant(W_S*S)/S; o_proj-style gk=0 has no slice."""
    W, H, _ = _make_problem()
    out_f, in_f = W.shape
    s = (1.0 + torch.rand(group_k)).clamp(max=2.0)
    wi, sc, zp = gptq_quantize_layer(
        W, H, group_k, group_size, q_bits, symmetric, awq_s=s,
    )
    W_gptq = group_dequantize(wi, sc, zp, group_size, in_f, symmetric)
    W_gptq[:, :group_k] = W_gptq[:, :group_k] / s.view(1, -1)        # /S bake-back
    canon = group_fakequant(W[:, :group_k].float() * s.view(1, -1),
                            group_size, q_bits, symmetric) / s.view(1, -1)
    sal_err = (W_gptq[:, :group_k] - canon).abs().max().item()
    assert sal_err < 1e-5, f"[{tag}] AWQ salient slice mismatch: {sal_err:.2e}"
    print(f"[OK] {tag:42s} AWQ salient (post-/S) max|Δ|={sal_err:.1e}")


def main():
    torch.manual_seed(0)
    cases = [
        # (symmetric, q_bits, group_size, group_k, tag)
        (False, 4, 64, 128, "asym INT4 gs64 gk128 (q/k/v/gate/up/down)"),
        (False, 4, 64, 0,   "asym INT4 gs64 gk0   (o_proj: fully GPTQ)"),
        (True,  4, 64, 128, "sym  INT4 gs64 gk128"),
        (False, 3, 64, 128, "asym INT3 gs64 gk128"),
        (False, 4, 32, 128, "asym INT4 gs32 gk128"),
        (False, 4, 128, 128, "asym INT4 gs128 gk128 (1 salient group)"),
    ]
    for c in cases:
        _check_case(*c)
    # GPTQ composed with AWQ-scale: salient slice stays on the amplified canonical grid.
    for sym, qb, gs, gk, tag in cases:
        if gk > 0:
            _check_awq_case(sym, qb, gs, gk, tag)
    print("\nAll GPTQ non-salient sanity tests passed.")


if __name__ == "__main__":
    main()
