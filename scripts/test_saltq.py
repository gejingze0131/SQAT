#!/usr/bin/env python
"""
Standalone tests for src/qat_saltq.py — CPU-only, no bitsandbytes / real model / GPU needed.

The three gates this covers are the ones the method's correctness actually rests on:

  1. DEPLOY == TRAIN, EXACTLY.  SALTQLinear.deployed_weight() must equal effective_weight() with
     max|Δ| == 0 (not an epsilon). That is the machine-checkable form of "merge-free": the salient
     segment's export rounding is literally the function the training STE applied, and the
     non-salient segment's codes are untouched while (s, z) are carried over verbatim.

  2. STEP-0 EQUIVALENCE.  With the asymmetric grid, a freshly built SALT-Q layer must reproduce
     the canonical GPTQ reconstruction bit-for-bit, because LSQ+ at `current_minmax` init IS the
     canonical affine grid. If this fails, training starts off a different grid than the
     initialization it inherited and the GPTQ init is silently wasted.

  3. FREEDOM ALLOCATION.  Gradients must reach the salient weights and both (s, z); the frozen
     codes must never receive one, and must be bit-identical before and after an optimizer step.

Run:  python scripts/test_saltq.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.gptq import gptq_quantize_layer
from src.qat_base import init_lsq_scale_sym, init_lsq_scale_zp_asym
from src.qat_saltq import SALTQLinear
from src.quant_primitives import group_dequantize

PASSED = 0
FAILED = 0


def check(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {msg}")
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}")


def build_layer(out_f, in_f, gk, gs, bits, symmetric, seed=0, hess=None):
    """Build a SALTQLinear the way build_saltq_base does: GPTQ codes + fp32 salient init."""
    torch.manual_seed(seed)
    W = torch.randn(out_f, in_f) * 0.02
    H = hess if hess is not None else (lambda X: X.t() @ X)(torch.randn(512, in_f))
    W_int, scale, zp = gptq_quantize_layer(
        W.clone(), H, gk, gs, bits, symmetric, percdamp=0.01, blocksize=gs * 2,
    )
    n_sal_g = gk // gs
    w_s = s_s = z_s = None
    if gk > 0:
        w_s = W[:, :gk].float().clone()
        if symmetric:
            s_s = init_lsq_scale_sym(w_s, gs, bits)
        else:
            s_s, z_s = init_lsq_scale_zp_asym(w_s, gs, bits)
    layer = SALTQLinear(
        in_features=in_f, out_features=out_f, group_k=gk, group_size=gs,
        q_bits=bits, symmetric=symmetric,
        codes=W_int[:, gk:], s_n=scale[:, n_sal_g:], z_n=zp[:, n_sal_g:],
        w_s=w_s, s_s=s_s, z_s=z_s,
    )
    return layer, W, (W_int, scale, zp)


CASES = [
    # (out, in, group_k, group_size, bits, symmetric, label)
    (64, 256, 64, 32, 2, False, "asym INT2 gs32 gk64"),
    (64, 256, 64, 32, 3, False, "asym INT3 gs32 gk64"),
    (64, 256, 128, 64, 4, False, "asym INT4 gs64 gk128"),
    (48, 192, 64, 64, 3, True, "sym  INT3 gs64 gk64"),
    (64, 256, 0, 32, 2, False, "asym INT2 gs32 gk0 (o_proj-like)"),
    (32, 256, 256 - 32, 32, 3, False, "asym INT3 gs32 mostly-salient"),
]


def test_deploy_equals_train():
    print("\ntest_deploy_equals_train  (gate: max|Δ| == 0, merge-free)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym)
        # perturb away from init so the test is not trivially satisfied by the initialization
        with torch.no_grad():
            if gk > 0:
                layer.weight_salient.add_(torch.randn_like(layer.weight_salient) * 0.01)
                layer.lsq_w_scale.mul_(1.13)
                if not sym:
                    layer.lsq_w_zp.add_(0.37)
            layer.saltq_s.mul_(0.91)
            if not sym:
                layer.saltq_z.add_(-0.44)
        err = (layer.deployed_weight() - layer.effective_weight()).abs().max().item()
        check(err == 0.0, f"{label:34s} deployed == effective, max|Δ|={err:.3e}")


def test_deployed_tensors_roundtrip():
    print("\ntest_deployed_tensors_roundtrip  (canonical group_dequantize reproduces the weight)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=1)
        q, s, z = layer.deployed_tensors()
        # group_dequantize is the shared export-side reconstruction used by every other method
        # in this repo; the affine branch is (q - z) * s, which is what a GPTQ kernel computes.
        W_deq = group_dequantize(q, s, z, gs, in_f, symmetric=False)
        err = (W_deq - layer.effective_weight()).abs().max().item()
        check(err == 0.0, f"{label:34s} group_dequantize round-trip, max|Δ|={err:.3e}")


def test_step0_matches_gptq_reconstruction():
    print("\ntest_step0_matches_gptq_reconstruction  (asym LSQ+ init IS the canonical grid)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        if sym:
            # Symmetric LSQ uses Qn = -2^(b-1) (one extra negative level) and is deliberately NOT
            # interchangeable with the canonical symmetric [-q_max, q_max] grid.
            continue
        layer, W, (W_int, scale, zp) = build_layer(out_f, in_f, gk, gs, bits, sym, seed=2)
        gptq_recon = group_dequantize(W_int, scale, zp, gs, in_f, symmetric=False)
        err = (layer.effective_weight() - gptq_recon).abs().max().item()
        check(err == 0.0,
              f"{label:34s} step-0 == GPTQ reconstruction, max|Δ|={err:.3e}")


def test_freedom_allocation():
    print("\ntest_freedom_allocation  (who gets gradients)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=3)
        codes_before = layer.codes.clone()
        x = torch.randn(8, in_f)
        layer(x).square().mean().backward()

        ok = True
        if gk > 0:
            ok &= layer.weight_salient.grad is not None and layer.weight_salient.grad.abs().sum() > 0
            ok &= layer.lsq_w_scale.grad is not None and layer.lsq_w_scale.grad.abs().sum() > 0
            if not sym:
                ok &= layer.lsq_w_zp.grad is not None and layer.lsq_w_zp.grad.abs().sum() > 0
        ok &= layer.saltq_s.grad is not None and layer.saltq_s.grad.abs().sum() > 0
        if not sym:
            ok &= layer.saltq_z.grad is not None and layer.saltq_z.grad.abs().sum() > 0
        check(bool(ok), f"{label:34s} salient weights + (s, z) all receive gradients")

        opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
        opt.step()
        check(torch.equal(layer.codes, codes_before),
              f"{label:34s} frozen codes unchanged after an optimizer step")
        check(not layer.codes.requires_grad and not layer.codes.is_floating_point(),
              f"{label:34s} codes are a non-grad int buffer")


def test_codes_not_in_state_dict():
    print("\ntest_codes_not_in_state_dict  (checkpoints must not carry the frozen codes)")
    layer, _, _ = build_layer(64, 256, 64, 32, 2, False, seed=4)
    keys = set(layer.state_dict().keys())
    check("codes" not in keys, f"codes excluded from state_dict (keys={sorted(keys)})")
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    check(trainable == {"saltq_s", "saltq_z", "weight_salient", "lsq_w_scale", "lsq_w_zp"},
          f"trainable set == {sorted(trainable)}")


def test_forward_matches_linear():
    print("\ntest_forward_matches_linear")
    for out_f, in_f, gk, gs, bits, sym, label in CASES[:3]:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=5)
        x = torch.randn(4, 7, in_f)
        y = layer(x)
        y_ref = x @ layer.effective_weight().t()
        err = (y - y_ref).abs().max().item()
        check(err < 1e-5, f"{label:34s} forward == x @ W_eff^T, max|Δ|={err:.3e}")


def main():
    print("=" * 68)
    print("  SALT-Q — Saliency-Allocated Low-bit Trainability")
    print("=" * 68)
    test_deploy_equals_train()
    test_deployed_tensors_roundtrip()
    test_step0_matches_gptq_reconstruction()
    test_freedom_allocation()
    test_codes_not_in_state_dict()
    test_forward_matches_linear()
    print("\n" + "=" * 68)
    print(f"  SALT-Q: {PASSED} passed, {FAILED} failed")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
