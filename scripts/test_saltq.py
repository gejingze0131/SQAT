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


def build_layer(out_f, in_f, gk, gs, bits, symmetric, seed=0, hess=None, train_scale=False,
                continuous_z=True):
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
        train_scale=train_scale, continuous_z=continuous_z, wq_dtype=torch.float32,
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
        codes_before = layer._codes_cpu.clone()
        x = torch.randn(8, in_f)
        layer(x).square().mean().backward()

        ok = True
        if gk > 0:
            ok &= layer.weight_salient.grad is not None and layer.weight_salient.grad.abs().sum() > 0
            ok &= layer.lsq_w_scale.grad is not None and layer.lsq_w_scale.grad.abs().sum() > 0
            if not sym:
                ok &= layer.lsq_w_zp.grad is not None and layer.lsq_w_zp.grad.abs().sum() > 0
        if not sym:
            ok &= layer.saltq_z.grad is not None and layer.saltq_z.grad.abs().sum() > 0
        check(bool(ok), f"{label:34s} salient weights + z receive gradients")

        opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
        opt.step()
        check(torch.equal(layer._codes_cpu, codes_before),
              f"{label:34s} frozen codes unchanged after an optimizer step")
        check(not layer._codes_cpu.requires_grad and not layer._codes_cpu.is_floating_point(),
              f"{label:34s} codes are a non-grad int tensor")


def test_codes_not_in_state_dict():
    print("\ntest_codes_not_in_state_dict  (checkpoints must not carry the frozen codes)")
    layer, _, _ = build_layer(64, 256, 64, 32, 2, False, seed=4)
    keys = set(layer.state_dict().keys())
    check("codes" not in keys and "wq" not in keys,
          f"codes and the precomputed q*s stay out of state_dict (keys={sorted(keys)})")
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    check(trainable == {"saltq_z", "weight_salient", "lsq_w_scale", "lsq_w_zp"},
          f"trainable set == {sorted(trainable)}")


def test_forward_matches_linear():
    """
    The z-only forward never builds W_eff: it is x @ (q*s)^T - pool_g(x) @ (z*s)^T. That is an
    algebraic rearrangement of x @ W_eff^T, so it must agree with the reference to fp round-off.
    This is the test that would catch a wrong sign, a mis-sliced pooled input, or a group
    misalignment between the pooled x and the zero-point rows.
    """
    print("\ntest_forward_matches_linear  (fast path == x @ W_eff^T)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=5)
        with torch.no_grad():   # move z off its init so the correction term is non-trivial
            if not sym:
                layer.saltq_z.add_(torch.randn_like(layer.saltq_z) * 0.5)
        x = torch.randn(4, 7, in_f)
        y = layer(x)
        y_ref = x @ layer.effective_weight().t()
        scale = y_ref.abs().max().item()
        err = (y - y_ref).abs().max().item()
        check(err / max(scale, 1e-9) < 1e-5,
              f"{label:34s} rel err {err / max(scale, 1e-9):.2e}")


def test_z_only_freedom():
    """
    Default (train_scale=False): z trains, s does not. This is the whole cost argument — dL/ds
    would force a full [out, in] weight-gradient GEMM (2N) while dL/dz collapses into a pooled
    input (2N/g), so s silently becoming trainable would quietly cost ~33% more FLOPs per step.
    """
    print("\ntest_z_only_freedom  (s must be frozen by default)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        if sym:
            continue
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=7)
        names = {n for n, _ in layer.named_parameters()}
        check("saltq_z" in names and "saltq_s" not in names,
              f"{label:34s} trainable: z yes, s no ({sorted(names)})")
        check(hasattr(layer, "wq") and not layer.wq.requires_grad,
              f"{label:34s} q*s precomputed as a frozen buffer")
        check(layer._codes_cpu.device.type == "cpu",
              f"{label:34s} codes kept on the host")

        x = torch.randn(8, in_f)
        layer(x).square().mean().backward()
        check(layer.saltq_z.grad is not None and layer.saltq_z.grad.abs().sum() > 0,
              f"{label:34s} z receives a gradient")


def test_train_scale_ablation():
    """train_scale=True must restore a trainable s and still satisfy the deploy-equality gate."""
    print("\ntest_train_scale_ablation")
    for out_f, in_f, gk, gs, bits, sym, label in CASES[:3]:
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=8, train_scale=True)
        names = {n for n, _ in layer.named_parameters()}
        check("saltq_s" in names, f"{label:34s} s is trainable in the ablation path")
        x = torch.randn(8, in_f)
        layer(x).square().mean().backward()
        check(layer.saltq_s.grad is not None and layer.saltq_s.grad.abs().sum() > 0,
              f"{label:34s} s receives a gradient")
        err = (layer.deployed_weight() - layer.effective_weight()).abs().max().item()
        check(err == 0.0, f"{label:34s} deployed == effective still exact, max|Δ|={err:.3e}")


def test_zero_point_can_actually_move():
    """
    The failure that produced 32.9 instead of 42.6 on the first real run.

    z is measured in quantization LEVELS, where the meaningful step is 1.0. Rounding it puts a
    dead zone in front of the parameter: gradients flow, but the effective weight does not change
    until z crosses half a level. Combined with a scale-sized lr, round(z) changed for 0 of 1.63M
    parameters over a full training run and the entire non-salient segment was inert.

    Continuous z has no dead zone: ANY update to z changes the weight. This test asserts that
    directly, so the regression cannot come back silently.
    """
    print("\ntest_zero_point_can_actually_move  (the dead-zone regression)")
    for out_f, in_f, gk, gs, bits, sym, label in CASES:
        if sym:
            continue
        # continuous: a sub-half-level nudge MUST change the weight
        layer, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=11, continuous_z=True)
        before = layer.effective_weight().clone()
        with torch.no_grad():
            layer.saltq_z.add_(0.01)
        moved = (layer.effective_weight() - before).abs().max().item()
        check(moved > 0, f"{label:34s} continuous z: a 0.01-level nudge moves W (|Δ|={moved:.2e})")

        # integer: the same nudge must be swallowed, which is exactly why it is not the default
        layer_i, _, _ = build_layer(out_f, in_f, gk, gs, bits, sym, seed=11, continuous_z=False)
        before_i = layer_i.effective_weight().clone()
        with torch.no_grad():
            layer_i.saltq_z.add_(0.01)
        moved_i = (layer_i.effective_weight() - before_i).abs().max().item()
        check(moved_i == 0.0,
              f"{label:34s} integer z: same nudge is swallowed by the dead zone (as expected)")

        # and the deploy gate must hold in BOTH parameterizations
        for lay, tag in ((layer, "continuous"), (layer_i, "integer")):
            err = (lay.deployed_weight() - lay.effective_weight()).abs().max().item()
            check(err == 0.0, f"{label:34s} {tag:10s} z: deployed == effective, max|Δ|={err:.3e}")


def test_optimizer_group_units():
    """Scales and zero-points must land in DIFFERENT optimizer groups — they have different units."""
    print("\ntest_optimizer_group_units")
    from src.qat_saltq import SCALE_PARAM_FRAGMENTS, ZEROPOINT_PARAM_FRAGMENTS
    check(not (set(SCALE_PARAM_FRAGMENTS) & set(ZEROPOINT_PARAM_FRAGMENTS)),
          "scale and zero-point name fragments are disjoint")
    layer, _, _ = build_layer(64, 256, 64, 32, 3, False, seed=12)
    names = [n for n, p in layer.named_parameters() if p.requires_grad]
    scales = [n for n in names if any(f in n for f in SCALE_PARAM_FRAGMENTS)]
    zps = [n for n in names if any(f in n for f in ZEROPOINT_PARAM_FRAGMENTS)]
    check(zps and not (set(scales) & set(zps)),
          f"zero-points {zps} classified apart from scales {scales}")


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
    test_z_only_freedom()
    test_train_scale_ablation()
    test_zero_point_can_actually_move()
    test_optimizer_group_units()
    print("\n" + "=" * 68)
    print(f"  SALT-Q: {PASSED} passed, {FAILED} failed")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
