#!/usr/bin/env python
"""
Standalone tests for src/qat_permute_sqat.py — CPU-only, no bitsandbytes / real model needed.

Covers the spec's test requirements:
  1. group fakequant: shape invariance, group_k % group_size assert, per-row/per-group
     qparams, STE backward.
  2. QKV fused residual == three separate delta GEMMs (MHA) and shapes correct.
  2b. GQA (q_out != kv_out) cat fallback == separate; shapes correct.
  3. Gate/Up fused residual == two separate delta GEMMs; shapes correct.
  4. Original q/k/v projection forward still runs (BnB/LoRA path not replaced); injector
     only ADDS the fused residual.
  5. No full merged-weight materialization (injector stores only the [out, group_k] slice).

Run:  python scripts/test_qat_permute_sqat.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qat_permute_sqat import (
    group_fakequant,
    groupwise_symmetric_fakequant,
    groupwise_asymmetric_fakequant,
    fused_qat_residual_outputs,
    FusedAttnQATInjector,
    FusedMLPQATInjector,
    DownProjQATInjector,
    _build_hadamard,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fake PEFT LoRA linear (mimics base_layer + lora_A/lora_B/lora_dropout)
# ---------------------------------------------------------------------------

class _FakeBase(nn.Module):
    def __init__(self, out_f, in_f):
        super().__init__()
        self.out_features = out_f
        self.in_features = in_f
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.05, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class FakeLoRALinear(nn.Module):
    """dequantize_layer() will hit the `weight.data.shape==(out,in)` fallback for _FakeBase."""

    def __init__(self, out_f, in_f, rank, scaling, dropout=0.0):
        super().__init__()
        self.base_layer = _FakeBase(out_f, in_f)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_f, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_f, bias=False)})
        self.lora_dropout = nn.ModuleDict(
            {"default": nn.Dropout(dropout) if dropout > 0 else nn.Identity()}
        )
        nn.init.normal_(self.lora_A["default"].weight, std=0.02)
        nn.init.normal_(self.lora_B["default"].weight, std=0.02)  # non-zero so deltas are real
        self.scaling = scaling

    def forward(self, x):
        base = self.base_layer(x)
        lora = F.linear(F.linear(self.lora_dropout["default"](x),
                                 self.lora_A["default"].weight),
                        self.lora_B["default"].weight) * self.scaling
        return base + lora


class FakeAttn(nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = q, k, v

    def forward(self, hidden_states):
        return (self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states))


# ---------------------------------------------------------------------------
# Reference single-projection delta (the thing fusion must reproduce)
# ---------------------------------------------------------------------------

def ref_delta_out(W_base_S, A_S, B, X_S, gs, qb, sym, scaling):
    Wc = W_base_S.float() + (B.float() @ A_S.float()) * scaling
    Wfq = group_fakequant(Wc, gs, qb, sym)
    return F.linear(X_S, (Wfq - Wc).to(X_S.dtype))


def _ok(name):
    print(f"  [PASS] {name}")


# ---------------------------------------------------------------------------
# Test 1: fakequant
# ---------------------------------------------------------------------------

def test_fakequant():
    print("test_fakequant")
    out_f, gk, gs = 8, 128, 64
    W = torch.randn(out_f, gk)

    for sym, qb in [(True, 4), (False, 4)]:
        Wq = group_fakequant(W, gs, qb, sym)
        assert Wq.shape == W.shape, "shape must be preserved"
    _ok("shape invariance (sym+asym)")

    # group_k % group_size assert
    raised = False
    try:
        group_fakequant(torch.randn(4, 100), 64, 4, True)
    except AssertionError:
        raised = True
    assert raised, "non-divisible group_k must assert"
    _ok("group_k % group_size assert")

    # per-row, per-group qparams: scale up one group of one row only -> only that group changes
    W2 = torch.randn(out_f, gk)
    Wq_a = groupwise_symmetric_fakequant(W2, gs, 7.0)
    W3 = W2.clone()
    W3[0, :gs] *= 50.0                      # blow up row 0, group 0 only
    Wq_b = groupwise_symmetric_fakequant(W3, gs, 7.0)
    # untouched rows/groups must be bit-identical (independent qparams)
    assert torch.allclose(Wq_a[1:], Wq_b[1:]), "other rows must be unaffected"
    assert torch.allclose(Wq_a[0, gs:], Wq_b[0, gs:]), "other group of same row must be unaffected"
    assert not torch.allclose(Wq_a[0, :gs], Wq_b[0, :gs]), "the scaled group must change"
    _ok("per-row per-group independence")

    # error bound: |Wq - W| <= scale (per group)
    Wg = W.reshape(out_f, gk // gs, gs)
    scale = (Wg.abs().amax(2, keepdim=True) / 7.0).clamp(min=1e-8)
    err = (groupwise_symmetric_fakequant(W, gs, 7.0).reshape(out_f, gk // gs, gs) - Wg).abs()
    assert (err <= scale + 1e-6).all(), "symmetric rounding error must be within one scale"
    _ok("symmetric error within one scale")

    # STE backward: gradient flows and is ~identity in the interior
    Wg2 = torch.randn(out_f, gk, requires_grad=True)
    groupwise_asymmetric_fakequant(Wg2, gs, 15.0).sum().backward()
    assert Wg2.grad is not None and torch.isfinite(Wg2.grad).all()
    assert abs(Wg2.grad.mean().item() - 1.0) < 0.2, "STE grad should be ~1 on average"
    _ok("STE backward (asymmetric)")


# ---------------------------------------------------------------------------
# Test 2/2b/3: fused residual == separate
# ---------------------------------------------------------------------------

def _fused_vs_separate(out_list, gk, gs, rank, scaling, sym, qb, tag):
    bases = [torch.randn(o, gk) * 0.05 for o in out_list]
    A_S   = [torch.randn(rank, gk) * 0.02 for _ in out_list]
    B     = [torch.randn(o, rank) * 0.02 for o in out_list]
    X_S   = torch.randn(3, 5, gk)

    fused = fused_qat_residual_outputs(
        torch.cat(bases, 0), A_S, B, tuple(out_list), X_S, gs, qb, sym, scaling
    )
    assert len(fused) == len(out_list)
    for i, o in enumerate(out_list):
        assert fused[i].shape == (3, 5, o), f"{tag}: split shape wrong at {i}"
        ref = ref_delta_out(bases[i], A_S[i], B[i], X_S, gs, qb, sym, scaling)
        assert torch.allclose(fused[i], ref, atol=1e-5, rtol=1e-4), f"{tag}: fused != separate at {i}"
    _ok(f"{tag}: fused == separate + split shapes")


def test_fused_residual():
    print("test_fused_residual")
    # Test 2: QKV MHA (uniform out -> BMM path)
    _fused_vs_separate([16, 16, 16], gk=128, gs=64, rank=4, scaling=2.0, sym=False, qb=4, tag="QKV-MHA-asym")
    _fused_vs_separate([16, 16, 16], gk=128, gs=64, rank=4, scaling=2.0, sym=True,  qb=4, tag="QKV-MHA-sym")
    # Test 2b: GQA (q_out != kv_out -> cat fallback)
    _fused_vs_separate([16, 8, 8], gk=128, gs=32, rank=4, scaling=1.5, sym=False, qb=4, tag="QKV-GQA-asym")
    # Test 3: Gate/Up
    _fused_vs_separate([20, 20], gk=256, gs=128, rank=8, scaling=0.5, sym=False, qb=4, tag="GateUp-asym")


# ---------------------------------------------------------------------------
# Test 4/5: injector plumbing — original forward preserved, no full-merge
# ---------------------------------------------------------------------------

def test_attn_injector():
    print("test_attn_injector")
    out_f, in_f, gk, gs, rank, scaling = 16, 192, 128, 64, 4, 2.0
    q = FakeLoRALinear(out_f, in_f, rank, scaling)
    k = FakeLoRALinear(out_f, in_f, rank, scaling)
    v = FakeLoRALinear(out_f, in_f, rank, scaling)
    attn = FakeAttn(q, k, v)

    inj = FusedAttnQATInjector(
        attn, q, k, v, group_k=gk, group_size=gs, q_bits=4, symmetric=False, lora_scaling=scaling,
    )

    # No full-merge: stored base is the salient slice only [out, group_k], not [out, in_f]
    assert inj.W_base_salient.shape == (3 * out_f, gk), "must store only the [sum_out, group_k] slice"
    assert gk < in_f, "test should exercise a proper slice (group_k < in_features)"
    _ok("no full-weight materialization (salient slice only)")

    # Original projections are untouched objects (forward not replaced)
    assert attn.q_proj is q and type(q) is FakeLoRALinear
    _ok("BnB/LoRA projection objects not replaced")

    h = torch.randn(2, 7, in_f)
    q_out, k_out, v_out = attn(h)

    # Reference: original base+lora forward, plus the fused delta on the salient slice
    X_S = h[..., :gk]
    for name, proj, out in [("q", q, q_out), ("k", k, k_out), ("v", v, v_out)]:
        base_lora = (F.linear(h, proj.base_layer.weight)
                     + F.linear(F.linear(h, proj.lora_A["default"].weight),
                                proj.lora_B["default"].weight) * scaling)
        W_base_S = proj.base_layer.weight[:, :gk]
        ref = base_lora + ref_delta_out(W_base_S, proj.lora_A["default"].weight[:, :gk],
                                        proj.lora_B["default"].weight, X_S, gs, 4, False, scaling)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), f"attn {name} output mismatch"
        # And the original forward really ran: output minus delta == base+lora
        assert torch.allclose(out - (ref - base_lora), base_lora, atol=1e-5)
    _ok("q/k/v outputs = original forward + fused QAT delta")

    inj.remove()
    q_out2, _, _ = attn(h)
    base_lora_q = (F.linear(h, q.base_layer.weight)
                   + F.linear(F.linear(h, q.lora_A["default"].weight),
                              q.lora_B["default"].weight) * scaling)
    assert torch.allclose(q_out2, base_lora_q, atol=1e-5), "after remove(), delta must be gone"
    _ok("remove() uninstalls hooks cleanly")


def test_mlp_and_down_injectors():
    print("test_mlp_and_down_injectors")
    inter, in_f, gk, gs, rank, scaling = 24, 160, 128, 64, 4, 1.0
    gate = FakeLoRALinear(inter, in_f, rank, scaling)
    up   = FakeLoRALinear(inter, in_f, rank, scaling)

    class FakeMLP(nn.Module):
        def __init__(s): super().__init__(); s.gate_proj = gate; s.up_proj = up
        def forward(s, h): return s.gate_proj(h), s.up_proj(h)

    mlp = FakeMLP()
    inj = FusedMLPQATInjector(mlp, gate, up, group_k=gk, group_size=gs,
                              q_bits=4, symmetric=False, lora_scaling=scaling)
    assert inj.W_base_salient.shape == (2 * inter, gk)
    h = torch.randn(2, 4, in_f)
    g_out, u_out = mlp(h)
    X_S = h[..., :gk]
    for proj, out in [(gate, g_out), (up, u_out)]:
        base_lora = (F.linear(h, proj.base_layer.weight)
                     + F.linear(F.linear(h, proj.lora_A["default"].weight),
                                proj.lora_B["default"].weight) * scaling)
        ref = base_lora + ref_delta_out(proj.base_layer.weight[:, :gk],
                                        proj.lora_A["default"].weight[:, :gk],
                                        proj.lora_B["default"].weight, X_S, gs, 4, False, scaling)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    _ok("gate/up fused injector matches reference")

    # down_proj single injector (forward_hook on the proj itself)
    down_in = 160
    down = FakeLoRALinear(32, down_in, rank, scaling)
    dinj = DownProjQATInjector(down, group_k=gk, group_size=gs,
                               q_bits=4, symmetric=False, lora_scaling=scaling)
    assert dinj.W_base_salient.shape == (32, gk)
    x = torch.randn(2, 4, down_in)
    out = down(x)
    base_lora = (F.linear(x, down.base_layer.weight)
                 + F.linear(F.linear(x, down.lora_A["default"].weight),
                            down.lora_B["default"].weight) * scaling)
    ref = base_lora + ref_delta_out(down.base_layer.weight[:, :gk],
                                    down.lora_A["default"].weight[:, :gk],
                                    down.lora_B["default"].weight, x[..., :gk], gs, 4, False, scaling)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    _ok("down_proj single injector matches reference")


def test_grad_flow_to_lora():
    print("test_grad_flow_to_lora")
    out_f, in_f, gk, gs, rank, scaling = 16, 128, 128, 64, 4, 2.0
    q = FakeLoRALinear(out_f, in_f, rank, scaling)
    k = FakeLoRALinear(out_f, in_f, rank, scaling)
    v = FakeLoRALinear(out_f, in_f, rank, scaling)
    attn = FakeAttn(q, k, v)
    FusedAttnQATInjector(attn, q, k, v, group_k=gk, group_size=gs,
                         q_bits=4, symmetric=False, lora_scaling=scaling)
    h = torch.randn(2, 5, in_f)
    loss = sum(o.pow(2).sum() for o in attn(h))
    loss.backward()
    for proj in (q, k, v):
        gA = proj.lora_A["default"].weight.grad
        gB = proj.lora_B["default"].weight.grad
        assert gA is not None and gB is not None and torch.isfinite(gA).all() and torch.isfinite(gB).all()
    _ok("gradients reach LoRA A/B through the fused STE residual")


def test_online_group_hadamard():
    print("test_online_group_hadamard")
    gk, gs, rank, scaling, qb, sym = 128, 32, 8, 2.0, 4, False
    out = 16
    W_base = torch.randn(out, gk) * 0.05
    A_S    = torch.randn(rank, gk) * 0.02
    B      = torch.randn(out, rank) * 0.02
    X_S    = torch.randn(4, gk)
    H      = _build_hadamard(gk)

    # H is a normalized, symmetric, self-inverse Walsh-Hadamard matrix
    assert torch.allclose(H @ H, torch.eye(gk), atol=1e-5), "H must satisfy H@H=I"
    assert torch.allclose(H, H.T, atol=1e-6), "H must be symmetric"
    _ok("Hadamard is symmetric + self-inverse")

    # 1) injector-with-hadamard == explicit rotated reference
    (delta_h,) = fused_qat_residual_outputs(
        W_base, [A_S], [B], (out,), X_S, gs, qb, sym, scaling, hadamard=H,
    )
    W_curr  = W_base + (B @ A_S) * scaling
    W_rot   = W_curr @ H
    delta_rot = group_fakequant(W_rot, gs, qb, sym) - W_rot
    ref     = (X_S @ H) @ delta_rot.T
    assert torch.allclose(delta_h, ref, atol=1e-5, rtol=1e-4), "hadamard injection != reference"
    _ok("hadamard injection matches rotated reference")

    # 2) bake-back equivalence: training (main + inject) == deployment dense weight
    #    deployment salient weight = dq(Q(W_curr@H)) @ H  (H baked back); output via plain matmul
    train_out  = (X_S @ W_curr.T) + delta_h
    W_dense    = group_fakequant(W_rot, gs, qb, sym) @ H           # dq(Q(W@H)) @ H
    deploy_out = X_S @ W_dense.T
    assert torch.allclose(train_out, deploy_out, atol=1e-4, rtol=1e-3), \
        "export bake-back must equal training injection"
    _ok("training injection == export bake-back (bit-equivalent deploy)")

    # 3) rotation actually changes the quant grid vs the non-hadamard path
    (delta_plain,) = fused_qat_residual_outputs(
        W_base, [A_S], [B], (out,), X_S, gs, qb, sym, scaling, hadamard=None,
    )
    assert not torch.allclose(delta_h, delta_plain, atol=1e-3), \
        "hadamard path should differ from the concentrated-group path"
    _ok("hadamard path differs from non-hadamard path")

    # 4) injector flag path: FusedAttnQATInjector(online_group_hadamard=True) builds H buffer
    q = FakeLoRALinear(out, gk, rank, scaling)
    k = FakeLoRALinear(out, gk, rank, scaling)
    v = FakeLoRALinear(out, gk, rank, scaling)
    attn = FakeAttn(q, k, v)
    inj = FusedAttnQATInjector(
        attn, q, k, v, group_k=gk, group_size=gs, q_bits=qb,
        symmetric=sym, lora_scaling=scaling, online_group_hadamard=True,
    )
    assert inj.group_hadamard is not None and inj.group_hadamard.shape == (gk, gk)
    h = torch.randn(2, gk)
    q_out, _, _ = attn(h)
    # q output = base+lora + hadamard-rotated delta
    base_lora = (F.linear(h, q.base_layer.weight)
                 + F.linear(F.linear(h, q.lora_A["default"].weight),
                            q.lora_B["default"].weight) * scaling)
    Wq_curr = q.base_layer.weight[:, :gk] + (q.lora_B["default"].weight @ q.lora_A["default"].weight[:, :gk]) * scaling
    Wq_rot  = Wq_curr @ H
    ref_q   = base_lora + (h @ H) @ (group_fakequant(Wq_rot, gs, qb, sym) - Wq_rot).T
    assert torch.allclose(q_out, ref_q, atol=1e-5, rtol=1e-4)
    inj.remove()
    _ok("FusedAttnQATInjector(online_group_hadamard=True) end-to-end")


def main():
    test_fakequant()
    test_fused_residual()
    test_attn_injector()
    test_mlp_and_down_injectors()
    test_grad_flow_to_lora()
    test_online_group_hadamard()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
