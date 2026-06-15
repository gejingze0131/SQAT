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
    group_k_for_module_name,
    auto_segment_with_fixed_group_k,
    compute_fakequant_param_stats,
    FusedAttnQATInjector,
    FusedMLPQATInjector,
    DownProjQATInjector,
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


def test_awq_scale():
    print("test_awq_scale")
    gk, gs, rank, scaling, qb, sym = 128, 32, 8, 2.0, 4, False
    out = 16
    W_base = torch.randn(out, gk) * 0.05
    A_S    = torch.randn(rank, gk) * 0.02
    B      = torch.randn(out, rank) * 0.02
    X_S    = torch.randn(4, gk)
    S      = (1.0 + torch.rand(gk)).clamp(max=2.0)            # per-input-channel scale in [1, 2]

    # 1) injector-with-awq_s == explicit amplified-space reference (X_S unchanged, /S in weight)
    (delta_a,) = fused_qat_residual_outputs(
        W_base, [A_S], [B], (out,), X_S, gs, qb, sym, scaling, awq_s=S,
    )
    W_curr  = W_base + (B @ A_S) * scaling
    s_row   = S.view(1, -1)
    delta_amp = group_fakequant(W_curr * s_row, gs, qb, sym) / s_row - W_curr
    ref     = X_S @ delta_amp.T
    assert torch.allclose(delta_a, ref, atol=1e-5, rtol=1e-4), "awq injection != amplified reference"
    _ok("awq injection matches amplified-space reference")

    # 2) bake-back equivalence: training (main + inject) == deployment dense weight quant(W*S)/S
    train_out  = (X_S @ W_curr.T) + delta_a
    W_dense    = group_fakequant(W_curr * s_row, gs, qb, sym) / s_row     # deployed = quant(W*S)/S
    deploy_out = X_S @ W_dense.T
    assert torch.allclose(train_out, deploy_out, atol=1e-4, rtol=1e-3), \
        "export /S bake-back must equal training injection"
    _ok("training injection == export /S bake-back (bit-equivalent deploy)")

    # 3) S=1 reduces exactly to the plain (no-scale) path
    (delta_plain,) = fused_qat_residual_outputs(
        W_base, [A_S], [B], (out,), X_S, gs, qb, sym, scaling, awq_s=None,
    )
    (delta_one,) = fused_qat_residual_outputs(
        W_base, [A_S], [B], (out,), X_S, gs, qb, sym, scaling, awq_s=torch.ones(gk),
    )
    assert torch.allclose(delta_one, delta_plain, atol=1e-6), "awq_s=1 must equal the plain path"
    assert not torch.allclose(delta_a, delta_plain, atol=1e-3), \
        "a non-trivial S must change the quant grid vs the plain path"
    _ok("awq_s=1 == plain path; non-trivial S differs")

    # 4) injector flag path: FusedAttnQATInjector(awq_s=S) builds the buffer + end-to-end q output
    q = FakeLoRALinear(out, gk, rank, scaling)
    k = FakeLoRALinear(out, gk, rank, scaling)
    v = FakeLoRALinear(out, gk, rank, scaling)
    attn = FakeAttn(q, k, v)
    inj = FusedAttnQATInjector(
        attn, q, k, v, group_k=gk, group_size=gs, q_bits=qb,
        symmetric=sym, lora_scaling=scaling, awq_s=S,
    )
    assert inj.awq_s is not None and inj.awq_s.shape == (gk,)
    h = torch.randn(2, gk)
    q_out, _, _ = attn(h)
    base_lora = (F.linear(h, q.base_layer.weight)
                 + F.linear(F.linear(h, q.lora_A["default"].weight),
                            q.lora_B["default"].weight) * scaling)
    Wq_curr = q.base_layer.weight[:, :gk] + (q.lora_B["default"].weight @ q.lora_A["default"].weight[:, :gk]) * scaling
    ref_q   = base_lora + h @ (group_fakequant(Wq_curr * s_row, gs, qb, sym) / s_row - Wq_curr).T
    assert torch.allclose(q_out, ref_q, atol=1e-5, rtol=1e-4)
    inj.remove()
    _ok("FusedAttnQATInjector(awq_s=S) end-to-end")


def test_per_module_group_k_meta():
    print("test_per_module_group_k_meta")
    meta = {
        "boundary_sizes": [2, 2],
        "segment_group_ks": [256, 64],
        "layer_group_ks": [256, 256, 64, 64],
        "down_layer_group_ks": [64, 128, 192, 128],
        "group_k": 256,
    }
    assert group_k_for_module_name("model.layers.0.self_attn.q_proj", meta) == 256
    assert group_k_for_module_name("model.layers.2.mlp.gate_proj", meta) == 64
    assert group_k_for_module_name("model.layers.0.mlp.down_proj", meta) == 64
    assert group_k_for_module_name("model.layers.2.mlp.down_proj", meta) == 192
    assert group_k_for_module_name("model.layers.2.self_attn.o_proj", meta) == 0

    legacy = {"boundary_sizes": [2, 2], "segment_group_ks": [256, 64], "group_k": 256}
    assert group_k_for_module_name("model.layers.2.mlp.down_proj", legacy) == 64
    _ok("down_proj uses down_layer_group_ks; legacy meta still falls back to residual k")


def test_fixed_group_k_auto_segments():
    print("test_fixed_group_k_auto_segments")
    num_layers, hidden = 4, 16
    second_moments = {}
    for layer in range(num_layers):
        attn = torch.ones(hidden)
        mlp = torch.ones(hidden)
        attn[layer:layer + 2] = 100.0 + layer
        mlp[layer + 4:layer + 6] = 90.0 + layer
        second_moments[(layer, "attn")] = attn
        second_moments[(layer, "mlp")] = mlp

    boundary_sizes, segment_group_ks, summary = auto_segment_with_fixed_group_k(
        second_moments=second_moments,
        hidden_size=hidden,
        num_layers=num_layers,
        group_k=8,
        max_segments=2,
        outlier_log_sigma=1.0,
    )
    assert sum(boundary_sizes) == num_layers
    assert segment_group_ks == [8] * len(boundary_sizes)
    assert summary["fixed_group_k"] == 8
    assert all(seg["group_k"] == 8 for seg in summary["segments"])
    _ok("fixed group_k applies to every auto-selected residual segment")


def test_fakequant_param_stats():
    print("test_fakequant_param_stats")

    class FakeSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeSelfAttn()
            self.mlp = FakeMLP()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])

    stats = compute_fakequant_param_stats(
        FakeModel(),
        layer_group_ks=[2, 4],
        down_layer_group_ks=[6, 8],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    assert stats["fakequant_params"] == 448
    assert stats["qat_target_weight_params"] == 1152
    assert stats["lora_target_weight_params"] == 1280
    assert abs(stats["ratio_of_qat_target_weights"] - 448 / 1152) < 1e-12
    assert stats["by_projection"]["down_proj"]["fakequant_params"] == 112
    _ok("fakequant parameter coverage counts salient columns only")


def main():
    test_fakequant()
    test_fused_residual()
    test_attn_injector()
    test_mlp_and_down_injectors()
    test_grad_flow_to_lora()
    test_awq_scale()
    test_per_module_group_k_meta()
    test_fixed_group_k_auto_segments()
    test_fakequant_param_stats()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
