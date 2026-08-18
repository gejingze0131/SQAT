"""Equivalence test for permute_common.apply_awq_folding_fp32.

The fold is only legitimate if the model's OUTPUT is unchanged. Each pairing rests on a
single-consumer claim that this test is here to falsify if it is wrong:

    q/k/v  cols[:gk] *= S   <->  input_layernorm.weight[:gk]          /= S
    gate/up cols[:gk] *= S  <->  post_attention_layernorm.weight[:gk] /= S
    down    cols[:gk] *= S  <->  up_proj.weight[rows :gk, :]          /= S

In fp32 the fold is exact up to floating-point reassociation. In fp16 (the dtype the real
permuted base is stored in) it also pays one rounding of the rescaled weights, which is reported
rather than asserted away — that error is ~1e-3 relative against an INT2 quantization error of
tens of percent, but it is not zero and should not be claimed to be.
"""

import os
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.permute_common import apply_awq_folding_fp32  # noqa: E402


def build(dtype, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=8, max_position_embeddings=64,
    )
    m = LlamaForCausalLM(cfg).to(dtype)
    m.eval()
    # RMSNorm weights start at exactly 1.0; make them non-trivial so a wrong fold cannot pass by
    # accident (dividing 1.0 by S and multiplying back is too forgiving a test).
    with torch.no_grad():
        for lyr in m.model.layers:
            lyr.input_layernorm.weight.copy_(1.0 + 0.3 * torch.randn(cfg.hidden_size).to(dtype))
            lyr.post_attention_layernorm.weight.copy_(
                1.0 + 0.3 * torch.randn(cfg.hidden_size).to(dtype))
    return m, cfg


def run(m, ids):
    with torch.no_grad():
        return m(ids).logits.float()


def check(dtype, gk, gk_d, label):
    m, cfg = build(dtype)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    y0 = run(m, ids)

    torch.manual_seed(1)
    L = cfg.num_hidden_layers
    # S in [1, 2], matching compute_awq_scales' normalization (min 1, clamped to max_s).
    awq = {
        "attn": 1.0 + torch.rand(L, gk),
        "mlp": 1.0 + torch.rand(L, gk),
        "down": 1.0 + torch.rand(L, gk_d),
    }
    n = apply_awq_folding_fp32(m, awq, [gk] * L, [gk_d] * L)
    y1 = run(m, ids)

    denom = y0.abs().max().item()
    amax = (y1 - y0).abs().max().item()
    rel = amax / max(denom, 1e-12)
    print(f"  {label:26s} layers folded={n}  max|Δlogits|={amax:.3e}  relative={rel:.3e}")
    return rel


def check_noop():
    m, cfg = build(torch.float32)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    y0 = run(m, ids)
    n = apply_awq_folding_fp32(m, None, [32] * cfg.num_hidden_layers, [64] * cfg.num_hidden_layers)
    y1 = run(m, ids)
    ok = n == 0 and torch.equal(y0, y1)
    print(f"  {'awq_scales=None is a no-op':26s} folded={n}  identical={torch.equal(y0, y1)}")
    return ok


def main():
    print("test_awq_folding — output equivalence of the offline AWQ fold\n")
    print("fp32 (fold should be exact up to reassociation):")
    r32 = check(torch.float32, gk=32, gk_d=64, label="hidden=64 gk=32 gkd=64")
    r32b = check(torch.float32, gk=16, gk_d=32, label="hidden=64 gk=16 gkd=32")
    print("\nfp16 (the dtype the real permuted base is saved in):")
    r16 = check(torch.float16, gk=32, gk_d=64, label="hidden=64 gk=32 gkd=64")
    print("\nno-op path:")
    ok_noop = check_noop()

    print("\n" + "=" * 66)
    fails = []
    if r32 > 1e-5 or r32b > 1e-5:
        fails.append(f"fp32 fold not equivalence-preserving (rel {max(r32, r32b):.3e} > 1e-5)")
    if r16 > 5e-2:
        fails.append(f"fp16 fold error unexpectedly large (rel {r16:.3e} > 5e-2)")
    if not ok_noop:
        fails.append("awq_scales=None was not a no-op")
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print(f"PASS — fp32 exact to {max(r32, r32b):.1e} relative; fp16 costs {r16:.1e} "
          f"relative from one weight rounding (INT2 quantization error is ~1e-1).")


if __name__ == "__main__":
    main()
