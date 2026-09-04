#!/usr/bin/env python
"""
Why a SALT-Q base can show >100% weight-space reconstruction error on layer 0 and still be right.

THE ALARM. scripts/check_saltq_base.py compares the frozen non-salient codes, dequantized, against
the permuted fp16 weights they came from, and fails a base whose relative Frobenius error reaches
100% ("worse than predicting zero"). On every `salient_init=gptq` / `gptq_latent` base built so far
— MetaMath INT2 (the one behind GSM8K 56.48) and Alpaca INT2 alike — layer 0's q_proj and k_proj
trip that rule at 160-172%, while every other layer sits at the expected 45-50%.

THE CAUSE. `salient_init=gptq*` sets obs_salient=True, which makes src/gptq.gptq_quantize_layer
set group_k=0 and run ONE OBS problem over the whole matrix, salient columns first. The salient
columns are the highest-E[x^2] channels by construction, so their INT2 quantization error is large
— and OBS deliberately absorbs it into the columns quantized later, i.e. into the non-salient
block. The non-salient codes therefore no longer approximate W[:, group_k:]; they approximate
W[:, group_k:] PLUS a correction that cancels the salient slice's error at the OUTPUT. Measuring
them against W[:, group_k:] measures the wrong target, and the more the salient slice needs
correcting the "worse" that number gets.

Layer 0 is where this is extreme: its attention input is the RMSNorm'd embedding, whose energy is
concentrated in a handful of massive-activation channels (measured below: the 256 salient columns
carry ~97% of E[x^2], max/median diagonal ~3e4). Whether a projection blows up is then decided by
its WEIGHTS, not by the calibration set — q_proj and k_proj put large weights on exactly those
channels (|W_salient| rms ~3.7x their own non-salient rms) and blow up; v_proj, whose salient
weights are SMALLER than its non-salient ones, does not, on the very same input and Hessian.

THE TEST. Weight-space distance is not what GPTQ minimizes; output error is. This script measures
it directly on layer 0, the one layer whose input needs no forward pass (embedding -> RMSNorm),
for the deployed weights, for the same weights with the compensation removed (RTN on the identical
grid), and for the salient slice put back in fp16.

    python scripts/diagnose_obs_compensation.py --base outputs/<run>/saltq_base_2bit_g32 \
        --permuted outputs/<run>/permuted_fp16_base

Reference numbers, Alpaca INT2 g32 k=256 base (salient_init=gptq_latent), 2 x 2048 C4 tokens:

    module   |W_sal|rms  |W_non|rms   wgt err%   sat%   OUT err%   OUT no-comp%
    q_proj       0.0374      0.0100     171.9    70.6       5.57           7.56
    k_proj       0.0370      0.0123     160.5    62.1       5.30           8.32
    v_proj       0.0082      0.0112      58.7    29.7      12.56          26.45

i.e. every layer whose weight error looks catastrophic has a LOWER output error than the same
layer with the compensation switched off. The 171% is the price OBS pays for the 5.57%, and the
minmax-init control base (obs_salient=False, no compensation, weight error a healthy 64%) scores
9.98% on the same layer.
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qat_saltq import SALTQ_BASE_FILENAME, SALTQ_META_FILENAME


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / b.norm().item() * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--base", default="outputs/saltq/saltq_base")
    ap.add_argument("--permuted", default=None,
                    help="permuted fp16 base (default: read from the SALT-Q meta)")
    ap.add_argument("--calib", default="datasets/c4_calib_1024.json",
                    help="raw-text JSON the activations are measured on")
    ap.add_argument("--windows", type=int, default=2, help="windows of 2048 tokens")
    args = ap.parse_args()

    meta = torch.load(os.path.join(args.base, SALTQ_META_FILENAME),
                      map_location="cpu", weights_only=False)
    gs, qmax = meta["group_size"], 2 ** meta["q_bits"] - 1
    permuted = args.permuted or meta["permuted_base_dir"]
    print(f"base        : {args.base}")
    print(f"salient_init: {meta.get('salient_init', 'minmax')}   INT{meta['q_bits']} g{gs}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(permuted)
    ids = []
    for t in json.load(open(args.calib)):
        ids.extend(tok(t if isinstance(t, str) else t["text"],
                       add_special_tokens=False).input_ids + [tok.eos_token_id])
        if len(ids) >= args.windows * 2048:
            break
    x_ids = torch.tensor(ids[:args.windows * 2048]).view(args.windows, 2048)

    f = safe_open(os.path.join(args.base, SALTQ_BASE_FILENAME), framework="pt", device="cpu")
    g = safe_open(os.path.join(permuted, "model.safetensors"), framework="pt", device="cpu")

    # Layer 0's attention input needs no forward pass: it is the permuted embedding, RMSNormed.
    emb = g.get_tensor("model.embed_tokens.weight").float()
    h = emb[x_ids].reshape(-1, emb.shape[1])
    ln = g.get_tensor("model.layers.0.input_layernorm.weight").float()
    x = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * ln

    gk = int(meta["layers"]["model.layers.0.self_attn.q_proj"]["group_k"])
    d = (x * x).mean(0)
    print(f"\nlayer-0 attention input, E[x^2] per column (permuted order):")
    print(f"  salient [0:{gk}] mean {d[:gk].mean():.4g}   non-salient mean {d[gk:].mean():.4g}   "
          f"ratio {d[:gk].mean() / d[gk:].mean():.0f}x")
    print(f"  the {gk} salient columns carry {100 * d[:gk].sum() / d.sum():.1f}% of E[x^2]; "
          f"max/median diagonal {d.max() / d.median():.0f}x")

    print(f"\n{'module':<9} {'|W_sal|rms':>10} {'|W_non|rms':>10} {'wgt err%':>9} {'RTNwgt%':>8} "
          f"{'sat%':>6} | {'OUT err%':>9} {'OUT no-comp%':>13} {'OUT sal-fp16%':>14}")
    for term in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{term}"
        codes = f.get_tensor(f"{name}.codes").float()
        s, z = f.get_tensor(f"{name}.s_n"), f.get_tensor(f"{name}.z_n")
        W = g.get_tensor(f"{name}.weight").float()
        out, in_n = codes.shape
        rec = ((codes.view(out, in_n // gs, gs) - z.unsqueeze(-1))
               * s.unsqueeze(-1)).reshape(out, in_n)
        Wn = W[:, gk:]
        # RTN on the SAME stored grid = the identical quantizer with the compensation removed.
        rtn_i = torch.round(torch.clamp(Wn.view(out, in_n // gs, gs) / s.unsqueeze(-1)
                                        + z.unsqueeze(-1), 0, qmax))
        rtn = ((rtn_i - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(out, in_n)
        # What the salient slice actually deploys at step 0: fakequant on its own stored grid.
        w_s = f.get_tensor(f"{name}.w_s").float()
        s_s, z_s = f.get_tensor(f"{name}.s_s"), f.get_tensor(f"{name}.z_s")
        wg = w_s.view(out, w_s.shape[1] // gs, gs)
        sal = ((torch.round(torch.clamp(wg / s_s.unsqueeze(-1) + z_s.unsqueeze(-1), 0, qmax))
                - z_s.unsqueeze(-1)) * s_s.unsqueeze(-1)).reshape(out, w_s.shape[1])
        y = x @ W.T
        sat = ((codes == 0) | (codes == qmax)).float().mean().item() * 100
        print(f"{term:<9} {W[:, :gk].pow(2).mean().sqrt():>10.4f} "
              f"{Wn.pow(2).mean().sqrt():>10.4f} {rel(rec, Wn):>9.2f} {rel(rtn, Wn):>8.2f} "
              f"{sat:>6.1f} | {rel(x @ torch.cat([sal, rec], 1).T, y):>9.2f} "
              f"{rel(x @ torch.cat([sal, rtn], 1).T, y):>13.2f} "
              f"{rel(x @ torch.cat([W[:, :gk], rec], 1).T, y):>14.2f}")
    print("\nOUT no-comp% > OUT err% means the compensation is doing its job: the weights are far "
          "from fp16\nbecause that is what makes the OUTPUT close to fp16. OUT sal-fp16% is the "
          "same non-salient\ncodes with the fp16 salient slice put back — it breaks the "
          "cancellation the codes were built for,\nso for v_proj it is WORSE than the fully "
          "quantized layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
