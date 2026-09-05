#!/usr/bin/env python
"""
Sanity-check a built SALT-Q base: are the frozen codes real low-bit codes, and is the salient
split where the metadata says it is?

Written after a false alarm. The initial training loss came in at 1.22 where ~1.8 was expected
(anchored on Permuted-SQAT's reported step-0 loss), which looked like "quantization is not
actually in effect". It was not — SALT-Q's step-0 model is GPTQ error-compensated across the
WHOLE weight, whereas PSQAT's step-0 crushes only its salient columns with min-max RTN, i.e.
exactly the columns min-max handles worst. Starting lower is the expected consequence, not a bug.

A loss value is a bad proxy for "is the quantizer wired up". These are the direct checks:

  * code range and histogram — INT-b affine codes must span [0, 2^b - 1] and, on roughly Gaussian
    weights, land in a bell shape across the levels. All-zero, single-valued, or clipped-to-one-end
    histograms mean the grid is broken.
  * reconstruction error — (q - z) * s against the permuted fp16 weight it came from. INT3 g64 is
    brutal; a few tens of percent relative Frobenius error is CORRECT. A near-zero error would mean
    the "quantized" weights are still effectively full precision. Reported next to the SAME
    quantizer with the error compensation removed (RTN on the identical stored grid), because on a
    salient_init=gptq* base the non-salient codes do NOT target their own fp16 weights — see below.
  * group_k per layer — o_proj must be 0 (no contiguous salient slice exists for it), everything
    else a positive multiple of group_size.

ON A salient_init=gptq / gptq_latent BASE, A >100% LAYER IS NOT NECESSARILY BROKEN. That setting
runs ONE OBS problem over the whole matrix with the salient (highest-E[x^2]) columns first, so the
salient slice's INT2 error is deliberately absorbed by the non-salient block: those codes target
"W_nonsalient + a correction that cancels the salient error at the OUTPUT", not W_nonsalient. Layer
0's q_proj/k_proj measure 160-172% on every such base built so far (MetaMath INT2, the one behind
GSM8K 56.48, and Alpaca INT2) while their OUTPUT error is 5.3-5.6% — LOWER than the 7.6-8.3% the
same weights get with the compensation removed. So this script now fails only when the grid itself
is broken (the RTN control is also >=100%), when the base does not use OBS over the salient slice,
or when MOST sampled layers are >=100% — the uniform pattern of the INT3 disaster below, as opposed
to the isolated layer-0 q/k signature of compensation. Measure the output directly with
scripts/diagnose_obs_compensation.py before rebuilding anything.

Runs read-only off disk on CPU, so it is safe to run against a base while training uses it.

Usage:
    python scripts/check_saltq_base.py                       # outputs/saltq/saltq_base
    python scripts/check_saltq_base.py --base <dir> --layers 6
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open

from src.qat_saltq import SALTQ_BASE_FILENAME, SALTQ_META_FILENAME


def main():
    ap = argparse.ArgumentParser(description="Sanity-check a SALT-Q frozen-code base")
    ap.add_argument("--base", default="outputs/saltq/saltq_base")
    ap.add_argument("--permuted", default=None,
                    help="Permuted fp16 base (default: read from the SALT-Q meta)")
    ap.add_argument("--layers", type=int, default=4, help="How many layers to sample")
    args = ap.parse_args()

    meta = torch.load(os.path.join(args.base, SALTQ_META_FILENAME),
                      map_location="cpu", weights_only=False)
    gs, qb, sym = meta["group_size"], meta["q_bits"], meta["symmetric"]
    qmin, qmax = (0, 2 ** qb - 1) if not sym else (-(2 ** (qb - 1)), 2 ** (qb - 1) - 1)
    permuted = args.permuted or meta["permuted_base_dir"]
    salient_init = str(meta.get("salient_init", "minmax"))
    # Whole-matrix OBS: the salient slice was quantized inside the same sweep, so its error was
    # compensated INTO the non-salient codes. That changes what the codes are supposed to equal.
    obs_salient = salient_init in ("gptq", "gptq_latent")

    print(f"base      : {args.base}")
    print(f"grid      : INT{qb} {'asym' if not sym else 'sym'}, group_size={gs}, "
          f"levels [{qmin}, {qmax}]")
    print(f"init      : salient_init={salient_init}"
          + ("  (whole-matrix OBS: the non-salient codes carry the salient slice's correction)"
             if obs_salient else ""))
    print(f"layers    : {len(meta['layers'])}")
    pc = meta["param_counts"]
    print(f"freedom   : {pc['trainable_salient_weights']/1e6:.1f}M full + "
          f"{pc['trainable_qparams']/1e6:.1f}M affine + {pc['frozen_codes']/1e6:.1f}M frozen\n")

    # group_k structure
    train_salient = bool(meta.get("train_salient", True))
    bad = []
    for n, info in meta["layers"].items():
        gk = int(info["group_k"])
        term = n.split(".")[-1]
        if not train_salient:
            # z-only-everywhere ablation: EVERY layer is folded into the frozen-codes-only
            # branch, so group_k=0 is expected across the board, not just for o_proj.
            if gk != 0:
                bad.append(f"{n}: train_salient=False base must have group_k=0 everywhere, got {gk}")
        elif term == "o_proj" and gk != 0:
            bad.append(f"{n}: o_proj must have group_k=0, got {gk}")
        elif term != "o_proj" and (gk <= 0 or gk % gs):
            bad.append(f"{n}: group_k={gk} is not a positive multiple of {gs}")
    print(f"train_salient: {train_salient}")
    print(f"group_k structure: {'OK' if not bad else 'PROBLEM'}")
    for b in bad[:5]:
        print(f"  {b}")

    names = [n for n in meta["layers"]][:: max(1, len(meta["layers"]) // args.layers)][:args.layers]
    perm_file = os.path.join(permuted, "model.safetensors")
    have_perm = os.path.exists(perm_file)
    if not have_perm:
        print(f"\n(permuted fp16 base not at {perm_file}; skipping reconstruction error)")

    print()
    ok = True
    over_100 = []
    with safe_open(os.path.join(args.base, SALTQ_BASE_FILENAME), framework="pt", device="cpu") as f:
        g = safe_open(perm_file, framework="pt", device="cpu") if have_perm else None
        for n in names:
            gk = int(meta["layers"][n]["group_k"])
            codes = f.get_tensor(f"{n}.codes")
            s = f.get_tensor(f"{n}.s_n")
            z = f.get_tensor(f"{n}.z_n")
            out, in_n = codes.shape
            hist = torch.bincount(codes.flatten().to(torch.int64) - qmin,
                                  minlength=qmax - qmin + 1).float()
            used = int((hist > 0).sum())
            print(f"{n}  group_k={gk}  non-salient={in_n}")
            print(f"  codes {codes.min().item()}..{codes.max().item()}   "
                  f"levels used {used}/{qmax - qmin + 1}   "
                  f"dist% {(hist / hist.sum() * 100).round().to(torch.int64).tolist()}")
            if used <= 2:
                print("  ^ PROBLEM: codes collapsed onto <=2 levels — the grid is broken")
                ok = False
            if g is not None:
                W = g.get_tensor(f"{n}.weight").float()[:, gk:]
                q = codes.float().view(out, in_n // gs, gs)
                rec = ((q - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(out, in_n)
                rel = (rec - W).norm() / W.norm() * 100
                # The SAME quantizer with the error compensation removed: round-to-nearest of the
                # fp16 weights onto the identical stored grid. It isolates the two causes of a
                # large number — a grid that cannot represent these weights (RTN is bad too) from
                # compensation that moved them on purpose (RTN is fine, the codes are not).
                rtn_i = torch.round(torch.clamp(
                    W.view(out, in_n // gs, gs) / s.unsqueeze(-1) + z.unsqueeze(-1), qmin, qmax))
                rtn = ((rtn_i - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(out, in_n)
                rel_rtn = (rtn - W).norm() / W.norm() * 100
                print(f"  reconstruction: relative Frobenius error {rel:.2f}%  "
                      f"(no-compensation control: RTN on the same grid {rel_rtn:.2f}%)")
                # A grid that cannot even hold the RTN of its own weights is broken outright, and
                # that is true whatever the init was.
                if rel_rtn >= 100.0:
                    print(f"  ^ PROBLEM: even RTN on this grid is {rel_rtn:.1f}% — the grid itself "
                          f"cannot represent these weights. Rebuild the base.")
                    ok = False
                elif rel >= 100.0:
                    # >= 100% means the dequantized weight is further from W than ZERO is. On a
                    # NON-OBS base the codes then carry no usable signal: an INT3 g64 base measured
                    # 114% on every layer but the first and its untrained model scored a 10.6 LM
                    # loss (uniform noise over a 32k vocab is 10.37), i.e. the run that used it was
                    # recovering from a destroyed model rather than fine-tuning.
                    # On an OBS base W is the WRONG TARGET for the non-salient codes (see the module
                    # docstring), so this is recorded and reported at the end, where the number of
                    # layers involved distinguishes compensation from a real disaster.
                    over_100.append((n, rel, rel_rtn))
                    if not obs_salient:
                        print(f"  ^ PROBLEM: {rel:.1f}% >= 100% — reconstruction is worse than "
                              f"predicting zero. Do NOT train on this base; rebuild it.")
                        ok = False
                    else:
                        print(f"  ^ NOTE: {rel:.1f}% >= 100% while RTN on the same grid is fine "
                              f"({rel_rtn:.1f}%) — the signature of OBS compensation, not damage.")
                if rel < 1.0:
                    print("  ^ PROBLEM: error is near zero — these weights are effectively fp")
                    ok = False
            print()

    # An OBS base is allowed a MINORITY of >=100% layers (in practice layer 0/1 q_proj and k_proj,
    # whose salient columns carry ~97% of the input energy). A majority is the uniform pattern of a
    # broken base, and no amount of compensation explains it.
    if over_100 and obs_salient:
        n_over, n_seen = len(over_100), len(names)
        print(f"{n_over} of {n_seen} sampled layers exceed 100% while their RTN control does not:")
        for n, r, rr in over_100:
            print(f"  {n}  {r:.1f}%  (RTN {rr:.1f}%)")
        if n_over > n_seen // 2:
            print("  ^ PROBLEM: that is MOST of the sampled layers — too many to be the salient "
                  "slice's\n    compensation. Rebuild the base.")
            ok = False
        else:
            print("  ^ Expected on a salient_init=gptq* base: those codes target "
                  "W_nonsalient PLUS the\n    correction that cancels the salient slice's error "
                  "at the OUTPUT, so W_nonsalient is not\n    what they approximate. Confirm with "
                  "scripts/diagnose_obs_compensation.py, which measures\n    the output error "
                  "directly (Alpaca INT2: layer-0 q_proj 171.9% weight / 5.57% output, "
                  "against\n    7.56% for the same weights with the compensation removed).")

    print("RESULT:", "base looks sane" if (ok and not bad) else "SOMETHING IS WRONG")
    return 0 if (ok and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
