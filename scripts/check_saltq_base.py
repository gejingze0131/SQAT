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
    the "quantized" weights are still effectively full precision.
  * group_k per layer — o_proj must be 0 (no contiguous salient slice exists for it), everything
    else a positive multiple of group_size.

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

    print(f"base      : {args.base}")
    print(f"grid      : INT{qb} {'asym' if not sym else 'sym'}, group_size={gs}, "
          f"levels [{qmin}, {qmax}]")
    print(f"layers    : {len(meta['layers'])}")
    pc = meta["param_counts"]
    print(f"freedom   : {pc['trainable_salient_weights']/1e6:.1f}M full + "
          f"{pc['trainable_qparams']/1e6:.1f}M affine + {pc['frozen_codes']/1e6:.1f}M frozen\n")

    # group_k structure
    bad = []
    for n, info in meta["layers"].items():
        gk = int(info["group_k"])
        term = n.split(".")[-1]
        if term == "o_proj" and gk != 0:
            bad.append(f"{n}: o_proj must have group_k=0, got {gk}")
        elif term != "o_proj" and (gk <= 0 or gk % gs):
            bad.append(f"{n}: group_k={gk} is not a positive multiple of {gs}")
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
                print(f"  reconstruction: relative Frobenius error {rel:.2f}%  "
                      f"(INT{qb} should be tens of percent)")
                if rel < 1.0:
                    print("  ^ PROBLEM: error is near zero — these weights are effectively fp")
                    ok = False
            print()

    print("RESULT:", "base looks sane" if (ok and not bad) else "SOMETHING IS WRONG")
    return 0 if (ok and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
