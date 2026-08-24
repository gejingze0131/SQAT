#!/usr/bin/env python
"""Does the optimizer preserve the gradient's concentration, per parameter tensor?

    stable rank = ||M||_F^2 / sigma_1^2  --  "how many directions is M spread over".
    flattening  = stable_rank(applied update) / stable_rank(gradient), PAIRED inside each tensor.

1.0x means the optimizer kept the gradient's shape. >>1 means it spread a concentrated gradient
over many directions, which for a per-(output, group) coefficient is the difference between a
decisive correction and a diffuse one.

    m = exp_avg                 the (EMA of the) gradient
    u = m / (sqrt(v) + eps)     what Adam actually applies

TWO TRAPS, BOTH OF WHICH COST A WRONG NUMBER HERE:

  1. A ratio of two MARGINAL medians is not a flattening ratio -- the numerator and denominator
     can come from different tensors. An earlier read of the z-only INT2 state quoted
     "1.83 -> 6.87 = 3.8x" that way; paired per tensor the same data gives 2.92x.
  2. SALT-Q's zero-point optimizer group holds saltq_z ([out, n_nonsal_g]) AND the salient slice's
     lsq_w_zp ([out, n_sal_g]) -- 224 + 192 tensors of very different widths -- so anything pooled
     over that group mixes two unrelated objects. Bucket by shape.

Usage
-----
    python scripts/measure_update_spectrum.py <checkpoint>/optimizer.pt --label "..."

Reads only the optimizer state, so it runs on CPU in seconds and needs no model.
"""
import argparse

import torch
from collections import defaultdict

def srank(M):
    e = torch.linalg.svdvals(M) ** 2
    return (e.sum() / e[0]).item(), (e[0] / e.sum()).item()

ap = argparse.ArgumentParser()
ap.add_argument("optimizer_pt")
ap.add_argument("--label", default="")
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--per_shape", type=int, default=20,
                help="tensors sampled per shape bucket; the medians are stable well below this")
args = ap.parse_args()
sd = torch.load(args.optimizer_pt, map_location="cpu", weights_only=False)
print(f"\n=== {args.label or args.optimizer_pt} ===")
print(f"{'grp':>3} {'lr':>10} {'shape':>14} {'n':>3}  {'sr(grad)':>9}{'top1':>7}  "
      f"{'sr(upd)':>9}{'top1':>7}  {'PAIRED flattening':>18}", flush=True)
for gi, g in enumerate(sd["param_groups"]):
    idxs = [i for i in g["params"] if i in sd["state"]]
    b = defaultdict(list)
    for i in idxs:
        if sd["state"][i]["exp_avg"].dim() == 2:
            b[tuple(sd["state"][i]["exp_avg"].shape)].append(i)
    for shp in sorted(b, key=lambda s: -len(b[s])):
        rows = []
        for i in b[shp][:args.per_shape]:
            st = sd["state"][i]; t = float(st.get("step", 0))
            m = st["exp_avg"].double() / (1 - 0.9 ** t)
            v = st["exp_avg_sq"].double() / (1 - 0.999 ** t)
            a, ta = srank(m); c, tc = srank(m / (v.sqrt() + args.eps))
            rows.append((a, ta, c, tc, c / a))
        med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
        print(f"{gi:>3} {g.get('initial_lr', float('nan')):>10.2e} {str(shp):>14} {len(rows):>3}  "
              f"{med(0):>9.2f}{med(1)*100:>6.1f}%  {med(2):>9.2f}{med(3)*100:>6.1f}%  "
              f"{med(4):>17.2f}x", flush=True)
