#!/usr/bin/env python
"""Per-optimizer-group Adam diagnostics, read straight off a checkpoint's `optimizer.pt`.

WHY THIS EXISTS. SALT-Q's three tiers live in three different UNITS (salient weights and the LSQ
scale in weight units, the zero-point in quantization LEVELS). src/trainer.py already gives each
tier its own `lr` and `weight_decay` for exactly that reason -- see the comment block at
src/qat_saltq.py:101. But `eps` and `max_grad_norm` are still SHARED, and both of them are
ABSOLUTE quantities: whether eps=1e-8 is negligible depends entirely on the unit the parameter
lives in. This script measures whether it actually is.

Two numbers per group.

1. THE COHERENCE RATIO  r = E|m| / sqrt(E v).
   For a stationary per-coordinate gradient g_t = mu + sigma*eps_t and Adam's defaults
   (beta1=0.9, beta2=0.999), after bias correction:
       m -> mu, with residual std sigma*sqrt((1-b1)/(1+b1)) = 0.2294*sigma
       v -> mu^2 + sigma^2
       r  = sqrt(rho^2 + 0.0526) / sqrt(rho^2 + 1),      rho = mu/sigma  (per-coordinate SNR)
   so r inverts to      rho^2 = (r^2 - 0.0526) / (1 - r^2).
   r = 0.2294 is the PURE-NOISE FLOOR (rho = 0): Adam still takes a step of size ~lr, but in a
   direction uncorrelated with the previous one, so the parameter DIFFUSES (displacement ~ sqrt(T))
   instead of descending (~T). r -> 1 is a perfectly consistent gradient. r BELOW the floor means
   the gradient is oscillating or decaying (non-stationary), which is normal at the end of a
   cosine schedule.

2. THE eps DAMPING  sqrt(v) / (sqrt(v) + eps).
   1.0 means eps is irrelevant; 0.5 means the tier's Adam step is halved by the eps floor alone.
   Reported at p10/p50 together with the share of coordinates whose sqrt(v) sits below eps.

CAVEAT ON THE TIME POINT. v is an EMA with beta2=0.999, i.e. a ~1000-step memory. Read off the
FINAL checkpoint of an 1845-step epoch it therefore describes roughly the second half of training,
not the plateau. Gradients on the plateau are smaller than after the escape (the anchor's logged
grad_norm p50 is 0.144 on the plateau vs 1.557 late), so the eps damping measured here is a LOWER
BOUND on the damping during the plateau. To measure the plateau directly, run with
`save_steps` set inside it and point this script at that checkpoint.

Usage
-----
    python scripts/measure_adam_conditioning.py \
        outputs/saltq_cs170k_int2_g32_ep1-2bit-saltq/checkpoint-1845/optimizer.pt \
        --label "SALT-Q INT2 g32 anchor"

`--label` is free text. Group order matches src/trainer.py's `_make_saltq_trainer_cls`:
salient weights, scales, zero-points, other -- identify them by `initial_lr` and `numel`.
"""

import argparse
import math

import torch

BETA1, BETA2 = 0.9, 0.999
NOISE_FLOOR = math.sqrt((1 - BETA1) / (1 + BETA1))          # 0.2294


def rho_from_r(r: float) -> float:
    """Invert the coherence ratio to the per-coordinate gradient SNR. 0.0 == pure noise."""
    r2 = r * r
    if r2 <= NOISE_FLOOR ** 2:
        return 0.0
    if r2 >= 1.0:
        return float("inf")
    return math.sqrt((r2 - NOISE_FLOOR ** 2) / (1.0 - r2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("optimizer_pt")
    ap.add_argument("--label", default="")
    ap.add_argument("--eps", type=float, default=1e-8,
                    help="the eps the run actually used (HF's adam_epsilon default is 1e-8; a "
                         "per-group override in trainer.py would make this per-group too)")
    ap.add_argument("--subsample", type=int, default=20000,
                    help="values kept per tensor for the sqrt(v) percentiles; the percentiles are "
                         "stable well below this and it keeps a 200M-parameter group in RAM")
    args = ap.parse_args()

    # weights_only=False: this is our own checkpoint and it carries non-tensor scheduler scalars.
    sd = torch.load(args.optimizer_pt, map_location="cpu", weights_only=False)
    state = sd["state"]

    print(f"\n{'=' * 118}")
    print(f"{args.label or args.optimizer_pt}")
    print(f"{'=' * 118}")
    print(f"{'grp':>3} {'initial_lr':>11} {'numel':>13} {'r=|m|/sqrt(v)':>14} {'rho(SNR)':>9}"
          f" {'sqrt(v) p10':>12}{'p50':>12}{'p90':>12} {'%<eps':>7} {'damp p50':>9} {'damp p10':>9}",
          flush=True)

    for gi, g in enumerate(sd["param_groups"]):
        idxs = [i for i in g["params"] if i in state]
        if not idxs:
            continue
        eps = float(g.get("eps", args.eps))

        n = 0
        rs = []                       # per-tensor r, so one huge tensor cannot dominate the median
        sv_samples = []
        for i in idxs:
            st = state[i]
            m = st["exp_avg"].double()
            v = st["exp_avg_sq"].double()
            step = float(st["step"]) if "step" in st else 0.0
            if step > 0:                                     # undo Adam's bias correction
                m = m / (1 - BETA1 ** step)
                v = v / (1 - BETA2 ** step)
            sv = v.sqrt()
            # E|m| / sqrt(E v) within a tensor: scale-free, so tensors are comparable
            rs.append(m.abs().mean().item() / (sv.mean().item() + 1e-300))
            n += m.numel()
            flat = sv.flatten()
            sv_samples.append(flat[:: max(1, flat.numel() // args.subsample)].clone())
            del m, v, sv, flat

        rs.sort()
        r_med = rs[len(rs) // 2]
        allv = torch.cat(sv_samples)
        p10, p50, p90 = (torch.quantile(allv, q).item() for q in (0.10, 0.50, 0.90))
        below = (allv < eps).double().mean().item() * 100

        print(f"{gi:>3} {g.get('initial_lr', float('nan')):>11.3e} {n:>13,} "
              f"{r_med:>14.4f} {rho_from_r(r_med):>9.3f} "
              f"{p10:>12.3e}{p50:>12.3e}{p90:>12.3e} {below:>6.1f}% "
              f"{p50 / (p50 + eps):>9.3f} {p10 / (p10 + eps):>9.3f}", flush=True)

    print(f"\n  pure-noise floor r = {NOISE_FLOOR:.4f} (rho = 0, the parameter diffuses); "
          f"r -> 1 is a perfectly consistent gradient.")
    print(f"  damp = sqrt(v)/(sqrt(v)+eps): 1.0 means eps is irrelevant, 0.5 means the tier's "
          f"Adam step is halved by eps alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
