#!/usr/bin/env python
"""
Why does SALT-Q lose to QA-LoRA when its parameterization strictly contains QA-LoRA's?

Both methods add the SAME KIND of term to a frozen quantized GEMM: a per-(output, group)
coefficient multiplying a pooled input.

  QA-LoRA   y += scaling * (B @ A) @ mean_g(x)        (B@A) is [out, G], rank <= r
  SALT-Q    y -= (z * s) @ sum_g(x)                   (z*s) is [out, G], FULL RANK

so SALT-Q's reachable set contains QA-LoRA's, and SALT-Q additionally trains real weights on the
salient columns. Expressiveness cannot be the explanation. This script measures what the two runs
actually DID with that freedom, in one shared unit.

Unit conversion matters and is easy to get wrong: QA-LoRA pools with a MEAN, SALT-Q with a SUM.
A coefficient c on sum_g(x) equals a coefficient c*group_size on mean_g(x). Everything below is
reported in mean-pooled units.

  python scripts/compare_qalora_saltq_capacity.py
"""

import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Hist:
    """Streaming quantiles/mean over tensors too large to concatenate.

    The previous version cat'ed every projection into one tensor and called kthvalue on it. At
    INT2 g32 that is ~200M zero-points per list and four lists; the process was OOM-killed, and
    because the call site piped through `tail` the pipeline still exited 0 with the buffered
    stdout lost — the script looked like it had succeeded and printed nothing. Accumulate
    instead: one fixed histogram per series, plus an exact running mean/max/count.
    """

    def __init__(self, lo=0.0, hi=1.0, nbins=20000):
        self.lo, self.hi, self.nb = float(lo), float(hi), int(nbins)
        self.h = torch.zeros(self.nb, dtype=torch.float64)
        self.n = 0
        self.total = 0.0
        self.mx = float("-inf")
        self.mn = float("inf")
        self.over = 0                      # mass above hi, so p99 cannot silently clip

    def add(self, t):
        t = t.detach().flatten().float()
        self.h += torch.histc(t.clamp(self.lo, self.hi), self.nb, self.lo, self.hi).double()
        self.n += t.numel()
        self.total += t.sum().item()
        self.mx = max(self.mx, t.max().item())
        self.mn = min(self.mn, t.min().item())
        self.over += int((t > self.hi).sum())
        return self

    def quantile(self, p):
        c = torch.cumsum(self.h, 0) / max(self.h.sum().item(), 1.0)
        i = int(torch.searchsorted(c, float(p)).item())
        i = min(i, self.nb - 1)
        return self.lo + (i + 1) * (self.hi - self.lo) / self.nb

    def mean(self):
        return self.total / max(self.n, 1)

    def frac_at_most(self, v):
        c = torch.cumsum(self.h, 0) / max(self.h.sum().item(), 1.0)
        i = min(int((v - self.lo) / (self.hi - self.lo) * self.nb), self.nb - 1)
        return float(c[max(i, 0)].item())


def q(t, p):
    return t.quantile(p) if isinstance(t, Hist) else \
        t.flatten().kthvalue(max(1, int(p * t.numel()))).values.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qalora_dir", default="outputs/qalora_cs170k_int3_ep1-3bit-qalora/final")
    ap.add_argument("--saltq_base", default="outputs/saltq_cs170k_int3_g64_ep1/saltq_base_3bit_g64")
    ap.add_argument("--saltq_ckpt", default="outputs/saltq_cs170k_int3_g64_ep1_nogbl-3bit-saltq/final")
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--max_projections", type=int, default=0,
                    help="evenly sample at most this many projections; 0 = all. Percentiles are "
                         "stable well before 224 layers and this keeps the run to seconds.")
    ap.add_argument("--bits", type=int, default=3,
                    help="sets the zero-point clamp [0, 2^bits - 1]; it is [0,3] at INT2 and "
                         "[0,7] at INT3, which is why the same |dz| means different things")
    args = ap.parse_args()
    GS = args.group_size

    cfg = json.load(open(os.path.join(args.qalora_dir, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"QA-LoRA  r={cfg['r']}  alpha={cfg['lora_alpha']}  scaling={scaling:.4f}  "
          f"dropout={cfg.get('lora_dropout')}")

    # ---- QA-LoRA: delta on mean-pooled input. B is zero-init, so final == delta. -------------
    qa = {}
    with safe_open(os.path.join(args.qalora_dir, "adapter_model.safetensors"), "pt") as f:
        keys = set(f.keys())
        for k in keys:
            if ".lora_A." not in k:
                continue
            kb = k.replace(".lora_A.", ".lora_B.")
            if kb not in keys:
                continue
            name = k.split(".lora_A.")[0].replace("base_model.model.", "")
            A = f.get_tensor(k).float()
            B = f.get_tensor(kb).float()
            qa[name] = (B @ A) * scaling                      # [out, G], mean-pooled units

    # ---- SALT-Q: delta on mean-pooled input = -(z_trained - z_base) * s * group_size ---------
    #
    # Everything below is streamed. Materialising z_base, s_base and z_trained together is ~2.4 GB
    # of fp32 at INT2 g32 (197M zero-points each); on a login node that swaps, and an earlier
    # version of this block sat for two hours with no output before it was killed. Keep only the
    # (file, key) address of every tensor, decide which projections to visit, and fetch each one
    # inside the loop.
    from src.qat_saltq import SALTQ_BASE_FILENAME
    base_pt = os.path.join(args.saltq_base, SALTQ_BASE_FILENAME)
    # z_s / s_s are the SALIENT slice's (scale, zero-point). They are normally irrelevant here --
    # under train_salient=true the salient columns are trainable REAL WEIGHTS and their z is not
    # what adapts. But a z-only run (train_salient=false) folds the salient slice into the
    # frozen-code pool, so its saltq_z spans EVERY group while the base's z_n spans only the
    # non-salient ones: 344 vs 340 for down_proj at g32 with group_k=128. Without the salient
    # halves the comparison raised "size of tensor a (344) must match tensor b (340)" and no
    # z-only checkpoint could be read at all.
    zb_at, sb_at, zs_at, ss_at = {}, {}, {}, {}
    with safe_open(base_pt, "pt") as f:
        for k in f.keys():
            if k.endswith(".z_n"):
                zb_at[k[: -len(".z_n")]] = k
            elif k.endswith(".s_n"):
                sb_at[k[: -len(".s_n")]] = k
            elif k.endswith(".z_s"):
                zs_at[k[: -len(".z_s")]] = k
            elif k.endswith(".s_s"):
                ss_at[k[: -len(".s_s")]] = k

    zt_at = {}
    for path in sorted(glob.glob(os.path.join(args.saltq_ckpt, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k.endswith(".saltq_z"):
                    zt_at[k[: -len(".saltq_z")].replace("base_model.model.", "")] = (path, k)

    print(f"\nprojections: QA-LoRA {len(qa)}, SALT-Q base {len(zb_at)}, "
          f"SALT-Q trained {len(zt_at)}")

    Qp = float(2 ** args.bits - 1)
    sq = Hist(0.0, 1.0); qav = Hist(0.0, 1.0)      # coefficients on the mean-pooled input
    dz = Hist(0.0, Qp); z0 = Hist(0.0, Qp)         # zero-point, in levels

    names = sorted(n for n in zt_at if n in zb_at and n in sb_at)
    if args.max_projections and len(names) > args.max_projections:
        step = max(1, len(names) // args.max_projections)
        names = names[::step][: args.max_projections]
        print(f"sampling {len(names)} of {len(zt_at)} projections (--max_projections); "
              f"percentiles are stable well before all 224")

    matched = 0
    with safe_open(base_pt, "pt") as fb:
        for i, name in enumerate(names):
            path, key = zt_at[name]
            with safe_open(path, "pt") as ft:
                z1 = ft.get_tensor(key).float()
            zbase = fb.get_tensor(zb_at[name]).float()
            sbase = fb.get_tensor(sb_at[name]).float()
            if z1.shape[1] != zbase.shape[1]:
                # z-only layout: the trained z covers the salient groups too. Rebuild the base as
                # [salient | non-salient] so the two line up group for group.
                n_extra = z1.shape[1] - zbase.shape[1]
                if n_extra <= 0 or name not in zs_at or name not in ss_at:
                    raise RuntimeError(
                        f"{name}: trained z has {z1.shape[1]} groups, base z_n has "
                        f"{zbase.shape[1]}, and no salient (z_s, s_s) is available to reconcile "
                        f"them. This is not the z-only layout; the base and checkpoint disagree.")
                zsal = fb.get_tensor(zs_at[name]).float()
                ssal = fb.get_tensor(ss_at[name]).float()
                if zsal.shape[1] != n_extra:
                    raise RuntimeError(
                        f"{name}: salient slice has {zsal.shape[1]} groups but the trained z has "
                        f"{n_extra} more than z_n; the checkpoint does not match this base.")
                zbase = torch.cat([zsal, zbase], dim=1)
                sbase = torch.cat([ssal, sbase], dim=1)
                del zsal, ssal
            d = z1 - zbase
            sq.add((d * sbase * GS).abs())
            dz.add(d.abs())
            z0.add(zbase)
            matched += 1
            del d, z1, zbase, sbase
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(names)}", flush=True)

    for _name, d in qa.items():
        qav.add(d.abs())
    print(f"matched SALT-Q projections: {matched}")

    print("\n=== |delta coefficient| on the MEAN-pooled input (the shared function space) ===")
    print(f"{'':22}{'p50':>12}{'p90':>12}{'p99':>12}{'max':>12}{'mean':>12}")
    for lab, t in (("QA-LoRA  (B@A)*sc", qav), ("SALT-Q   -dz*s*g", sq)):
        print(f"{lab:22}{q(t,.5):12.3e}{q(t,.9):12.3e}{q(t,.99):12.3e}"
              f"{t.mx:12.3e}{t.mean():12.3e}")
    r = qav.mean() / max(sq.mean(), 1e-30)
    print(f"\n  QA-LoRA moves the shared coefficient {r:.1f}x further than SALT-Q on average.")

    print("\n=== SALT-Q zero-point, in its own units ===")
    print(f"  |dz| levels      p50={q(dz,.5):.4f}  p90={q(dz,.9):.4f}  max={dz.mx:.4f}")
    print(f"  z_init          p50={q(z0,.5):.3f}  min={z0.mn:.3f}  max={z0.mx:.3f}")
    # The clamp is [0, 2^b - 1] and it MOVES with the bit width: [0,7] at INT3 but only [0,3] at
    # INT2, so the same |dz| is a much larger share of the available travel. Groups initialised
    # at an edge cannot move outward at all.
    at_lo = z0.frac_at_most(1e-6)
    at_hi = 1.0 - z0.frac_at_most(Qp - 1e-6)
    print(f"  fraction of groups initialised AT the clamp: low {100*at_lo:.2f}%  "
          f"high {100*at_hi:.2f}%   (clamp [0, {Qp:.0f}] at INT{args.bits})")
    print(f"  |dz| p50 as a share of the half-range {Qp/2:.1f}: {q(dz,.5)/(Qp/2)*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
