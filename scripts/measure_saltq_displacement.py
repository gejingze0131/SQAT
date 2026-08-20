"""
Measure how far a finished SALT-Q run actually moved each trainable tier — in that tier's own
unit. This is the tool that turns "is the learning rate right?" from an argument into a number.

Why it exists: every SALT-Q lr is set against a different unit (weight units for the salient
weights and the LSQ scale, quantization LEVELS for the zero-points), and two of those units are
the grid step s(b) = range / (2^b - 1), which changes with the bit width. A rate that is correct
at INT3 is 2.33x too small at INT2. Both of the run-invalidating bugs so far were invisible in the
loss curve and obvious here:

  run1  INT3, integer z at a scale-sized lr   -> |Δz| = 0.000 levels  (non-salient tier dead) 32.9
  run2  INT3, the derived rates               -> |ΔW_S| = 0.58 steps, |Δz| = 0.039 levels    44.8
  run3  INT2, but still on INT3's rates       -> |ΔW_S| = 0.25 steps  (half the intended)    31.8

Targets to compare against (derivation in configs/saltq.yaml):
  |ΔW_S|   ~0.5 grid steps      below ~0.1 the salient codes never flip and the "full weight
                                freedom" tier is fictitious
  |Δs_S|   ~3% relative
  |Δz_N|   0.1-0.3 levels       past ~0.5 the shift stops being an error correction

Usage:
  python scripts/measure_saltq_displacement.py \
      --base outputs/saltq/saltq_base_2bit_g64 \
      --ckpt outputs/saltq-2bit-saltq/final \
      --salient_lr 2e-4 --scales_lr 1e-5 --zp_lr 1e-3
"""

import argparse
import glob
import os

import torch
from safetensors import safe_open

P = (0.1, 0.5, 0.9)


def _files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.safetensors")))
    return [path]


def _read(files, key):
    for f in files:
        with safe_open(f, "pt") as fh:
            if key in fh.keys():
                return fh.get_tensor(key).float()
    return None


def _modules(files):
    out = set()
    for f in files:
        with safe_open(f, "pt") as fh:
            out.update(k[:-4] for k in fh.keys() if k.endswith(".s_n"))
    return sorted(out)


def _q(t, ps=P):
    t = t.flatten().float()
    if t.numel() > 4_000_000:                      # torch.quantile caps out around 16M
        t = t[torch.randperm(t.numel())[:4_000_000]]
    return [torch.quantile(t, p).item() for p in ps]


def _fmt(vals, f="{:9.3f}"):
    return "  ".join(f.format(v) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="frozen-code base (holds the INITIAL values)")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint dir (final/)")
    ap.add_argument("--salient_lr", type=float, default=None)
    ap.add_argument("--scales_lr", type=float, default=None)
    ap.add_argument("--zp_lr", type=float, default=None)
    ap.add_argument("--layers", type=str, default="0,4,10,16,22,28,31",
                    help="which decoder layers to sample")
    args = ap.parse_args()

    base = _files(args.base)
    ckpt = _files(os.path.join(args.ckpt, "saltq_trainable.safetensors")
                  if os.path.isdir(args.ckpt) else args.ckpt)

    meta = torch.load(os.path.join(args.base, "saltq_meta.pt"),
                      map_location="cpu", weights_only=False)
    bits, gs = int(meta["q_bits"]), int(meta["group_size"])

    # If the run recorded its own lrs, they beat anything passed on the command line.
    cm_path = os.path.join(args.ckpt, "saltq_ckpt_meta.pt")
    lrs = {}
    if os.path.exists(cm_path):
        lrs = (torch.load(cm_path, map_location="cpu", weights_only=False)
               .get("learning_rates") or {})
    lr_w = lrs.get("salient_lr", args.salient_lr)
    lr_s = lrs.get("scales_lr", args.scales_lr)
    lr_z = lrs.get("zp_lr", args.zp_lr)

    want = tuple(f"layers.{i}." for i in args.layers.split(","))
    mods = [m for m in _modules(base) if any(w in m for w in want)]
    print(f"base  : {args.base}   (INT{bits} g{gs})")
    print(f"ckpt  : {args.ckpt}")
    print(f"lrs   : salient={lr_w}  scales={lr_s}  zp={lr_z}"
          f"{'  (from the checkpoint)' if lrs else '  (from the command line)'}")
    print(f"sample: {len(mods)} projections\n")

    acc = {k: [] for k in ("s_s", "d_w", "d_s", "d_zn", "d_zs")}
    for m in mods:
        s_s = _read(base, m + ".s_s")
        if s_s is not None:
            acc["s_s"].append(s_s.flatten())
        w0, w1 = _read(base, m + ".w_s"), _read(ckpt, m + ".weight_salient")
        if w0 is not None and w1 is not None:
            acc["d_w"].append((w1 - w0).abs().flatten()[::97])
        s0, s1 = s_s, _read(ckpt, m + ".lsq_w_scale")
        if s0 is not None and s1 is not None:
            acc["d_s"].append(((s1 - s0).abs() / s0.abs().clamp_min(1e-12)).flatten())
        z0, z1 = _read(base, m + ".z_n"), _read(ckpt, m + ".saltq_z")
        if z0 is not None and z1 is not None:
            acc["d_zn"].append((z1 - z0).abs().flatten()[::13])
        y0, y1 = _read(base, m + ".z_s"), _read(ckpt, m + ".lsq_w_zp")
        if y0 is not None and y1 is not None:
            acc["d_zs"].append((y1 - y0).abs().flatten())

    C = {k: torch.cat(v) for k, v in acc.items() if v}
    step = _q(C["s_s"])[1]
    print("                                        p10        p50        p90")
    print(f"  grid step s(INT{bits})  [weight]      {_fmt(_q(C['s_s']), '{:9.3e}')}")
    if "d_w" in C:
        d = _q(C["d_w"])
        print(f"  |ΔW_S|  [GRID STEPS]  target ~0.5  {_fmt([v / step for v in d])}")
        if lr_w:
            print(f"     -> c = |ΔW_S|/lr             {_fmt([v / lr_w for v in d], '{:9.1f}')}")
    if "d_s" in C:
        d = _q(C["d_s"])
        print(f"  |Δs_S|  [RELATIVE]    target ~3%   {_fmt([100 * v for v in d])}  %")
    if "d_zn" in C:
        d = _q(C["d_zn"])
        print(f"  |Δz_N|  [LEVELS]      target .1-.3 {_fmt(d)}")
        print(f"     -> in weight units              {_fmt([v * step for v in d], '{:9.3e}')}")
    if "d_zs" in C:
        print(f"  |Δz_S|  [LEVELS] salient LSQ zp    {_fmt(_q(C['d_zs']))}")

    if "d_w" in C:
        moved = _q(C["d_w"])[1] / step
        if moved < 0.2:
            print(f"\n  NOTE: the salient tier moved {moved:.2f} of a grid step at the MEDIAN. The "
                  f"~0.5 target and this\n  threshold are both MetaMath-derived and have not held "
                  f"on Commonsense-170k: there the best\n  score (81.46) came from exactly "
                  f"0.079 steps, while 0.273 -- inside the recommended band --\n  scored 79.04, "
                  f"the worst of the three. A median under 0.1 does not mean the tier is inert: "
                  f"p90\n  runs several times the median, and the salient LSQ scale moves the grid "
                  f"itself, shifting whole\n  groups of codes at once. Re-derive the target per "
                  f"dataset before acting on it.")
    if "d_zn" in C and _q(C["d_zn"])[1] < 1e-3:
        print("\n  WARNING: the zero-points did not move. That is the run1 failure — z is in "
              "quantization\n  LEVELS, so it needs an lr 2-3 orders above the scale lr, and "
              "continuous_z must be true.")


if __name__ == "__main__":
    main()
