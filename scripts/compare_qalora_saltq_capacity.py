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


def q(t, p):
    return t.flatten().kthvalue(max(1, int(p * t.numel()))).values.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qalora_dir", default="outputs/qalora_cs170k_int3_ep1-3bit-qalora/final")
    ap.add_argument("--saltq_base", default="outputs/saltq_cs170k_int3_g64_ep1/saltq_base_3bit_g64")
    ap.add_argument("--saltq_ckpt", default="outputs/saltq_cs170k_int3_g64_ep1_nogbl-3bit-saltq/final")
    ap.add_argument("--group_size", type=int, default=64)
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
    from src.qat_saltq import SALTQ_BASE_FILENAME
    zb, sb = {}, {}
    with safe_open(os.path.join(args.saltq_base, SALTQ_BASE_FILENAME), "pt") as f:
        for k in f.keys():
            if k.endswith(".z_n"):
                zb[k[: -len(".z_n")]] = f.get_tensor(k).float()
            elif k.endswith(".s_n"):
                sb[k[: -len(".s_n")]] = f.get_tensor(k).float()
    zt = {}
    for p in sorted(glob.glob(os.path.join(args.saltq_ckpt, "*.safetensors"))):
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if k.endswith(".saltq_z"):
                    zt[k[: -len(".saltq_z")].replace("base_model.model.", "")] = f.get_tensor(k).float()

    print(f"\nprojections: QA-LoRA {len(qa)}, SALT-Q base {len(zb)}, SALT-Q trained {len(zt)}")

    qa_all, sq_all, sq_dz, z0_all = [], [], [], []
    matched = 0
    for name, z1 in zt.items():
        if name not in zb:
            continue
        dz = (z1 - zb[name])
        s = sb[name]
        sq_all.append((dz * s * GS).abs().flatten())
        sq_dz.append(dz.abs().flatten())
        z0_all.append(zb[name].flatten())
        matched += 1
    for name, d in qa.items():
        qa_all.append(d.abs().flatten())

    sq = torch.cat(sq_all); qav = torch.cat(qa_all)
    dz = torch.cat(sq_dz);  z0 = torch.cat(z0_all)
    print(f"matched SALT-Q projections: {matched}")

    print("\n=== |delta coefficient| on the MEAN-pooled input (the shared function space) ===")
    print(f"{'':22}{'p50':>12}{'p90':>12}{'p99':>12}{'max':>12}{'mean':>12}")
    for lab, t in (("QA-LoRA  (B@A)*sc", qav), ("SALT-Q   -dz*s*g", sq)):
        print(f"{lab:22}{q(t,.5):12.3e}{q(t,.9):12.3e}{q(t,.99):12.3e}"
              f"{t.max().item():12.3e}{t.mean().item():12.3e}")
    r = qav.mean().item() / max(sq.mean().item(), 1e-30)
    print(f"\n  QA-LoRA moves the shared coefficient {r:.1f}x further than SALT-Q on average.")

    print("\n=== SALT-Q zero-point, in its own units ===")
    print(f"  |dz| levels      p50={q(dz,.5):.4f}  p90={q(dz,.9):.4f}  max={dz.max().item():.4f}")
    print(f"  z_init          p50={q(z0,.5):.3f}  min={z0.min().item():.3f}  max={z0.max().item():.3f}")
    # The clamp is [Qn, Qp]; saturated groups cannot move outward at all.
    for lo, hi in ((0.0, 7.0),):
        at_lo = (z0 <= lo + 1e-6).float().mean().item()
        at_hi = (z0 >= hi - 1e-6).float().mean().item()
        print(f"  fraction of groups initialised AT the clamp: low {100*at_lo:.2f}%  "
              f"high {100*at_hi:.2f}%   (clamp [{lo:.0f}, {hi:.0f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
