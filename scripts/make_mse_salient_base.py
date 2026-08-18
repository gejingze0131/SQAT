"""Derive a SALT-Q base whose SALIENT LSQ grid is MSE-optimal instead of min-max.

WHY. SALT-Q's salient slice is the one place in the pipeline that uses a plain min-max grid:
GPTQ pins those columns to the canonical RTN grid with no OBS compensation, and build_saltq_base
then discards that result and re-derives the LSQ init from the captured fp32 weights via
`init_lsq_scale_zp_asym` (= min-max). At INT2 there are only 4 levels and s = (wmax-wmin)/3, so a
single outlier in a group of 32 inflates the scale and collapses the rest of the group onto one or
two levels.

The latent weight is full-precision fp32 and demonstrably free to move (measured 0.504 grid steps
over a run), but it can only ever land on the 4 points that (s, z) define — and (s, z) barely move
during training (measured |Δs_S| = 2.79% relative). So whatever the init puts down is, in
practice, the grid the run finishes with. That makes the init a first-class hyper-parameter at
2 bits, which is what this script exists to change.

WHAT IT DOES. Copies an existing base tensor-for-tensor and rewrites ONLY `<layer>.s_s` /
`<layer>.z_s`, recomputed per (row, group) by minimizing the squared reconstruction error over a
symmetric shrink of the clipping range. The frozen int8 `codes` and the non-salient `s_n` / `z_n`
are copied bitwise, so a run against this base differs from one against the source base by exactly
the salient grid initialization — the codes are a one-shot discrete choice (AGENTS.md §6
invariant 7) and re-running GPTQ would re-roll them for no reason.

WHAT IT DOES NOT TOUCH. The deploy-equivalence invariant (`deployed == trained`, max|Δ| == 0) is
structurally unaffected: export reads whatever (s, z) the checkpoint carries, and training and
export use the same tensors regardless of how they were initialized. The property this DOES give
up is the M2 diagnostic "step-0 == GPTQ reconstruction" — by construction, since the whole point
is to start on a better grid than GPTQ's min-max pin. That test asserts formula agreement between
the asym LSQ+ path and the canonical affine path; it still holds for a min-max-initialized base
and should keep being run against one.

Usage:
    python scripts/make_mse_salient_base.py \
        --src outputs/saltq/saltq_base_2bit_g32 \
        --dst outputs/saltq/saltq_base_2bit_g32_mse
"""

import argparse
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SALTQ_BASE_FILENAME = "saltq_base.safetensors"
SALTQ_META_FILENAME = "saltq_meta.pt"


def mse_optimal_scale_zp(W: torch.Tensor, group_size: int, q_bits: int, n_grid: int = 40,
                         a_min: float = 0.40):
    """
    Per-(row, group) asymmetric affine grid minimizing ||dequant(quant(w)) - w||^2.

    Searches a symmetric shrink of the min-max range: [c - a*r, c + a*r] for a in [a_min, 1.0],
    where c and r are the group's midpoint and half-range. a = 1.0 reproduces min-max exactly, so
    the result is never worse than the current initialization.

    Returns (scale, zp), both [out, ng], matching init_lsq_scale_zp_asym's contract.
    """
    Qp = 2 ** q_bits - 1
    out_f, in_f = W.shape
    Wg = W.reshape(out_f, in_f // group_size, group_size)

    lo, hi = Wg.amin(-1), Wg.amax(-1)
    c, r = (hi + lo) / 2, (hi - lo) / 2

    best_err = None
    best_s = None
    best_z = None
    for a in torch.linspace(a_min, 1.0, n_grid):
        l = c - a * r
        h = c + a * r
        s = ((h - l) / max(Qp, 1)).clamp(min=1e-8)
        z = torch.round(-l / s).clamp(0, Qp)
        q = torch.clamp(torch.round(Wg / s.unsqueeze(-1)) + z.unsqueeze(-1), 0, Qp)
        err = (((q - z.unsqueeze(-1)) * s.unsqueeze(-1) - Wg) ** 2).sum(-1)
        if best_err is None:
            best_err, best_s, best_z = err, s, z
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_s = torch.where(take, s, best_s)
            best_z = torch.where(take, z, best_z)
    return best_s, best_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing SALT-Q base dir")
    ap.add_argument("--dst", required=True, help="new base dir to write")
    ap.add_argument("--n_grid", type=int, default=40)
    ap.add_argument("--a_min", type=float, default=0.40)
    args = ap.parse_args()

    src_file = os.path.join(args.src, SALTQ_BASE_FILENAME)
    meta = torch.load(os.path.join(args.src, SALTQ_META_FILENAME),
                      map_location="cpu", weights_only=False)
    gs, qb = int(meta["group_size"]), int(meta["q_bits"])
    if meta.get("symmetric", False):
        raise SystemExit("[MSE-init] symmetric bases are not handled (asym-only path).")
    print(f"[MSE-init] src={args.src}  INT{qb} g{gs}")

    os.makedirs(args.dst, exist_ok=True)
    tensors = {}
    n_slices = 0
    mm_err = mse_err = pw = 0.0

    with safe_open(src_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            # s_s / z_s are recomputed from w_s below; everything else is copied bitwise.
            if k.endswith(".s_s") or k.endswith(".z_s"):
                continue
            tensors[k] = t

        for k in keys:
            if not k.endswith(".w_s"):
                continue
            name = k[: -len(".w_s")]
            W = tensors[k].float()
            s_new, z_new = mse_optimal_scale_zp(W, gs, qb, args.n_grid, args.a_min)

            # Report the error the OLD (min-max) grid would have had, for the record.
            s_old = f.get_tensor(f"{name}.s_s").float()
            z_old = f.get_tensor(f"{name}.z_s").float()
            Wg = W.reshape(W.shape[0], -1, gs)
            Qp = 2 ** qb - 1
            for s, z, acc in ((s_old, z_old, "old"), (s_new, z_new, "new")):
                q = torch.clamp(torch.round(Wg / s.unsqueeze(-1)) + z.unsqueeze(-1), 0, Qp)
                e = (((q - z.unsqueeze(-1)) * s.unsqueeze(-1) - Wg) ** 2).sum().item()
                if acc == "old":
                    mm_err += e
                else:
                    mse_err += e
            pw += (Wg ** 2).sum().item()

            tensors[f"{name}.s_s"] = s_new.contiguous()
            tensors[f"{name}.z_s"] = z_new.contiguous()
            n_slices += 1
            if n_slices % 32 == 0:
                print(f"  [{n_slices}] {name}", flush=True)

    if n_slices == 0:
        raise SystemExit("[MSE-init] no .w_s tensors found — is this a train_salient=False base?")

    save_file(tensors, os.path.join(args.dst, SALTQ_BASE_FILENAME))

    meta["salient_init"] = "mse_optimal_clip"
    meta["salient_init_src_base"] = os.path.abspath(args.src)
    torch.save(meta, os.path.join(args.dst, SALTQ_META_FILENAME))
    for extra in os.listdir(args.src):
        if extra in (SALTQ_BASE_FILENAME, SALTQ_META_FILENAME):
            continue
        s = os.path.join(args.src, extra)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, extra))

    print(
        f"\n[MSE-init] rewrote the salient grid of {n_slices} slices -> {args.dst}\n"
        f"[MSE-init]   salient relF error   min-max {100 * (mm_err / pw) ** 0.5:.2f}%"
        f"   ->   MSE-optimal {100 * (mse_err / pw) ** 0.5:.2f}%"
        f"   (reduced {100 * (1 - (mse_err / mm_err) ** 0.5):.1f}%)\n"
        f"[MSE-init]   codes and non-salient (s_n, z_n) copied BITWISE from the source base."
    )


if __name__ == "__main__":
    main()
