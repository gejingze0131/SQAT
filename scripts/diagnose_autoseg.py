"""Why does the auto-segmentation run start at loss 4.262 when the legacy one starts at 2.401?

Two questions, answered separately, because they have different failure modes.

(1) IS THE PERMUTED BASE STILL THE SAME FUNCTION?
    The permutation + boundary gathers are supposed to be an exact rewrite of the original model.
    Auto segmentation raises the number of segments from 2 to 4, i.e. from 1 runtime boundary
    gather to 3, so there is more machinery to get wrong. This loads the original fp16 model and
    the saved permuted base, registers the gathers from the saved meta, and compares logits on
    identical inputs. If this fails, nothing downstream is meaningful.

    Prior: probably PASSES. The all-on run uses the SAME segmentation and the same 3 gathers and
    trained normally (step-10 loss 2.361, final 38.82), so a broken gather would have shown there
    too. But "probably" is not a measurement.

(2) IS THE SALIENT SLICE HARDER TO QUANTIZE UNDER AUTO SEGMENTATION?
    This is the hypothesis the loss numbers actually point at. Auto segmentation gives segment 0 a
    group_k of 256 covering 229 outliers, against a flat 128 before, so the salient slice now spans
    channels of far more varied magnitude. The slice is quantized with a per-(row, 32-column)
    min-max grid, and min-max is set by the widest element in the group — so mixing large and small
    columns in one group inflates the scale for all of them. Reordering the block by magnitude is
    exactly what would fix that, and the all-on run (auto segmentation WITH reordering) does start
    at a normal loss. This measures the salient reconstruction error of each saved base directly.

Run on a compute node: it loads two 7B models for (1).
"""

import argparse
import glob
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.permute_common import register_boundary_gathers_from_meta  # noqa: E402


def check_equivalence(orig_name, permuted_dir, n_tok=32, seed=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1] functional equivalence: {permuted_dir}\n    vs original {orig_name}")
    meta = torch.load(os.path.join(permuted_dir, "sqat_permute_meta.pt"),
                      map_location="cpu", weights_only=False)
    print(f"    segments={len(meta.get('boundary_sizes') or [])} "
          f"boundary_gathers={len(meta['boundary_perms'])} "
          f"segment_group_ks={meta.get('segment_group_ks')}")

    tok = AutoTokenizer.from_pretrained(orig_name, use_fast=True)
    torch.manual_seed(seed)
    ids = torch.randint(0, 30000, (1, n_tok))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m0 = AutoModelForCausalLM.from_pretrained(orig_name, torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True).to(dev).eval()
    with torch.no_grad():
        y0 = m0(ids.to(dev)).logits.float().cpu()
    del m0
    torch.cuda.empty_cache()

    m1 = AutoModelForCausalLM.from_pretrained(permuted_dir, torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True).to(dev).eval()
    hooks = register_boundary_gathers_from_meta(m1, meta)
    print(f"    registered {len(hooks)} boundary gathers")
    with torch.no_grad():
        y1 = m1(ids.to(dev)).logits.float().cpu()
    del m1
    torch.cuda.empty_cache()

    amax = (y1 - y0).abs().max().item()
    rel = amax / max(y0.abs().max().item(), 1e-12)
    # Same argmax matters more than raw logit distance for a fp16 rewrite.
    agree = (y0.argmax(-1) == y1.argmax(-1)).float().mean().item()
    print(f"    max|Δlogits|={amax:.4e}  relative={rel:.4e}  argmax agreement={agree*100:.2f}%")
    return rel, agree


def salient_recon_error(base_dir, n_layers=16):
    """Relative Frobenius error of the salient slice under its own stored min-max grid."""
    meta = torch.load(os.path.join(base_dir, "saltq_meta.pt"),
                      map_location="cpu", weights_only=False)
    qb, gs = int(meta["q_bits"]), int(meta["group_size"])
    Qp = 2 ** qb - 1
    files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    tot_e = tot_p = 0.0
    per = []
    with safe_open(files[0], framework="pt", device="cpu") as f:
        ws_keys = [k for k in f.keys() if k.endswith(".w_s")]
        sel = ws_keys[:: max(1, len(ws_keys) // n_layers)][:n_layers]
        for k in sel:
            name = k[: -len(".w_s")]
            W = f.get_tensor(k).float()
            s = f.get_tensor(f"{name}.s_s").float()
            z = f.get_tensor(f"{name}.z_s").float()
            Wg = W.reshape(W.shape[0], -1, gs)
            q = torch.clamp(torch.round(Wg / s.unsqueeze(-1)) + z.unsqueeze(-1), 0, Qp)
            e = (((q - z.unsqueeze(-1)) * s.unsqueeze(-1) - Wg) ** 2).sum().item()
            p = (Wg ** 2).sum().item()
            tot_e += e
            tot_p += p
            # within-group spread, the thing a shared min-max scale actually pays for
            spread = ((Wg.amax(-1) - Wg.amin(-1)) / Wg.abs().mean(-1).clamp(min=1e-12)).median()
            per.append((name.replace("model.layers.", "L"), W.shape[1],
                        (e / p) ** 0.5, spread.item()))
    return (tot_e / tot_p) ** 0.5, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--permuted", default="outputs/saltq_autoseg/permuted_fp16_base")
    ap.add_argument("--base_new", default="outputs/saltq_autoseg/saltq_base_2bit_g32")
    ap.add_argument("--base_ref", default="outputs/saltq/saltq_base_2bit_g32")
    ap.add_argument("--skip_equiv", action="store_true")
    args = ap.parse_args()

    if not args.skip_equiv:
        check_equivalence(args.orig, args.permuted)

    print("\n[2] salient-slice reconstruction under the stored min-max grid")
    for tag, d in (("legacy k=128 (40.9 run)", args.base_ref),
                   ("autoseg k=96..320", args.base_new)):
        if not os.path.isdir(d):
            print(f"    {tag}: MISSING {d}")
            continue
        agg, per = salient_recon_error(d)
        widths = sorted({w for _, w, _, _ in per})
        print(f"    {tag:26s} aggregate relF = {agg*100:6.2f}%   group_k seen={widths}")
        worst = sorted(per, key=lambda r: -r[2])[:4]
        for n, w, e, sp in worst:
            print(f"        worst: {n:32s} gk={w:4d} relF={e*100:6.2f}% spread={sp:6.2f}")


if __name__ == "__main__":
    main()
