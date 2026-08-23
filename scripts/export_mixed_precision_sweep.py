#!/usr/bin/env python
"""
Mixed-precision salient ablation: fp16 salient slice + GPTQ non-salient, swept over group_k.
NO TRAINING. This is the method SALT-Q claims to replace.

WHAT IT ANSWERS. SALT-Q's thesis is that TRAINING the salient slice (plus trainable (s,z) on the
frozen non-salient codes) can stand in for the usual mixed-precision trick of just keeping the
salient columns in fp16. That trick has never been measured on this pipeline, so "SALT-Q is as
good as mixed precision at a fraction of the bits" has been an assertion, not a result. This
script measures it, at the same segmentation, the same salient channels, the same GPTQ code path.

CONSTRUCTION. Start from the QLoRA-finetuned MERGED fp16 checkpoint (the 82.42 upper bound), so
the task adaptation is already in the weights and the only variable left is how the weights are
deployed. Then:

  1. Apply the SAME three equivalence transforms SALT-Q's permuted base uses, replaying the
     permutations SAVED in that base's sqat_permute_meta.pt rather than recalibrating:
         apply_segment_permutation_fp32     (residual-stream P_k)
         apply_block_internal_permutations_fp32  (MLP P4_l)
         apply_hadamard_rotation_fp32       (per-head H; deterministic, carries no state)
     Replaying rather than recomputing is the point: the salient channels must be the ones SALT-Q
     protects, not the ones this particular checkpoint would nominate. The recomputed boundary
     perms are asserted equal to the saved ones, which fails loudly if the replay drifted.
  2. Override every group_k in the meta to the swept value.
  3. gptq_quantize_model_sequential(..., keep_salient_fp16=True): columns [0:group_k) are left at
     full precision and the rest is GPTQ'd at INT-b g-s with the independent non-salient
     sub-Hessian -- byte-for-byte the code path that builds SALT-Q's frozen base and QA-LoRA's
     base, with the single flag flipped.
  4. Save a dense fp16 checkpoint + the meta, so runs/eval_vllm.sh folds the boundary gathers and
     scores it exactly like every other row of the table.

awq_scales is deliberately None. It is what SALT-Q's own base build passes, and with
keep_salient_fp16 the un-baking step at the end of gptq_quantize_model_sequential would divide
the RESTORED fp16 salient columns by S -- corrupting the very slice the ablation is about.

READ THE CURVE IN EFFECTIVE BITS, NOT IN group_k. A point that keeps 38% of the target weights in
fp16 is an 8-bit model; comparing its accuracy to SALT-Q's 3.0 is only meaningful once both are
on the same axis. The script prints the fp16 share and the effective bit width for each point.

  python scripts/export_mixed_precision_sweep.py \
      --model_path outputs/qlora_none_cs170k_int3_ep1-3bit-none-merged-eval \
      --perm_meta_dir outputs/saltq_cs170k_int3_g64_ep1/permuted_fp16_base \
      --config configs/saltq_cs170k_int3_g64_ep1_sal5e5.yaml \
      --bits 3 --group_size 64 \
      --group_k 64 128 256 512 1024 2048 \
      --output_root outputs/mixedprec_int3_g64
"""

import argparse
import copy
import gc
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from transformers import AutoModelForCausalLM

from src.data import build_data_collator, load_calibration_data
from src.model_loader import load_tokenizer
from src.permute_common import (
    apply_block_internal_permutations_fp32,
    apply_hadamard_rotation_fp32,
    apply_segment_permutation_fp32,
    load_perm_meta,
)


def _replay_permutation(model, meta, device):
    """Re-apply the saved permutation to a fresh dense checkpoint, in the builder's order."""
    segment_perms = {int(k): list(v) for k, v in meta["segment_perms"].items()}
    boundary_sizes = [int(x) for x in meta["boundary_sizes"]]
    internal = {}
    for key, v in meta["block_internal_perms"].items():
        layer_str, term = key.split("_", 1)
        internal[(int(layer_str), term)] = list(v)

    boundary_perms = apply_segment_permutation_fp32(model, segment_perms, boundary_sizes)
    apply_block_internal_permutations_fp32(model, internal)

    cfgm = model.config
    head_dim = cfgm.hidden_size // cfgm.num_attention_heads
    apply_hadamard_rotation_fp32(model, cfgm.num_hidden_layers,
                                 cfgm.num_key_value_heads, head_dim)

    # The replay is only valid if it reproduces the deployment basis the meta describes. If these
    # disagree, the boundary gathers stored in the meta no longer match the weights and the
    # exported model would be silently wrong rather than broken.
    saved = meta["boundary_perms"]
    assert len(saved) == len(boundary_perms), (
        f"replayed {len(boundary_perms)} boundary perms, meta has {len(saved)}")
    for i, (a, b) in enumerate(zip(saved, boundary_perms)):
        assert torch.equal(a.cpu(), b.cpu()), f"boundary perm {i} differs from the saved base"
    return boundary_perms


def _meta_with_group_k(meta, group_k):
    m = copy.deepcopy(meta)
    n_seg = len(m["segment_group_ks"])
    n_lay = len(m["layer_group_ks"])
    m["group_k"] = int(group_k)
    m["fixed_group_k"] = int(group_k)
    m["segment_group_ks"] = [int(group_k)] * n_seg
    m["layer_group_ks"] = [int(group_k)] * n_lay
    m["down_layer_group_ks"] = [int(group_k)] * len(m["down_layer_group_ks"])
    assert n_lay > 0 and n_seg > 0
    return m


def _cost(model_cfg, targets, group_k, bits):
    """fp16 share of the TARGET weights and the resulting effective bit width."""
    h, i = model_cfg.hidden_size, model_cfg.intermediate_size
    shapes = {"q_proj": (h, h), "k_proj": (h, h), "v_proj": (h, h), "o_proj": (h, h),
              "gate_proj": (i, h), "up_proj": (i, h), "down_proj": (h, i)}
    total = fp16 = 0
    for t in targets:
        out_f, in_f = shapes[t]
        total += out_f * in_f
        if t != "o_proj":                       # o_proj always has group_k = 0
            fp16 += out_f * min(int(group_k), in_f)
    share = fp16 / total
    return share, share * 16.0 + (1.0 - share) * bits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="merged dense fp16 checkpoint (LoRA already folded, NO perm meta)")
    ap.add_argument("--perm_meta_dir", required=True,
                    help="the SALT-Q permuted_fp16_base whose permutation is replayed")
    ap.add_argument("--config", required=True, help="calibration data, target modules, defaults")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--group_k", type=int, nargs="+", required=True)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.model_path, "sqat_permute_meta.pt")):
        print(f"ERROR: {args.model_path} is already permuted. This script permutes its input; "
              f"pass the un-permuted merged checkpoint.", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(open(args.config))
    bits = args.bits or int(cfg["model"]["quant_bits"])
    group_size = args.group_size or int(cfg["qat"].get("group_size", 128))
    symmetric = bool(cfg["qat"].get("symmetric", False))
    targets = list(cfg["lora"]["target_modules"])
    dtype = getattr(torch, cfg["model"]["dtype"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for k in args.group_k:
        if k % group_size:
            print(f"ERROR: group_k={k} is not a multiple of group_size={group_size}. A quant "
                  f"group would straddle the salient boundary and the scale/zp layout has no way "
                  f"to represent that (src/gptq.py asserts it).", file=sys.stderr)
            return 2

    meta = load_perm_meta(args.perm_meta_dir)
    print(f"[MixedPrec] permutation replayed from {args.perm_meta_dir}: "
          f"boundary_sizes={meta['boundary_sizes']}, base group_k={meta['group_k']}")

    tokenizer = load_tokenizer(cfg, name=args.model_path)
    cal = load_calibration_data(cfg, tokenizer)
    cal_loader = DataLoader(cal, batch_size=args.batch_size,
                            collate_fn=build_data_collator(tokenizer), shuffle=False)

    from src.gptq import gptq_quantize_model_sequential

    for k in args.group_k:
        out_dir = os.path.join(args.output_root, f"k{k}")
        if os.path.exists(os.path.join(out_dir, "config.json")):
            print(f"[MixedPrec] k={k}: already at {out_dir} — skipping")
            continue

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
        model.to(device).eval()
        _replay_permutation(model, meta, device)

        share, eff = _cost(model.config, targets, k, bits)
        print(f"[MixedPrec] k={k}: fp16 share of target weights = {share*100:.2f}%, "
              f"effective bits = {eff:.2f}")

        ref = {n: p.detach().float().cpu().clone()
               for n, p in model.named_parameters()
               if n.endswith(".weight") and n.split(".")[-2] in targets}

        gptq_quantize_model_sequential(
            model, cal_loader, targets,
            perm_group_k=k,
            group_size=group_size,
            q_bits=bits,
            symmetric=symmetric,
            device=device,
            perm_meta=_meta_with_group_k(meta, k),
            percdamp=args.percdamp,
            blocksize=args.blocksize,
            nsamples=args.nsamples,
            awq_scales=None,          # see the module docstring — NOT optional here
            keep_salient_fp16=True,   # the whole point of this ablation
        )

        # Two errors, because one of them must be ~0 and the other must not. The salient slice is
        # supposed to come back untouched; the non-salient block is supposed to carry the INT-b
        # error. A single aggregate number would hide either failure.
        sal_err, non_err = [], []
        for n, p in model.named_parameters():
            if n not in ref:
                continue
            term = n.split(".")[-2]
            gk = 0 if term == "o_proj" else min(k, ref[n].shape[1])
            r, q = ref[n], p.detach().float().cpu()
            if gk:
                sal_err.append((r[:, :gk] - q[:, :gk]).abs().max().item())
            nb_r, nb_q = r[:, gk:], q[:, gk:]
            non_err.append((nb_r - nb_q).norm().item() / max(nb_r.norm().item(), 1e-12))
        non_err.sort()
        print(f"[MixedPrec] k={k}: salient max-abs drift = {max(sal_err):.3e} "
              f"(must be 0) | non-salient rel err p50 = {non_err[len(non_err)//2]:.4f}")
        if max(sal_err) > 1e-6:
            raise RuntimeError("the fp16 salient slice was modified — keep_salient_fp16 is not "
                               "doing what this ablation assumes")
        if non_err[len(non_err) // 2] < 1e-3:
            raise RuntimeError("non-salient median relative error is ~0 — nothing was quantized")

        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        torch.save(_meta_with_group_k(meta, k), os.path.join(out_dir, "sqat_permute_meta.pt"))
        torch.save({"q_bits": bits, "group_size": group_size, "symmetric": symmetric,
                    "group_k": k, "keep_salient_fp16": True,
                    "fp16_share": share, "effective_bits": eff,
                    "source_model": os.path.abspath(args.model_path),
                    "perm_meta_dir": os.path.abspath(args.perm_meta_dir)},
                   os.path.join(out_dir, "mixedprec_meta.pt"))
        print(f"[MixedPrec] k={k}: saved -> {out_dir}")

        del model, ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
