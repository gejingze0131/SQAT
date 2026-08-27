#!/usr/bin/env python
"""Build a RANK-ORDERED permutation meta for the mixed-precision sweeps (not for training).

WHY. The permuted base's segment permutations put the top-group_k salient channels first and
the remaining channels in ORIGINAL INDEX ORDER (permute_common._build_segment_perm), and even the
top-k list is index-sorted (_select_bucket_from_mask returns sorted(selected)). The sweep
(export_mixed_precision_sweep.py) only overrides group_k, so for k > 128 it keeps "top-128 +
(k-128) arbitrary columns" in fp16 -- the flat 128..512 stretch of the INT3 curves, confirmed by
the teacher-forced probe (loss flat within 0.01 nats from k=256 on). This script rebuilds the
permutation with rank_order=True and group_k=2048 under the SAME fixed segmentation, so that for
every k <= 2048 the prefix [0:k) is the true top-k by the same saliency ranking (outliers first,
then score), and the k=128 SET coincides with the base's.

The SALT-Q base and every trained result are untouched: this meta lives in its own directory and
is consumed only via --perm_meta_dir of the sweep. The checkpoint weights it saves are the same
model permuted differently; only sqat_permute_meta.pt matters downstream.

  python scripts/build_rankperm_meta.py --config configs/saltq_cs170k_int3_g64_ep1_sal5e5.yaml \
      --base_meta_dir outputs/saltq_cs170k_int3_g64_ep1/permuted_fp16_base \
      --save_dir outputs/mixedprec_rankperm_k2048 --group_k 2048
"""
import argparse, os, sys, torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_calibration_data, build_data_collator
from src.model_loader import load_tokenizer
from src.permute_common import build_permuted_fp16_checkpoint, load_perm_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_meta_dir", required=True, help="the training base whose segmentation and top-k sets must be reproduced")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--group_k", type=int, default=2048)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    sp = cfg["qat"]["saltq"]
    base = load_perm_meta(a.base_meta_dir)
    boundary_sizes = [int(x) for x in base["boundary_sizes"]]
    base_k = int(base["group_k"])
    print(f"[rankperm] base segmentation {boundary_sizes}, base group_k {base_k}; building rank-ordered k={a.group_k}")

    tok = load_tokenizer(cfg)
    cal = load_calibration_data(cfg, tok)
    loader = DataLoader(cal, batch_size=cfg["training"]["per_device_eval_batch_size"],
                        collate_fn=build_data_collator(tok), shuffle=False)
    meta = build_permuted_fp16_checkpoint(
        model_name=cfg["model"]["name"], tokenizer=tok, calibration_dataloader=loader,
        boundary_sizes=boundary_sizes,                       # FIXED: no autoseg re-split at this k
        save_dir=a.save_dir, target_modules=cfg["lora"].get("target_modules"),
        group_k=a.group_k, group_size=int(cfg["qat"].get("group_size", 128)),
        top_k_ratio=float(sp.get("top_k_ratio", 0.01)),
        outlier_log_sigma=float(sp.get("outlier_log_sigma", 3.0)),
        down_outlier_log_sigma=float(sp.get("down_outlier_log_sigma", sp.get("outlier_log_sigma", 3.0))),
        dtype=getattr(torch, cfg["model"]["dtype"]),
        awq_alpha=(sp.get("awq_scale", {}) or {}).get("alpha", 0.5),
        awq_max=(sp.get("awq_scale", {}) or {}).get("max", 2.0),
        max_segments=int(sp.get("max_segments", 4)),
        fold_awq=False, reorder_salient=False,
        rank_order=True,
        # The base was built by autoseg (legacy_topk_ratio_mode False = sigma outliers). With a
        # manual boundary_sizes the builder would default to the legacy top_k_ratio selector,
        # whose outlier sets differ -> different "outliers first" order -> different top-128 in
        # segments with >128 outliers (measured: segments 1-3 mismatched). Force the sigma path.
        legacy_topk_ratio=bool(base.get("legacy_topk_ratio_mode", False)),
    )
    meta = meta.get("model", meta)

    # ---- the pre-registered checks: same segmentation, same top-128 SETS, nested prefixes -----
    assert [int(x) for x in meta["boundary_sizes"]] == boundary_sizes, (meta["boundary_sizes"], boundary_sizes)
    ok = True
    for seg, p_new in meta["segment_perms"].items():
        p_new = list(p_new.tolist() if hasattr(p_new, "tolist") else p_new)
        p_old = base["segment_perms"][seg]; p_old = list(p_old.tolist() if hasattr(p_old, "tolist") else p_old)
        same = set(p_new[:base_k]) == set(p_old[:base_k])
        ok &= same
        print(f"[rankperm] segment {seg}: top-{base_k} set == base: {same}; first 6 ranked = {p_new[:6]}; "
              f"positions {base_k}..{base_k+5} = {p_new[base_k:base_k+6]}")
    for key, p_new in list(meta["block_internal_perms"].items())[:3]:
        p_old = base["block_internal_perms"][key]
        p_new = list(p_new.tolist() if hasattr(p_new, "tolist") else p_new); p_old = list(p_old.tolist() if hasattr(p_old, "tolist") else p_old)
        print(f"[rankperm] {key}: top-{base_k} set == base: {set(p_new[:base_k]) == set(p_old[:base_k])}")
    if not ok:
        print("[rankperm] ERROR: a segment's top-k set differs from the base -- the sweep would not be matched at k=128", file=sys.stderr)
        return 3
    print(f"[rankperm] OK -> {a.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
