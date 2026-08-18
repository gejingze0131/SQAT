#!/usr/bin/env python
"""
Main training entry point.

Usage:
  # Standard QLoRA (no QAT)
  accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml

  # Finetune on MetaMath
  accelerate launch --num_processes 4 scripts/train.py \
    --config configs/default.yaml \
    --train_dataset metamath \
    --prompt_template metamath

  # With SQAT
  accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml --qat_mode sqat

  # With QA-LoRA (asymmetric only)
  accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml --qat_mode qalora --asymmetric

  # With Full QAT
  accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml --qat_mode full

  # Export only (from existing checkpoint)
  python scripts/train.py --export_only --checkpoint_dir outputs/qlora-4bit-none/checkpoint-600

  # Override bit width
  accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml --bits 3
"""

import os
import sys
import argparse
import hashlib
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
from accelerate import Accelerator

from src.model_loader import load_model_and_tokenizer, load_tokenizer
from src.data import (
    build_data_collator,
    load_calibration_data,
    load_dataset_for_training,
)
from src.trainer import build_trainer
from src.qat_base import get_qat_handler
from src.export import export_merged_only, merge_and_export, export_adapter_only


def _permuted_base_reusable(permuted_dir: str, cfg: dict, sp_cfg: dict):
    """Return the reason a permuted fp16 base can be reused as-is, or None if it must be rebuilt.

    The permuted base depends ONLY on the saliency/segmentation config — it is built BEFORE any
    quantization and carries no bit-width. So across a bit sweep (3bit -> 2bit) rebuilding it is
    not just an hour of calibration plus a 13 GB write: the calibration pass would be re-run, and
    any drift in the chosen segmentation would silently make the two runs differ by more than the
    one variable under test. Reuse it whenever every field that shaped it matches.
    """
    meta_pt = os.path.join(permuted_dir, "sqat_permute_meta.pt")
    if not (os.path.isdir(permuted_dir) and os.path.exists(meta_pt)):
        return None
    try:
        meta = torch.load(meta_pt, map_location="cpu", weights_only=False)
    except Exception:
        return None

    awq = (sp_cfg.get("awq_scale", {}) or {})
    want = {
        "boundary_sizes":         sp_cfg.get("boundary_sizes"),
        "fixed_group_k":          sp_cfg.get("group_k"),
        "group_size":             cfg["qat"].get("group_size", 128),
        "top_k_ratio":            sp_cfg.get("top_k_ratio", 0.01),
        "outlier_log_sigma":      sp_cfg.get("outlier_log_sigma", 3.0),
        "down_outlier_log_sigma": sp_cfg.get(
            "down_outlier_log_sigma", sp_cfg.get("outlier_log_sigma", 3.0)),
        "awq_alpha":              awq.get("alpha", 0.5),
        "awq_max":                awq.get("max", 2.0),
    }
    for key, wanted in want.items():
        have = meta.get(key)
        if isinstance(wanted, (list, tuple)) or isinstance(have, (list, tuple)):
            if list(have or []) != list(wanted or []):
                return None
        elif wanted is None or have is None:
            if wanted is not have:
                return None
        elif float(have) != float(wanted):
            return None

    # Boolean transforms baked into the saved weights. Compared separately (and defaulted to
    # False) so that a base written before these existed still matches a config that wants
    # neither — the numeric path above would treat a missing key as a mismatch and force a
    # needless 13 GB rebuild.
    for key, wanted in (
        ("awq_folded", bool((sp_cfg.get("awq_scale", {}) or {}).get("fold", False))),
        ("salient_reordered", bool(sp_cfg.get("reorder_salient", False))),
    ):
        if bool(meta.get(key, False)) != wanted:
            return None
    return (f"group_k={meta.get('group_k')} boundary_sizes={meta.get('boundary_sizes')} "
            f"group_size={meta.get('group_size')}")


def _saltq_base_dir_for(cfg: dict, explicit: str | None) -> str:
    """Where this run's FROZEN GPTQ codes live.

    The codes ARE bit-width- and group-size-specific, so the path has to be too: the original
    default (<output_dir>/saltq_base) meant that starting a 2-bit run would silently overwrite
    the 3-bit codes that the already-trained 3-bit checkpoint points at, making that checkpoint
    unexportable. New default: <output_dir>/saltq_base_{bits}bit_g{group_size}, falling back to
    the legacy un-suffixed dir when it exists AND matches this config (so the 3-bit run keeps
    finding its own base).
    """
    if explicit:
        return explicit
    out = cfg["training"]["output_dir"]
    bits = int(cfg["model"]["quant_bits"])
    gs = int(cfg["qat"].get("group_size", 128))
    legacy = os.path.join(out, "saltq_base")
    if _saltq_base_matches(legacy, cfg) is not None:
        return legacy
    return os.path.join(out, f"saltq_base_{bits}bit_g{gs}")


def _permutation_fingerprint(perm_meta: dict) -> str:
    """Stable hash of every index array the frozen GPTQ codes are addressed by.

    The codes are integers at FIXED COLUMN POSITIONS of the permuted weight. They mean nothing
    except relative to the permutation that produced those positions, and nothing in the file
    format says which one that was. Reusing a frozen-code base across a REBUILT permutation is
    therefore not a degradation, it is a scramble — and a silent one: the histograms stay
    healthy, training converges, and only the score is wrong.

    That is not hypothetical. The INT3 g64 commonsense run rebuilt the permuted base (the
    calibration data had changed) while `--saltq_base_dir` pointed at codes from the previous
    permutation. Reconstruction error came out at 114% (worse than emitting zeros), the untrained
    model sat at loss 10.6 against ln(32000) = 10.37, and the whole run was training its way back
    out of uniform noise.
    """
    def _norm(v):
        # A tensor and the list it round-trips to MUST hash the same: perm_meta stores
        # boundary_perms as tensors and segment_perms as lists, and an export/reload can turn one
        # into the other. Note `v or []` is unusable here — bool() of a multi-element tensor
        # raises, which is what the first version of this function did.
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().flatten().tolist()
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple)) else v

    h = hashlib.sha256()
    for key in ("boundary_sizes", "layer_group_ks", "down_layer_group_ks", "segment_group_ks"):
        h.update(f"{key}={_norm(perm_meta.get(key))}".encode())
    for key in ("segment_perms", "block_internal_perms"):
        for name, perm in sorted((perm_meta.get(key) or {}).items(), key=lambda kv: str(kv[0])):
            h.update(f"{key}[{name}]={_norm(perm)}".encode())
    return h.hexdigest()[:16]


def _saltq_base_permutation_matches(base_dir: str, perm_meta: dict) -> bool:
    """True if base_dir's codes were GPTQ'd against exactly this permutation."""
    from src.qat_saltq import SALTQ_META_FILENAME
    meta_pt = os.path.join(base_dir, SALTQ_META_FILENAME)
    if not os.path.exists(meta_pt):
        return False
    try:
        stored = torch.load(meta_pt, map_location="cpu", weights_only=False).get("perm_meta")
    except Exception:
        return False
    if not stored:
        return False
    return _permutation_fingerprint(stored) == _permutation_fingerprint(perm_meta)


def _saltq_base_matches(base_dir: str, cfg: dict):
    """Return a description of an existing frozen-code base if it matches cfg, else None.

    A mismatch here is not recoverable at runtime — the codes are a one-shot discrete choice, so
    reusing 3-bit codes under a 2-bit config would produce a model whose (s, z) address a grid
    that does not exist.
    """
    from src.qat_saltq import SALTQ_META_FILENAME
    meta_pt = os.path.join(base_dir, SALTQ_META_FILENAME)
    if not (os.path.isdir(base_dir) and os.path.exists(meta_pt)):
        return None
    try:
        meta = torch.load(meta_pt, map_location="cpu", weights_only=False)
    except Exception:
        return None
    sq_cfg = cfg["qat"].get("saltq", {}) or {}
    if (int(meta.get("q_bits", -1)) != int(cfg["model"]["quant_bits"])
            or int(meta.get("group_size", -1)) != int(cfg["qat"].get("group_size", 128))
            or bool(meta.get("symmetric", False)) != bool(cfg["qat"].get("symmetric", False))
            or bool(meta.get("train_salient", True)) != bool(sq_cfg.get("train_salient", True))):
        return None
    return (f"INT{meta.get('q_bits')} g{meta.get('group_size')} "
            f"{'sym' if meta.get('symmetric') else 'asym'}"
            f"{'' if meta.get('train_salient', True) else ' train_salient=False'}")


def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if overrides.get("qat_mode"):
        cfg["qat"]["mode"] = overrides["qat_mode"]
    if overrides.get("bits"):
        cfg["model"]["quant_bits"] = overrides["bits"]
    if overrides.get("symmetric") is not None:
        cfg["qat"]["symmetric"] = overrides["symmetric"]
    if overrides.get("output_dir"):
        cfg["training"]["output_dir"] = overrides["output_dir"]
    if overrides.get("epochs"):
        cfg["training"]["num_epochs"] = overrides["epochs"]
    if overrides.get("lr"):
        cfg["training"]["learning_rate"] = overrides["lr"]
    if overrides.get("lora_rank"):
        cfg["lora"]["rank"] = overrides["lora_rank"]
    if overrides.get("top_k_ratio"):
        cfg["qat"]["sqat"]["top_k_ratio"] = overrides["top_k_ratio"]
    if overrides.get("input_top_k") is not None:
        cfg["qat"]["sqat"]["input_top_k"] = overrides["input_top_k"]
    if overrides.get("output_top_k") is not None:
        cfg["qat"]["sqat"]["output_top_k"] = overrides["output_top_k"]
    if overrides.get("salient_gain_alpha") is not None:
        cfg["qat"]["sqat"]["salient_gain_alpha"] = overrides["salient_gain_alpha"]
    if overrides.get("salient_gain_max") is not None:
        cfg["qat"]["sqat"]["salient_gain_max"] = overrides["salient_gain_max"]
    if overrides.get("awq_scale") is not None:
        cfg["qat"].setdefault("sqat_permute", {}).setdefault("awq_scale", {})["enabled"] = \
            overrides["awq_scale"]
    if overrides.get("gptq_nonsalient") is not None:
        cfg["qat"].setdefault("sqat_permute", {}).setdefault("gptq", {})["enabled"] = \
            overrides["gptq_nonsalient"]
    if overrides.get("enable_lsq") is not None:
        cfg["qat"].setdefault("lsq", {})["enabled"] = overrides["enable_lsq"]
    # An explicit lr on the command line has to beat the per-bit table in the yaml, so it clears
    # the corresponding *_by_bits map — otherwise the map would silently win and a sweep would
    # measure the config's value instead of the one that was asked for.
    for _ov, _key in (("salient_lr", "salient_lr"),
                      ("saltq_scales_lr", "scales_lr"),
                      ("zp_lr", "zp_lr")):
        if overrides.get(_ov) is not None:
            _sq = cfg["qat"].setdefault("saltq", {})
            _sq[_key] = overrides[_ov]
            _sq.pop(f"{_key}_by_bits", None)
    if overrides.get("train_layernorms") is not None:
        cfg["qat"].setdefault("saltq", {})["train_layernorms"] = overrides["train_layernorms"]
    if overrides.get("report_to"):
        cfg["training"]["report_to"] = overrides["report_to"]

    if cfg["qat"].get("mode") == "qalora":
        cfg["qat"]["symmetric"] = False

    # Data overrides
    if overrides.get("train_dataset"):
        cfg["data"]["train_dataset"] = overrides["train_dataset"]
    if overrides.get("prompt_template"):
        cfg["data"]["prompt_template"] = overrides["prompt_template"]
    if overrides.get("train_split"):
        cfg["data"]["train_split"] = overrides["train_split"]
    if overrides.get("val_split") is not None:
        cfg["data"]["val_split"] = overrides["val_split"]
    if overrides.get("max_train_samples") is not None:
        cfg["data"]["max_train_samples"] = overrides["max_train_samples"]
    if overrides.get("max_eval_samples") is not None:
        cfg["data"]["max_eval_samples"] = overrides["max_eval_samples"]
    if overrides.get("validation_size") is not None:
        cfg["data"]["validation_size"] = overrides["validation_size"]
    if overrides.get("num_proc") is not None:
        cfg["data"]["num_proc"] = overrides["num_proc"]
    if overrides.get("sub_task"):
        cfg["data"]["sub_task"] = overrides["sub_task"]

    return cfg


def main():
    parser = argparse.ArgumentParser(description="QLoRA + QAT Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--qat_mode",
        type=str,
        choices=["none", "full", "sqat", "qalora", "sqat_permute", "saltq"],
        default=None,
    )
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--symmetric", dest="symmetric", action="store_true", default=None,
                        help="Use symmetric quantization kernels.")
    parser.add_argument("--asymmetric", dest="symmetric", action="store_false",
                        help="Use affine asymmetric quantization kernels with zero_point.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--top_k_ratio", type=float, default=None,
                        help="Top-k ratio for original input-side SQAT.")
    parser.add_argument(
        "--salient_gain_alpha", type=float, default=None,
        help="AWQ-style saliency amplification exponent alpha. "
             "D[j] = (E[x_j^2])^alpha, normalized so min(D)=1. "
             "0.5 is the AWQ default. 0.0 disables amplification (default).",
    )
    parser.add_argument("--salient_gain_max", type=float, default=2.0,
                        help="Maximum value for AWQ-style saliency amplification D. "
                             "If not set, defaults to 2.0.")
    parser.add_argument("--awq_scale", dest="awq_scale",
                        action="store_true", default=None,
                        help="sqat_permute: AWQ-style per-channel scaling of the salient slice "
                             "(q/k/v share S1, gate/up share S2, down its own S3); quantize in the "
                             "amplified space, bake 1/S into the dense weight at export.")
    parser.add_argument("--no_awq_scale", dest="awq_scale",
                        action="store_false",
                        help="sqat_permute: disable AWQ-style salient scaling (original scheme).")
    parser.add_argument("--gptq_nonsalient", dest="gptq_nonsalient",
                        action="store_true", default=None,
                        help="sqat_permute: at export, GPTQ-quantize the non-salient columns "
                             "(salient slice stays on the canonical grid).")
    parser.add_argument("--no_gptq_nonsalient", dest="gptq_nonsalient",
                        action="store_false",
                        help="sqat_permute: disable GPTQ for non-salient cols (plain RTN export).")
    parser.add_argument("--enable_lsq", dest="enable_lsq",
                        action="store_true", default=None,
                        help="Use LSQ/LSQ+ learnable quantization scale (asym→learn scale+zp, "
                             "sym→learn scale), init current_minmax. Replaces min-max fakequant "
                             "for full QAT (and sqat_permute salient slice). Default off.")
    parser.add_argument("--no_enable_lsq", dest="enable_lsq",
                        action="store_false",
                        help="Disable LSQ (use the original per-step min-max scale).")
    parser.add_argument("--salient_lr", type=float, default=None,
                        help="saltq: lr for the salient WEIGHT parameters (real weights, not an "
                             "adapter — must be well below a LoRA lr). Default = training.lr.")
    parser.add_argument("--saltq_scales_lr", type=float, default=None,
                        help="saltq: lr for the SCALES (weight units).")
    parser.add_argument("--zp_lr", type=float, default=None,
                        help="saltq: lr for the ZERO-POINTS. They live in quantization-level "
                             "units (meaningful step 1.0), so this must be 2-3 orders of "
                             "magnitude above the scale lr or they never move.")
    parser.add_argument("--train_layernorms", dest="train_layernorms",
                        action="store_true", default=None,
                        help="saltq: also train the RMSNorm weights (they stay fp16 at deploy, so "
                             "this is free extra task-adaptation freedom).")
    parser.add_argument("--no_train_layernorms", dest="train_layernorms", action="store_false",
                        help="saltq: keep the RMSNorm weights frozen (default).")
    parser.add_argument("--report_to", type=str, default=None)

    # Data overrides
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="Training dataset name/path. e.g. metamath or meta-math/MetaMathQA")
    parser.add_argument("--prompt_template", type=str, default=None,
                        choices=["commonsense_qa", "alpaca", "metamath", "metamathqa", "meta_math"])
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None,
                        help="Validation split name. Use empty string in config if you want no eval split.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--validation_size", type=float, default=None,
                        help="If dataset has no validation split, carve one out from train "
                             "(e.g. 0.01 or 1000 if using datasets train_test_split semantics).")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--sub_task", nargs="+", default=None,
                        help="Restrict training to these `type` values, optionally capped as "
                             "name:N (e.g. --sub_task boolq piqa:2000). Default: every task.")

    # Resume training from a Trainer checkpoint (e.g. .../checkpoint-6000). For sqat_permute this
    # REUSES the existing permuted fp16 base (it must NOT be regenerated — a fresh permute would
    # not match the checkpoint's LoRA). Optionally override the base dir with --permuted_base_dir.
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a Trainer checkpoint dir to resume training from.")
    parser.add_argument("--permuted_base_dir", type=str, default=None,
                        help="sqat_permute / saltq: explicit permuted fp16 base dir to reuse on "
                             "resume (defaults to <output_dir>/permuted_fp16_base).")
    parser.add_argument("--saltq_base_dir", type=str, default=None,
                        help="saltq: explicit frozen-code base dir (defaults to "
                             "<output_dir>/saltq_base). Must be reused on resume/export.")

    # Export mode
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--export_dequant", action="store_true",
                        help="Export dequantized weights (FP16) instead of merged INT4")
    parser.add_argument("--gptq_full", action="store_true",
                        help="SQAT ablation: GPTQ the FULL merged weight (no salient slice, no "
                             "AWQ) — isolates the Selective-QAT contribution. Needs --export_dequant.")
    parser.add_argument("--export_merged_only", action="store_true",
                        help="Export merged weights only (no quantize and dequantize)")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--merge_output_dir", type=str, default=None)
    parser.add_argument("--adapter_only", action="store_true",
                        help="Save adapter weights only (no merge)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, vars(args))
    set_seed(cfg["training"]["seed"])

    qat_mode = cfg["qat"]["mode"]
    bits = cfg["model"]["quant_bits"]
    symmetric = cfg["qat"].get("symmetric", True)
    print("=" * 70)
    print(f"  QLoRA Training — {bits}-bit | QAT mode: {qat_mode} | symmetric={symmetric}")
    print("=" * 70)
    print(f"  Train dataset:   {cfg['data']['train_dataset']}")
    print(f"  Train split:     {cfg['data'].get('train_split', 'train')}")
    print(f"  Val split:       {cfg['data'].get('val_split')}")
    print(f"  Sub-tasks:       {cfg['data'].get('sub_task') or 'all'}")
    if qat_mode == "qalora":
        print("  QA-LoRA:         asymmetric affine quantization only")

    accelerator = Accelerator()

    # --- Export-only mode ---
    if args.export_only:
        assert args.checkpoint_dir, "--checkpoint_dir required for --export_only"

        # For SQAT export-only, we need metadata.
        # The metadata is saved alongside the adapter checkpoint.
        sqat_metadata = None
        if qat_mode in {"sqat"}:
            meta_path = os.path.join(args.checkpoint_dir, "sqat_metadata.pt")
            if os.path.exists(meta_path):
                sqat_metadata = torch.load(meta_path, map_location="cpu")
                print(f"[Export] Loaded SQAT metadata from {meta_path}")
            else:
                print(f"[Export] WARNING: SQAT mode but no metadata at {meta_path}")
                print(f"[Export] PTQ will use standard rounding (potential mismatch!)")

        if qat_mode == "sqat_permute":
            perm_meta_path = os.path.join(args.checkpoint_dir, "sqat_permute_meta.pt")
            if os.path.exists(perm_meta_path):
                # The adapter was trained on the PERMUTED fp16 base, so the merge must reload
                # that exact base (not the original) — otherwise permuted LoRA is applied to
                # un-permuted weights. The base path is recorded in perm_meta.
                _pm         = torch.load(perm_meta_path, map_location="cpu")
                _model_meta = _pm.get("model", _pm) if isinstance(_pm, dict) else {}
                _base       = (_model_meta or {}).get("permuted_base_dir")
                if _base and os.path.isdir(_base):
                    print(f"[Export] sqat_permute: using permuted fp16 base {_base}")
                    cfg["model"]["name"] = _base
                else:
                    print(f"[Export] WARNING: sqat_permute permuted_base_dir missing/not found "
                          f"({_base!r}); merge would use the ORIGINAL base and be INCORRECT.")
            else:
                print(f"[Export] WARNING: sqat_permute mode but no metadata at {perm_meta_path}")

        if qat_mode == "qalora":
            # The adapter was trained against the frozen GPTQ INT-b base; reload that exact base for
            # the dequant export. For merged-only (no-quant upper bound) use the ORIGINAL fp16 base.
            qa_meta_path  = os.path.join(args.checkpoint_dir, "qalora_meta.pt")
            intb_base_dir = (torch.load(qa_meta_path, map_location="cpu").get("intb_base_dir")
                             if os.path.exists(qa_meta_path) else None)
            if not (intb_base_dir and os.path.isdir(intb_base_dir)):
                raise FileNotFoundError(
                    f"[Export] QA-LoRA intb base dir missing/not found ({intb_base_dir!r}). The "
                    f"export REQUIRES the GPTQ base the adapter was trained against — re-check "
                    f"{qa_meta_path} (was the base dir deleted?). Refusing to export the wrong base."
                )
            elif args.export_merged_only:
                _bm_path = os.path.join(intb_base_dir, "qalora_base_meta.pt")
                _orig    = (torch.load(_bm_path, map_location="cpu").get("orig_base_name")
                            if os.path.exists(_bm_path) else None)
                cfg["model"]["name"] = _orig or intb_base_dir
                print(f"[Export] QA-LoRA merged-only: original fp16 base {cfg['model']['name']}")
            else:
                cfg["model"]["name"] = intb_base_dir
                print(f"[Export] QA-LoRA: GPTQ INT-b base {intb_base_dir}")

        if qat_mode == "saltq":
            # SALT-Q has no adapter and no merge: the deployed weight is read straight out of the
            # trained parameters. A checkpoint only stores the trainable tensors plus a pointer
            # back to the frozen-code base, so that base is REQUIRED here.
            from src.qat_saltq import export_saltq, saltq_base_dir_from_checkpoint

            saltq_base = args.saltq_base_dir or saltq_base_dir_from_checkpoint(args.checkpoint_dir)
            if not (saltq_base and os.path.isdir(saltq_base)):
                raise FileNotFoundError(
                    f"[Export] SALT-Q frozen-code base not found ({saltq_base!r}). The checkpoint "
                    f"stores only trainable tensors; pass --saltq_base_dir explicitly or check "
                    f"saltq_ckpt_meta.pt in {args.checkpoint_dir}."
                )
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
            export_saltq(
                cfg=cfg, tokenizer=tokenizer, saltq_base_dir=saltq_base,
                checkpoint_dir=args.checkpoint_dir, output_dir=args.merge_output_dir,
            )
            return

        # Don't need to load quantized model — export loads FP16 base separately
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
        if args.export_merged_only:
            print("\n[Export] Exporting merged weights only (no quant/dequant)...")
            export_merged_only(
                model=None, tokenizer=tokenizer, cfg=cfg,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.merge_output_dir,
            )
        else:
            merge_and_export(
                model=None, tokenizer=tokenizer, cfg=cfg,
                export_dequant=args.export_dequant,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.merge_output_dir,
                sqat_metadata=sqat_metadata,
                gptq_full=args.gptq_full,
            )
        return

    # --- SQAT-Permute: permute in fp16 and re-save BEFORE the NF4 load ---------------
    # load_in_4bit quantizes at load time; permuting an already-NF4 model would be a
    # dequant->permute->requant round-trip (double quantization). Instead, rank 0 loads fp16,
    # runs the validated permute/fold, and saves the permuted base; then ALL ranks reload it
    # through the standard NF4 path so NF4 quantizes the permuted weights exactly once.
    # The boundary gather is a runtime residual reorder (cannot be folded): it is re-registered
    # in prepare_model for training and on the exported model for inference (eval scripts).
    #
    # SALT-Q shares this entire pre-step: it needs exactly the same permuted fp16 base (the
    # salient channels must sit in the leading group_k columns before anything is quantized).
    # It then adds one more offline stage (build_saltq_base) that GPTQ-quantizes that base into
    # the frozen integer codes it trains (s, z) on top of.
    perm_meta = None
    saltq_meta = None
    saltq_base_dir = None
    if qat_mode in ("sqat_permute", "saltq"):
        from src.qat_permute_sqat import build_permuted_fp16_checkpoint, load_perm_meta

        sp_cfg       = cfg["qat"]["saltq" if qat_mode == "saltq" else "sqat_permute"]
        permuted_dir = args.permuted_base_dir or os.path.join(
            cfg["training"]["output_dir"], "permuted_fp16_base")
        _perm_reuse = _permuted_base_reusable(permuted_dir, cfg, sp_cfg)

        if args.resume_from_checkpoint:
            # 恢复训练：必须复用原训练的 permuted base，绝不重新生成 —— 新的 saliency/permute
            # 会与 checkpoint 里的 LoRA 错位，训练会立刻崩坏。
            _meta_pt = os.path.join(permuted_dir, "sqat_permute_meta.pt")
            if not (os.path.isdir(permuted_dir) and os.path.exists(_meta_pt)):
                raise FileNotFoundError(
                    f"[SQAT-Permute][Resume] 找不到原训练的 permuted base: {permuted_dir} "
                    f"(缺 sqat_permute_meta.pt)。恢复训练必须复用与 checkpoint 一致的 permuted "
                    f"base — 用 --permuted_base_dir 显式指定，或确认它未被删除。"
                )
            if accelerator.is_main_process:
                print(f"\n[SQAT-Permute][Resume] 复用已有 permuted base（不重新生成）: {permuted_dir}")
        elif _perm_reuse is not None:
            # Bit-independent artifact: reuse it across the bit sweep so 3-bit and 2-bit differ by
            # exactly one variable (and skip the calibration pass + 13 GB rewrite).
            if accelerator.is_main_process:
                print(f"\n[SQAT-Permute] 复用已有 permuted base（配置一致，不重建）: {permuted_dir}\n"
                      f"[SQAT-Permute]   {_perm_reuse}")
        elif accelerator.is_main_process:
            print("\n[SQAT-Permute] Building permuted fp16 base (permute BEFORE NF4)...")
            sp_tok = load_tokenizer(cfg)
            cal_dataset  = load_calibration_data(cfg, sp_tok)
            cal_dataloader = DataLoader(
                cal_dataset,
                batch_size=cfg["training"]["per_device_eval_batch_size"],
                collate_fn=build_data_collator(sp_tok), shuffle=False,
            )
            build_permuted_fp16_checkpoint(
                model_name=cfg["model"]["name"],
                tokenizer=sp_tok,
                calibration_dataloader=cal_dataloader,
                boundary_sizes=sp_cfg.get("boundary_sizes"),
                save_dir=permuted_dir,
                target_modules=cfg["lora"].get("target_modules"),
                group_k=sp_cfg.get("group_k"),
                group_size=cfg["qat"].get("group_size", 128),
                top_k_ratio=sp_cfg.get("top_k_ratio", 0.01),
                outlier_log_sigma=sp_cfg.get("outlier_log_sigma", 3.0),
                down_outlier_log_sigma=sp_cfg.get(
                    "down_outlier_log_sigma", sp_cfg.get("outlier_log_sigma", 3.0)
                ),
                dtype=getattr(torch, cfg["model"]["dtype"]),
                device=accelerator.device,
                awq_alpha=(sp_cfg.get("awq_scale", {}) or {}).get("alpha", 0.5),
                awq_max=(sp_cfg.get("awq_scale", {}) or {}).get("max", 2.0),
                max_segments=sp_cfg.get("max_segments", 4),
                # Fold S into the weights offline (producers divided) so nothing downstream needs
                # to know about AWQ, and order the salient block by post-fold magnitude so the
                # channels S amplifies share a group with each other.
                fold_awq=bool((sp_cfg.get("awq_scale", {}) or {}).get("fold", False)),
                reorder_salient=bool(sp_cfg.get("reorder_salient", False)),
            )
        accelerator.wait_for_everyone()

        # All ranks: point the base model at the PERMUTED fp16 checkpoint and read perm_meta.
        cfg["model"]["name"] = permuted_dir
        perm_meta = load_perm_meta(permuted_dir)
        print(f"[SQAT-Permute] Using permuted base {permuted_dir} "
              f"(num_runtime_permutes={len(perm_meta['boundary_perms'])})")

    # --- SALT-Q: GPTQ the permuted base into the FROZEN codes + initial (s, z) -----------------
    # This is the only extra offline stage vs sqat_permute. It is what makes the non-salient 98%
    # trainable-in-deployment-form: after this the codes never change again, and training only
    # moves the affine (s, z) that a GPTQ checkpoint already carries in its metadata slots.
    if qat_mode == "saltq":
        from src.qat_saltq import SALTQ_META_FILENAME, build_saltq_base

        sq_cfg = cfg["qat"]["saltq"]
        sq_gptq = sq_cfg.get("gptq", {}) or {}
        saltq_base_dir = _saltq_base_dir_for(cfg, args.saltq_base_dir)
        _sq_meta_pt = os.path.join(saltq_base_dir, SALTQ_META_FILENAME)
        _sq_match = _saltq_base_matches(saltq_base_dir, cfg)
        # (bits, group_size, symmetric) is NOT enough to license reuse. The codes also address a
        # specific permutation, and `permuted_dir` above may have just been rebuilt underneath
        # them. See _permutation_fingerprint for what that costs.
        _sq_perm_ok = (_sq_match is None
                       or _saltq_base_permutation_matches(saltq_base_dir, perm_meta))
        if not _sq_perm_ok:
            raise RuntimeError(
                f"[SALT-Q] {saltq_base_dir} 的 frozen codes 是针对另一份 permutation 做的 GPTQ，"
                f"与 {permuted_dir} 当前的 permutation 不一致。\n"
                f"  codes 记录的是 permuted 权重里 FIXED 列位置上的整数，换一份 permutation 就等于"
                f"把每个量化组的成员打散——重构误差会超过 100%（比直接输出 0 还差），而直方图、"
                f"loss 曲线全都看不出来。\n"
                f"  修法二选一：删掉 {saltq_base_dir} 让它跟着新 permutation 重跑 GPTQ；或者用 "
                f"--permuted_base_dir 指向当初生成这批 codes 的那份 permuted base。"
            )

        if args.resume_from_checkpoint:
            if not os.path.exists(_sq_meta_pt):
                raise FileNotFoundError(
                    f"[SALT-Q][Resume] 找不到原训练的 frozen-code base: {saltq_base_dir}。恢复训练"
                    f"必须复用同一份 codes —— 重新 GPTQ 会给出不同的离散码字，checkpoint 里训好的 "
                    f"(s, z) 与 salient 权重会全部错位。用 --saltq_base_dir 显式指定。"
                )
            if accelerator.is_main_process:
                print(f"\n[SALT-Q][Resume] 复用已有 frozen-code base（不重新 GPTQ）: {saltq_base_dir}")
        elif os.path.exists(_sq_meta_pt) and _sq_match is None:
            # An existing base under this path was built for a DIFFERENT grid. Never overwrite it
            # silently: some checkpoint elsewhere still points at those codes.
            raise RuntimeError(
                f"[SALT-Q] {saltq_base_dir} 里已有一份 frozen-code base，但它的 (bits, group_size, "
                f"symmetric) 与本次配置不一致。codes 是一次性的离散选择，覆盖它会让指向它的 "
                f"checkpoint 永久失效。请换一个 --saltq_base_dir，或先把旧的移开。"
            )
        elif _sq_match is not None:
            if accelerator.is_main_process:
                print(f"\n[SALT-Q] 复用已有 frozen-code base（{_sq_match}，配置一致，不重新 GPTQ）: "
                      f"{saltq_base_dir}")
        elif accelerator.is_main_process:
            print("\n[SALT-Q] Building frozen-code base (GPTQ on the permuted fp16 base)...")
            sq_tok = load_tokenizer(cfg, name=permuted_dir)
            cal_dataset  = load_calibration_data(cfg, sq_tok)
            cal_loader   = DataLoader(
                cal_dataset,
                batch_size=int(sq_gptq.get("batch_size", 2)),
                collate_fn=build_data_collator(sq_tok),
                shuffle=False,
            )
            build_saltq_base(
                permuted_base_dir=permuted_dir,
                perm_meta=perm_meta,
                tokenizer=sq_tok,
                calibration_dataloader=cal_loader,
                target_terminals=cfg["lora"]["target_modules"],
                group_size=cfg["qat"].get("group_size", 128),
                q_bits=cfg["model"]["quant_bits"],
                symmetric=cfg["qat"].get("symmetric", False),
                save_dir=saltq_base_dir,
                device=accelerator.device,
                percdamp=float(sq_gptq.get("percdamp", 0.01)),
                blocksize=int(sq_gptq.get("blocksize", 128)),
                nsamples=int(sq_gptq.get("nsamples", 128)),
                dtype=getattr(torch, cfg["model"]["dtype"]),
                train_salient=bool(sq_cfg.get("train_salient", True)),
            )
        accelerator.wait_for_everyone()
        print(f"[SALT-Q] Using frozen-code base {saltq_base_dir}")

    # --- QA-LoRA: build the GPTQ INT-b frozen base BEFORE loading (faithful, no NF4 double-quant) --
    # The official QA-LoRA trains on a REAL GPTQ INT-b base (no NF4). Rank 0 quantizes the fp16 base
    # to INT-b g{group_size} once (calibrated), saves it; all ranks then load THAT fp16 checkpoint
    # frozen via model_loader's qalora path. The grid is identical train↔export (same checkpoint).
    # NB: build the base under the SAME "-{bits}bit-{mode}" dir the Trainer uses (src/trainer.py), so
    # it sits next to the final/ checkpoint regardless of the un-suffixed cfg output_dir.
    if qat_mode == "qalora":
        from src.qalora import build_qalora_intb_base

        _suffixed_out = f"{cfg['training']['output_dir']}-{bits}bit-{qat_mode}"
        qa_base_dir = os.path.join(_suffixed_out, "qalora_intb_base")
        qa_gptq = (cfg["qat"].get("qalora", {}) or {}).get("gptq", {}) or {}

        if args.resume_from_checkpoint:
            if not (os.path.isdir(qa_base_dir)
                    and os.path.exists(os.path.join(qa_base_dir, "qalora_base_meta.pt"))):
                raise FileNotFoundError(
                    f"[QA-LoRA][Resume] missing GPTQ base {qa_base_dir}; resume must reuse the base "
                    f"the checkpoint was trained against (a fresh GPTQ base would not match the LoRA)."
                )
            if accelerator.is_main_process:
                print(f"\n[QA-LoRA][Resume] reusing existing GPTQ base: {qa_base_dir}")
        elif accelerator.is_main_process:
            print("\n[QA-LoRA] Building GPTQ INT-b base (quantize BEFORE training, no NF4)...")
            qa_tok = load_tokenizer(cfg)
            cal_dataset  = load_calibration_data(cfg, qa_tok)
            cal_dataloader = DataLoader(
                cal_dataset,
                batch_size=int(qa_gptq.get("batch_size", 2)),
                collate_fn=build_data_collator(qa_tok), shuffle=False,
            )
            build_qalora_intb_base(
                model_name=cfg["model"]["name"],
                tokenizer=qa_tok,
                calibration_dataloader=cal_dataloader,
                target_terminals=cfg["lora"]["target_modules"],
                group_size=cfg["qat"].get("group_size", 128),
                q_bits=cfg["model"]["quant_bits"],
                symmetric=cfg["qat"].get("symmetric", False),
                save_dir=qa_base_dir,
                device=accelerator.device,
                percdamp=float(qa_gptq.get("percdamp", 0.01)),
                blocksize=int(qa_gptq.get("blocksize", 128)),
                nsamples=int(qa_gptq.get("nsamples", 128)),
                dtype=getattr(torch, cfg["model"]["dtype"]),
            )
        accelerator.wait_for_everyone()
        # All ranks: load the GPTQ INT-b base in fp16 (model_loader's qalora path), not NF4.
        cfg["model"]["name"] = qa_base_dir
        print(f"[QA-LoRA] Using GPTQ INT-b base {qa_base_dir}")

    # --- Load model ---
    print("\n[1/5] Loading model and tokenizer...")
    if qat_mode == "saltq":
        # No bitsandbytes, no PEFT: SALT-Q's layers ARE the quantized layers (frozen int8 codes
        # + trainable salient weights + trainable (s, z)). Nothing here is a BF16 side-car that
        # would later have to be merged.
        from src.qat_saltq import build_saltq_model

        tokenizer = load_tokenizer(cfg, name=saltq_base_dir)
        model, saltq_meta = build_saltq_model(
            saltq_base_dir,
            dtype=getattr(torch, cfg["model"]["dtype"]),
            param_dtype=torch.float32,
            gradient_checkpointing=True,
            train_layernorms=bool(cfg["qat"]["saltq"].get("train_layernorms", False)),
            train_scale=bool(cfg["qat"]["saltq"].get("train_scale", False)),
            continuous_z=bool(cfg["qat"]["saltq"].get("continuous_z", True)),
        )
        base_model_ref = model
    else:
        model, tokenizer, base_model_ref = load_model_and_tokenizer(cfg)

    # --- Load data ---
    print("\n[2/5] Loading datasets...")
    # Tokenize on rank 0 FIRST, then let the other ranks read the finished cache.
    #
    # load_dataset_for_training calls datasets.map(num_proc=4) with a content-addressed cache
    # path, so under `accelerate launch --num_processes 4` all four ranks compute the SAME cache
    # filenames and race to write them, each with four worker processes of its own. Usually one
    # rank wins and the rest silently reuse its files; occasionally a loser finishes writing after
    # the winner has already renamed the temp file away and datasets' post-write
    # `os.chmod(cache_file_name, ...)` dies with FileNotFoundError on
    # cache-<hash>_00000_of_00004.arrow. That killed the AWQ-legacy run after the bases were
    # already built. main_process_first serializes it: rank 0 populates the cache, the others
    # enter afterwards and hit it, which is also ~4x less tokenization work.
    with accelerator.main_process_first():
        train_dataset, eval_dataset = load_dataset_for_training(cfg, tokenizer)
    print(f"  Train: {len(train_dataset)} samples")
    if eval_dataset is not None:
        print(f"  Eval:  {len(eval_dataset)} samples")
    else:
        print("  Eval:  None")

    # --- QAT setup ---
    print(f"\n[3/5] Setting up QAT handler: {qat_mode}")
    qat_handler = get_qat_handler(cfg)

    # SQAT needs a calibration dataloader here. SQAT-Permute already did calibration +
    # permute/fold in the fp16 pre-step above, so it only passes perm_meta.
    qat_kwargs = {}
    if qat_mode == "sqat":
        print("  Loading calibration data for SQAT...")
        cal_dataset = load_calibration_data(cfg, tokenizer)
        cal_dataloader = DataLoader(
            cal_dataset,
            batch_size=cfg["training"]["per_device_eval_batch_size"],
            collate_fn=build_data_collator(tokenizer),
            shuffle=False,
        )
        qat_kwargs["calibration_dataloader"] = cal_dataloader
        qat_kwargs["tokenizer"] = tokenizer
    elif qat_mode == "sqat_permute":
        qat_kwargs["perm_meta"] = perm_meta
        qat_kwargs["tokenizer"] = tokenizer
    elif qat_mode == "saltq":
        qat_kwargs["saltq_meta"] = saltq_meta
        qat_kwargs["saltq_base_dir"] = saltq_base_dir
        qat_kwargs["tokenizer"] = tokenizer

    model = qat_handler.prepare_model(model, cfg, **qat_kwargs)

    # --- Build trainer ---
    print("\n[4/5] Building trainer...")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
        qat_handler=qat_handler,
        saltq_base_dir=saltq_base_dir,
    )

    # --- Train ---
    print("\n[5/5] Starting training...")
    if args.resume_from_checkpoint:
        print(f"  Resuming from checkpoint: {args.resume_from_checkpoint}")
        if qat_mode == "saltq":
            print(f"  [一致性提醒] SALT-Q 恢复训练复用 frozen-code base {saltq_base_dir}。"
                  f"codes 是离散的、一次性 GPTQ 决定的——只要 base 一致，(s,z) 与 salient 权重"
                  f"就能接着训；base 不一致则 checkpoint 完全无效。")
        if qat_mode == "sqat_permute":
            _awq = bool((cfg["qat"]["sqat_permute"].get("awq_scale", {}) or {}).get("enabled", False))
            print(f"  [一致性提醒] 本次 awq_scale={_awq}, group_size={perm_meta['group_size']} "
                  f"(来自复用的 permuted base)。AWQ 改变训练时的 fakequant 网格——必须与该 checkpoint "
                  f"原始训练时一致，否则 resume 后 loss 会跳变。GPTQ 仅在导出生效，不影响训练连续性。")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # --- Collect metadata BEFORE any save/unwrap ---
    sqat_metadata = None
    if qat_mode in {"sqat"}:
        from src.export import collect_sqat_metadata
        sqat_metadata = collect_sqat_metadata(model)

    sqat_permute_metadata = None
    if qat_mode == "sqat_permute":
        from src.export import collect_sqat_permute_metadata
        sqat_permute_metadata = collect_sqat_permute_metadata(model)

    # --- Save final checkpoint ---
    final_dir = os.path.join(trainer.args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal adapter saved to {final_dir}")

    # Persist metadata for future export-only runs
    if sqat_metadata:
        meta_path = os.path.join(final_dir, "sqat_metadata.pt")
        torch.save(sqat_metadata, meta_path)
        print(f"SQAT metadata saved to {meta_path}")

    # QA-LoRA: record the GPTQ INT-b base dir so --export_only can reload the exact frozen base the
    # adapter was trained against (cfg["model"]["name"] is already that dir here).
    if qat_mode == "qalora":
        torch.save(
            {"intb_base_dir": cfg["model"]["name"]},
            os.path.join(final_dir, "qalora_meta.pt"),
        )
        print(f"QA-LoRA base dir recorded: {cfg['model']['name']}")

    # FullQAT + LSQ: the learned scale[,zp] are self-registered nn.Parameters that PEFT
    # save_pretrained does NOT persist. Save them next to the final adapter so --export_only
    # can read the exact training grid. The injectors' remove() (on_train_end) only restores
    # the module forward, so the params still hold their trained values here.
    if qat_mode == "full" and hasattr(qat_handler, "save_lsq_scales"):
        qat_handler.save_lsq_scales(final_dir)

    if sqat_permute_metadata:
        from src.export import save_sqat_permute_meta
        save_sqat_permute_meta(sqat_permute_metadata, final_dir)

    # --- Export ---
    if accelerator.is_main_process:
        if qat_mode == "saltq":
            # No merge, no re-quantization: export_saltq reads the deployed weight straight out of
            # the trained parameters and asserts it equals the training-time weight exactly.
            if cfg.get("export", {}).get("merge_and_save", False):
                from src.qat_saltq import export_saltq

                print("\n[SALT-Q] Exporting deployed model (merge-free)...")
                export_saltq(
                    cfg=cfg, tokenizer=tokenizer, saltq_base_dir=saltq_base_dir,
                    model=accelerator.unwrap_model(model),
                    output_dir=args.merge_output_dir,
                )
            print("\nDone!")
            return

        if cfg.get("export", {}).get("merge_and_save", False):
            print("\nExporting for vLLM (INT4 GPTQ)...")

            if args.adapter_only:
                export_adapter_only(model, tokenizer, cfg)
            else:
                merge_and_export(
                    model,
                    tokenizer,
                    cfg,
                    sqat_metadata=sqat_metadata,
                    export_dequant=args.export_dequant,
                )

        print("\nDone!")


if __name__ == "__main__":
    main()
