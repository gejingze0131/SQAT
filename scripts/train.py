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
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
from accelerate import Accelerator

from src.model_loader import load_model_and_tokenizer
from src.data import load_dataset_for_training, load_calibration_data
from src.trainer import build_trainer
from src.qat_base import get_qat_handler
from src.export import export_merged_only, merge_and_export, export_adapter_only


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

    return cfg


def main():
    parser = argparse.ArgumentParser(description="QLoRA + QAT Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--qat_mode",
        type=str,
        choices=["none", "full", "sqat", "qalora"],
        default=None,
    )
    parser.add_argument("--bits", type=int, choices=[3, 4], default=None)
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

    # Export mode
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--export_dequant", action="store_true",
                        help="Export dequantized weights (FP16) instead of merged INT4")
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
    print(f"  Prompt template: {cfg['data'].get('prompt_template', cfg['data']['train_dataset'])}")
    print(f"  Train split:     {cfg['data'].get('train_split', 'train')}")
    print(f"  Val split:       {cfg['data'].get('val_split', 'validation')}")
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
            )
        return

    # --- Load model ---
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer, base_model_ref = load_model_and_tokenizer(cfg)

    # --- Load data ---
    print("\n[2/5] Loading datasets...")
    train_dataset, eval_dataset = load_dataset_for_training(cfg, tokenizer)
    print(f"  Train: {len(train_dataset)} samples")
    if eval_dataset is not None:
        print(f"  Eval:  {len(eval_dataset)} samples")
    else:
        print("  Eval:  None")

    # --- QAT setup ---
    print(f"\n[3/5] Setting up QAT handler: {qat_mode}")
    qat_handler = get_qat_handler(cfg)

    # For SQAT variants, we need a calibration dataloader
    qat_kwargs = {}
    if qat_mode in {"sqat"}:
        print("  Loading calibration data for SQAT...")
        cal_dataset = load_calibration_data(cfg, tokenizer)
        from transformers import DataCollatorForSeq2Seq
        cal_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt",
        )
        cal_dataloader = DataLoader(
            cal_dataset,
            batch_size=cfg["training"]["per_device_eval_batch_size"],
            collate_fn=cal_collator,
            shuffle=False,
        )
        qat_kwargs["calibration_dataloader"] = cal_dataloader
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
    )

    # --- Train ---
    print("\n[5/5] Starting training...")
    trainer.train()

    # --- Collect SQAT metadata BEFORE any save/unwrap ---
    sqat_metadata = None
    if qat_mode in {"sqat"}:
        from src.export import collect_sqat_metadata
        sqat_metadata = collect_sqat_metadata(model)

    # --- Save final checkpoint ---
    final_dir = os.path.join(trainer.args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal adapter saved to {final_dir}")

    # Persist SQAT metadata for future export-only runs
    if sqat_metadata:
        meta_path = os.path.join(final_dir, "sqat_metadata.pt")
        torch.save(sqat_metadata, meta_path)
        print(f"SQAT metadata saved to {meta_path}")

    # --- Export ---
    if accelerator.is_main_process:
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
