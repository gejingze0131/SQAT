#!/usr/bin/env python
"""
export_permute_fp16_ablation.py — Methodology UPPER-BOUND ablation for Permuted Selective-QAT.

What our method does (run_permute_sqat.sh):
  permute the residual stream so the salient d_model channels gather into the leading slice
  [0:group_k]; PROTECT that slice with LSQ-style fakequant during QAT; export the non-salient
  columns [group_k:] with GPTQ. The salient slice is deployed as a low-bit (canonical-grid)
  quantized weight that the LoRA was trained against.

What THIS ablation does (the upper bound of that methodology):
  run the IDENTICAL permute algorithm (same calibration set, same saliency selection, same
  P_k / P4 / Hadamard transforms, same boundary gathers), but instead of QAT-protecting the
  salient slice, keep it at FULL PRECISION fp16. The non-salient columns are still GPTQ'd.
  => the ONLY difference vs. our method is salient = fp16 (here) vs. salient = QAT (ours).

No training. We start from a PLAIN QLoRA checkpoint (qat_mode=none, e.g.
outputs/qlora-none-math-3bit-none/final) whose adapter was trained on the ORIGINAL
(un-permuted) NF4 base. Flow:

  1. Calibrate E[x^2] on the CLEAN fp16 base (exactly as build_permuted_fp16_checkpoint /
     permute-SQAT does) -> select salient channels -> build P_k / P4 permutations + boundary
     perms. Using the clean base means the salient slice / permutation are IDENTICAL to the
     real permute-SQAT run, so the comparison is rigorous.
  2. Merge the trained QLoRA adapter into a dense fp16 base IN ORIGINAL ORDER
     (dequant(NF4 base) + B@A * scaling), reusing the export merge helpers.
  3. Apply the three equivalence transforms (P_k residual permute, P4 MLP-internal permute,
     per-head Hadamard) to the merged dense weights. This is a pure equivalence transform on
     the merged weight, so + boundary gathers it reproduces the original merged model exactly.
  4. PTQ: salient slice [0:group_k] -> kept fp16; non-salient [group_k:] -> GPTQ. o_proj has no
     salient slice (group_k=0) -> fully GPTQ. Same group_size / asym-sym as the config.
  5. Save the dense model + tokenizer + sqat_permute_meta.pt (boundary gathers) so the eval
     scripts auto-register the runtime residual reorder, exactly like a real permute export.

All meta (boundary_sizes, group_k, group_size, top_k_ratio, outlier_log_sigma, calibration
samples/seqlen, q_bits, symmetric, target_modules, lora alpha/rank) is read from the SAME
configs/sqat_permute_*.yaml that the permute-SQAT run uses, so nothing-but-the-method differs.

Usage:
  python scripts/export_permute_fp16_ablation.py \
      --config         configs/sqat_permute_math.yaml \
      --checkpoint_dir outputs/qlora-none-math-3bit-none/final \
      --output_dir     outputs/qlora-permute-fp16salient-ablation-math-3bit-dequant-eval
"""

import os
import sys
import argparse
import gc

# Add project root to path (mirror scripts/train.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from src.data import load_calibration_data
from src.export import (
    _load_adapter_for_export,
    _merge_lora_into_dense,
    save_dequantized_model,
    save_sqat_permute_meta,
)
from src.qat_permute_sqat import (
    PERM_META_FILENAME,
    _boundary_layer_indices,
    _build_segment_perm,
    _collect_second_moments,
    apply_block_internal_permutations_fp32,
    apply_hadamard_rotation_fp32,
    apply_segment_permutation_fp32,
    gptq_quantize_model_sequential,
    select_internal_salient_channels,
    select_salient_channels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Permuted Selective-QAT upper-bound ablation: fp16 salient + GPTQ non-salient."
    )
    p.add_argument("--config", type=str, default="configs/sqat_permute_math.yaml",
                   help="Same sqat_permute config the permute-SQAT run uses (meta source).")
    p.add_argument("--checkpoint_dir", type=str,
                   default="outputs/qlora-none-math-3bit-none/final",
                   help="A PLAIN QLoRA (qat_mode=none) checkpoint dir holding the trained adapter.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write the dense dequantized export (+ sqat_permute_meta.pt).")
    p.add_argument("--bits", type=int, choices=[2, 3, 4], default=None,
                   help="Override quant_bits (otherwise read from config).")
    p.add_argument("--symmetric", dest="symmetric", action="store_true", default=None)
    p.add_argument("--asymmetric", dest="symmetric", action="store_false",
                   help="Affine asymmetric quantization (matches the permute-SQAT default).")
    p.add_argument("--dequant_dtype", type=str, default="float16",
                   choices=["float16", "bfloat16"])
    return p.parse_args()


def load_cfg(config_path: str, bits, symmetric) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if bits is not None:
        cfg["model"]["quant_bits"] = bits
    if symmetric is not None:
        cfg["qat"]["symmetric"] = symmetric
    # This export is the sqat_permute pipeline regardless of the checkpoint's own qat_mode.
    cfg["qat"]["mode"] = "sqat_permute"
    return cfg


def build_calibration_loader(cfg: dict, tokenizer) -> DataLoader:
    """Same calibration data + collator + batching as build_permuted_fp16_checkpoint (train.py)."""
    cal_dataset = load_calibration_data(cfg, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    return DataLoader(
        cal_dataset,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        collate_fn=collator,
        shuffle=False,
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config, args.bits, args.symmetric)
    set_seed(cfg["training"].get("seed", 42))

    sp_cfg         = cfg["qat"]["sqat_permute"]
    boundary_sizes = sp_cfg["boundary_sizes"]
    group_k        = sp_cfg.get("group_k", 128)
    group_size     = cfg["qat"].get("group_size", 128)
    top_k_ratio    = sp_cfg.get("top_k_ratio", 0.01)
    outlier_sigma  = sp_cfg.get("outlier_log_sigma", 3.0)
    q_bits         = cfg["model"]["quant_bits"]
    symmetric      = cfg["qat"].get("symmetric", True)
    base_name      = cfg["model"]["name"]
    target_modules = cfg["lora"]["target_modules"]
    lora_scaling   = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
    base_dtype     = getattr(torch, cfg["model"].get("dtype", "float16"))

    gptq_cfg = (sp_cfg.get("gptq", {}) or {})
    percdamp  = float(gptq_cfg.get("percdamp", 0.01))
    blocksize = int(gptq_cfg.get("blocksize", 128))
    nsamples  = int(gptq_cfg.get("nsamples", 128))
    gptq_bs   = int(gptq_cfg.get("batch_size", 2))

    output_dir = args.output_dir or (
        f"{cfg['training']['output_dir']}-permute-fp16salient-ablation-"
        f"{q_bits}bit-dequant-eval"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  Permuted Selective-QAT UPPER-BOUND ablation (fp16 salient + GPTQ non-salient)")
    print(f"  Config:      {args.config}")
    print(f"  Checkpoint:  {args.checkpoint_dir}  (plain QLoRA adapter)")
    print(f"  Base model:  {base_name}")
    print(f"  Output:      {output_dir}")
    print(f"  INT{q_bits} | group_size={group_size} | group_k={group_k} | "
          f"symmetric={symmetric} | boundary_sizes={boundary_sizes}")
    print("=" * 72)

    assert os.path.isdir(args.checkpoint_dir), f"checkpoint_dir not found: {args.checkpoint_dir}"
    assert sum(boundary_sizes) > 0

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---------------------------------------------------------------------------------
    # 1) Calibrate the CLEAN fp16 base and select salient channels (== permute-SQAT).
    # ---------------------------------------------------------------------------------
    print("\n[1/5] Calibrating E[x^2] on the clean fp16 base for saliency selection ...")
    cal_loader = build_calibration_loader(cfg, tokenizer)

    clean_base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=base_dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    clean_base.eval()

    d_model        = clean_base.config.hidden_size
    num_layers     = clean_base.config.num_hidden_layers
    num_kv_heads   = clean_base.config.num_key_value_heads
    num_attn_heads = clean_base.config.num_attention_heads
    head_dim       = d_model // num_attn_heads
    assert sum(boundary_sizes) == num_layers, (
        f"sum(boundary_sizes)={sum(boundary_sizes)} != num_hidden_layers={num_layers}"
    )

    second_moments = _collect_second_moments(
        clean_base, cal_loader, num_layers, device, collect_internal=True,
    )
    residual_salient = select_salient_channels(
        second_moments, d_model, boundary_sizes,
        top_k_ratio=top_k_ratio, group_k=group_k,
        group_size=group_size, outlier_log_sigma=outlier_sigma,
    )
    num_segments = len(boundary_sizes)
    segment_perms = {
        k: _build_segment_perm(residual_salient[k], d_model) for k in range(num_segments)
    }
    internal_salient = select_internal_salient_channels(second_moments, num_layers, group_k=group_k)

    del clean_base, second_moments
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------------
    # 2) Merge the plain QLoRA adapter into a dense fp16 base (ORIGINAL channel order).
    #    dequant(NF4 base) + B@A * scaling — the deployed weight the adapter was trained for.
    # ---------------------------------------------------------------------------------
    print("\n[2/5] Merging the QLoRA adapter into a dense fp16 base (original order) ...")
    import bitsandbytes as bnb  # noqa: F401  (ensures bnb is importable before NF4 load)
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"].get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg["model"].get("double_quant", True),
    )
    base_nf4 = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True,
    )
    peft_model = _load_adapter_for_export(base_nf4, args.checkpoint_dir, cfg)

    merged = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    count = _merge_lora_into_dense(
        peft_model, merged, target_modules, lora_scaling, qat_mode="none", group_size=group_size,
    )
    print(f"  Merged {count} LoRA layers into the dense base.")
    del peft_model, base_nf4
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------------
    # 3) Apply the three equivalence transforms to the MERGED dense weights.
    #    Pure equivalence transform on the merged weight => + boundary gathers it reproduces
    #    the original merged model. (Hadamard skipped for GQA; Llama-2-7b is MHA so it runs.)
    # ---------------------------------------------------------------------------------
    print("\n[3/5] Applying P_k / P4 / Hadamard equivalence transforms to the merged weights ...")
    merged.to(device)
    boundary_perms = apply_segment_permutation_fp32(merged, segment_perms, boundary_sizes)
    block_internal = apply_block_internal_permutations_fp32(merged, internal_salient)
    apply_hadamard_rotation_fp32(merged, num_layers, num_kv_heads, head_dim)

    bli = _boundary_layer_indices(boundary_sizes)
    perm_meta = {
        "boundary_perms":         [bp.cpu() for bp in boundary_perms],
        "boundary_layer_indices": bli,
        "segment_perms":          {k: list(v) for k, v in segment_perms.items()},
        "block_internal_perms":   {f"{k[0]}_{k[1]}": v for k, v in block_internal.items()},
        "group_k":                group_k,
        "group_size":             group_size,
        "boundary_sizes":         list(boundary_sizes),
        "d_model":                d_model,
        # This ablation does NOT use AWQ-style salient scaling (salient is fp16, no quant grid).
        "awq_scale":              False,
        # Carry q_bits/symmetric for downstream readers (mirrors the trained export meta).
        "q_bits":                 q_bits,
        "symmetric":              symmetric,
        "ablation":               "permute_fp16_salient",
    }

    # ---------------------------------------------------------------------------------
    # 4) PTQ: salient slice [0:group_k] kept fp16; non-salient [group_k:] GPTQ'd.
    #    o_proj has no salient slice (group_k=0 inside the sequential quantizer) -> fully GPTQ.
    #    Boundary gathers are registered inside the quantizer (perm_meta) so activations/Hessians
    #    are captured in the deployment basis.
    # ---------------------------------------------------------------------------------
    print("\n[4/5] GPTQ on non-salient columns (salient slice kept fp16) ...")
    gptq_loader = build_calibration_loader(cfg, tokenizer)
    merged.to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gptq_quantize_model_sequential(
        merged, gptq_loader, target_modules,
        perm_group_k=group_k, group_size=group_size, q_bits=q_bits,
        symmetric=symmetric, device=device, perm_meta=perm_meta,
        percdamp=percdamp, blocksize=blocksize, nsamples=nsamples,
        awq_scales=None,            # ablation: no AWQ scaling
        keep_salient_fp16=True,     # ablation: salient slice stays full precision
    )
    merged.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------------
    # 5) Save dense model (weights already deployed in place) + tokenizer + perm_meta.
    # ---------------------------------------------------------------------------------
    print(f"\n[5/5] Saving dense export + {PERM_META_FILENAME} -> {output_dir}")
    save_dtype = torch.float16 if args.dequant_dtype == "float16" else torch.bfloat16
    save_dequantized_model(merged, tokenizer, output_dir, cfg, dtype=save_dtype)
    # Save in the {"layers", "model"} wrapper shape the readers expect (they unwrap "model").
    save_sqat_permute_meta({"layers": {}, "model": perm_meta}, output_dir)

    print("\n" + "=" * 72)
    print("  Ablation export complete.")
    print(f"  Model + {PERM_META_FILENAME} -> {output_dir}")
    print("  Evaluate with the usual eval scripts (they auto-register the boundary gathers):")
    print(f"    python scripts/eval_math.py --model_path {output_dir} --num_fewshot 5 "
          f"--output_dir results/math")
    print("=" * 72)


if __name__ == "__main__":
    main()
