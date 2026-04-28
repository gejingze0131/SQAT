#!/usr/bin/env python3
"""
Standalone exporter for three evaluation targets:

1) Baseline Llama-2-7B (no export needed; we record the model id in a manifest)
2) Naive INT3 -> dequant export starting from an NF4 bitsandbytes load of Llama-2-7B
3) GPTQ INT3 -> dequant export using Hugging Face Transformers + GPTQModel

Typical usage
-------------
# Baseline manifest + naive NF4->INT3(dequant) + GPTQ INT3(dequant)
python export_llama2_int3_dequant.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --output-root outputs/llama2_int3_exports \
  --mode all

# Only naive export
python export_llama2_int3_dequant.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --output-root outputs/llama2_int3_exports \
  --mode naive

# Only GPTQ export, using a local calibration file
python export_llama2_int3_dequant.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --output-root outputs/llama2_int3_exports \
  --mode gptq \
  --gptq-dataset-file /path/to/calibration.jsonl \
  --gptq-text-column text

Notes
-----
- This script is for export only. Run your own eval script on:
  * manifest['baseline_model_id']
  * manifest['naive_nf4_int3_dequant_dir']
  * manifest['gptq_int3_dequant_dir']
- The naive path intentionally starts from an NF4 bitsandbytes load, dequantizes it,
  then applies a simple symmetric per-group INT3 quantize+dequantize pass to linear weights.
- The GPTQ path quantizes directly from the original HF model id using GPTQConfig,
  saves an optional GPTQ checkpoint, dequantizes in-memory, and saves a dense fp16/bf16 model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    set_seed,
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline / naive INT3-dequant / GPTQ INT3-dequant for Llama-2-7B.")

    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["baseline", "naive", "gptq", "all"], default="all")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tokenizer", action="store_true", help="Save tokenizer into exported directories.")

    # General export dtype
    parser.add_argument("--dense-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")

    # Naive NF4 -> INT3 -> dequant export
    parser.add_argument("--naive-bits", type=int, default=3)
    parser.add_argument("--naive-group-size", type=int, default=128)
    parser.add_argument("--naive-scheme", type=str, choices=["symmetric", "asymmetric"], default="symmetric",
                        help="Naive quantization scheme for the NF4->INT3->dequant path.")
    parser.add_argument("--naive-device-map", type=str, default="auto")
    parser.add_argument("--naive-double-quant", action="store_true", default=True)
    parser.add_argument("--no-naive-double-quant", dest="naive_double_quant", action="store_false")
    parser.add_argument("--naive-skip-modules", type=str, default="lm_head", help="Comma-separated substrings of module names to skip in naive quantization.")
    parser.add_argument("--naive-bnb-compute-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")

    # GPTQ export
    parser.add_argument("--gptq-bits", type=int, default=3)
    parser.add_argument("--gptq-group-size", type=int, default=128)
    parser.add_argument("--gptq-device-map", type=str, default="auto")
    parser.add_argument("--gptq-backend", type=str, default=None, help="GPTQ backend. Leave unset for default backend selection.")
    parser.add_argument("--gptq-dataset", type=str, default="c4", help="Built-in GPTQ dataset: c4, c4-new, or wikitext2.")
    parser.add_argument("--gptq-dataset-file", type=str, default=None, help="Optional local .txt/.jsonl/.json calibration corpus.")
    parser.add_argument("--gptq-text-column", type=str, default="text")
    parser.add_argument("--gptq-num-samples", type=int, default=128)
    parser.add_argument("--gptq-batch-size", type=int, default=1)
    parser.add_argument("--gptq-model-seqlen", type=int, default=2048)
    parser.add_argument("--gptq-damp-percent", type=float, default=0.1)
    parser.add_argument("--gptq-desc-act", action="store_true")
    parser.add_argument("--gptq-no-sym", dest="gptq_sym", action="store_false")
    parser.add_argument("--gptq-sym", dest="gptq_sym", action="store_true", default=True)
    parser.add_argument("--gptq-no-true-sequential", dest="gptq_true_sequential", action="store_false")
    parser.add_argument("--gptq-true-sequential", dest="gptq_true_sequential", action="store_true", default=True)
    parser.add_argument("--gptq-max-memory", type=str, default=None, help='JSON dict passed to from_pretrained, e.g. {"0": "40GiB", "cpu": "120GiB"}')
    parser.add_argument("--save-gptq-quantized", action="store_true", help="Also save the intermediate GPTQ quantized checkpoint before dequantization.")

    return parser.parse_args()



def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]



def parse_device_map(arg: str):
    if arg in {"auto", "cpu", "cuda", "balanced", "balanced_low_0", "sequential"}:
        return arg
    try:
        return json.loads(arg)
    except Exception:
        return arg



def parse_max_memory(arg: Optional[str]) -> Optional[dict]:
    if not arg:
        return None
    data = json.loads(arg)
    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out



def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id



def is_skipped_module(name: str, skip_tokens: Iterable[str]) -> bool:
    return any(tok and tok in name for tok in skip_tokens)


# -----------------------------------------------------------------------------
# Naive INT3 quantize -> dequant
# -----------------------------------------------------------------------------


def symmetric_quant_dequant_per_group(
    weight: torch.Tensor,
    bits: int = 3,
    group_size: int = 128,
) -> torch.Tensor:
    """Naive symmetric per-group quantize+dequant on the input-channel axis."""
    assert weight.ndim == 2, f"Expected 2D weight, got shape {tuple(weight.shape)}"
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    qmax = 2 ** (bits - 1) - 1
    if qmax <= 0:
        raise ValueError(f"Invalid qmax for bits={bits}")

    orig_dtype = weight.dtype
    w = weight.detach().float().cpu()
    out_features, in_features = w.shape
    num_groups = math.ceil(in_features / group_size)
    pad = num_groups * group_size - in_features

    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))

    w_grouped = w.view(out_features, num_groups, group_size)
    scales = (w_grouped.abs().amax(dim=2, keepdim=True) / qmax).clamp(min=1e-8)
    w_q = torch.round(torch.clamp(w_grouped / scales, -qmax, qmax))
    w_dq = (w_q * scales).view(out_features, -1)[:, :in_features].contiguous()

    return w_dq.to(orig_dtype)


def asymmetric_quant_dequant_per_group(
    weight: torch.Tensor,
    bits: int = 3,
    group_size: int = 128,
) -> torch.Tensor:
    """Naive asymmetric per-group quantize+dequant on the input-channel axis."""
    assert weight.ndim == 2, f"Expected 2D weight, got shape {tuple(weight.shape)}"
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    qmin = 0
    qmax = 2 ** bits - 1

    orig_dtype = weight.dtype
    w = weight.detach().float().cpu()
    out_features, in_features = w.shape
    num_groups = math.ceil(in_features / group_size)
    pad = num_groups * group_size - in_features

    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))

    w_grouped = w.view(out_features, num_groups, group_size)
    w_min = w_grouped.amin(dim=2, keepdim=True)
    w_max = w_grouped.amax(dim=2, keepdim=True)

    scales = ((w_max - w_min) / max(qmax - qmin, 1)).clamp(min=1e-8)
    zero_points = torch.round(qmin - w_min / scales).clamp(qmin, qmax)

    w_q = torch.round(w_grouped / scales + zero_points).clamp(qmin, qmax)
    w_dq = ((w_q - zero_points) * scales).view(out_features, -1)[:, :in_features].contiguous()

    return w_dq.to(orig_dtype)


def naive_quant_dequant_per_group(
    weight: torch.Tensor,
    bits: int = 3,
    group_size: int = 128,
    scheme: str = "symmetric",
) -> torch.Tensor:
    if scheme == "symmetric":
        return symmetric_quant_dequant_per_group(weight, bits=bits, group_size=group_size)
    if scheme == "asymmetric":
        return asymmetric_quant_dequant_per_group(weight, bits=bits, group_size=group_size)
    raise ValueError(f"Unsupported naive quantization scheme: {scheme}")


@torch.no_grad()
def export_naive_nf4_to_int3_dequant(args: argparse.Namespace, manifest: dict) -> Path:
    out_dir = Path(args.output_root) / "naive_nf4_to_int3_dequant"
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_dtype = get_torch_dtype(args.dense_dtype)
    bnb_compute_dtype = get_torch_dtype(args.naive_bnb_compute_dtype)
    device_map = parse_device_map(args.naive_device_map)
    skip_tokens = [x.strip() for x in args.naive_skip_modules.split(",") if x.strip()]

    print("=" * 80)
    print("[Naive] Loading tokenizer")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    ensure_pad_token(tokenizer)

    print("=" * 80)
    print("[Naive] Loading NF4 bitsandbytes model")
    print("=" * 80)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=args.naive_double_quant,
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        torch_dtype=dense_dtype,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("=" * 80)
    print("[Naive] Dequantizing NF4 model to dense weights")
    print("=" * 80)
    model.dequantize()
    model.to("cpu")

    print("=" * 80)
    print(f"[Naive] Applying {args.naive_scheme} INT{args.naive_bits} quantize+dequant to nn.Linear weights")
    print("=" * 80)
    touched, skipped = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight is None or module.weight.ndim != 2:
            continue
        if is_skipped_module(name, skip_tokens):
            skipped += 1
            continue
        module.weight.data = naive_quant_dequant_per_group(
            module.weight.data,
            bits=args.naive_bits,
            group_size=args.naive_group_size,
            scheme=args.naive_scheme,
        )
        touched += 1
        if module.bias is not None:
            module.bias.data = module.bias.data.to(dense_dtype).cpu()

    print(f"[Naive] Quantized+dequantized {touched} linear layers; skipped {skipped} layers")

    model.config.torch_dtype = str(dense_dtype).replace("torch.", "")
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    print("=" * 80)
    print(f"[Naive] Saving dense export to: {out_dir}")
    print("=" * 80)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    if args.save_tokenizer:
        tokenizer.save_pretrained(out_dir)

    manifest["naive_nf4_int3_dequant_dir"] = str(out_dir)
    manifest["naive_nf4_int3_dequant_scheme"] = args.naive_scheme
    manifest["naive_nf4_int3_dequant_bits"] = args.naive_bits
    manifest["naive_nf4_int3_dequant_group_size"] = args.naive_group_size
    return out_dir


# -----------------------------------------------------------------------------
# GPTQ INT3 -> dequant export
# -----------------------------------------------------------------------------


def load_calibration_texts_from_file(path: str, text_column: str, max_samples: int) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    texts: List[str] = []
    suffix = p.suffix.lower()

    if suffix == ".txt":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    if len(texts) >= max_samples:
                        break
        return texts

    if suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if text_column in obj and obj[text_column]:
                    texts.append(str(obj[text_column]))
                    if len(texts) >= max_samples:
                        break
        return texts

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and text_column in item and item[text_column]:
                    texts.append(str(item[text_column]))
                if len(texts) >= max_samples:
                    break
            return texts
        raise ValueError(f"Unsupported JSON structure in {path}; expected a list of strings or dicts.")

    raise ValueError(f"Unsupported calibration file format: {path}")


@torch.no_grad()
def export_gptq_int3_to_dequant(args: argparse.Namespace, manifest: dict) -> Path:
    out_dir = Path(args.output_root) / "gptq_int3_dequant"
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_dtype = get_torch_dtype(args.dense_dtype)
    device_map = parse_device_map(args.gptq_device_map)
    max_memory = parse_max_memory(args.gptq_max_memory)

    print("=" * 80)
    print("[GPTQ] Loading tokenizer")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    ensure_pad_token(tokenizer)

    dataset_obj: str | List[str]
    if args.gptq_dataset_file:
        dataset_obj = load_calibration_texts_from_file(
            args.gptq_dataset_file,
            text_column=args.gptq_text_column,
            max_samples=args.gptq_num_samples,
        )
        print(f"[GPTQ] Loaded {len(dataset_obj)} calibration texts from {args.gptq_dataset_file}")
    else:
        dataset_obj = args.gptq_dataset
        print(f"[GPTQ] Using built-in GPTQ dataset: {dataset_obj}")

    gptq_kwargs = dict(
        bits=args.gptq_bits,
        tokenizer=tokenizer,
        dataset=dataset_obj,
        group_size=args.gptq_group_size,
        damp_percent=args.gptq_damp_percent,
        desc_act=args.gptq_desc_act,
        sym=args.gptq_sym,
        true_sequential=args.gptq_true_sequential,
        model_seqlen=args.gptq_model_seqlen,
        batch_size=args.gptq_batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.gptq_backend:
        gptq_kwargs["backend"] = args.gptq_backend
    gptq_config = GPTQConfig(**gptq_kwargs)

    print("=" * 80)
    print("[GPTQ] Quantizing model with GPTQConfig")
    print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        quantization_config=gptq_config,
        torch_dtype=dense_dtype,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    model.eval()

    if args.save_gptq_quantized:
        quant_dir = Path(args.output_root) / "gptq_int3_quantized"
        quant_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"[GPTQ] Saving intermediate quantized checkpoint to: {quant_dir}")
        print("=" * 80)
        model.to("cpu")
        model.save_pretrained(quant_dir, safe_serialization=True, max_shard_size="5GB")
        if args.save_tokenizer:
            tokenizer.save_pretrained(quant_dir)
        # Move back only if needed; for dequant we can proceed on CPU.

    print("=" * 80)
    print("[GPTQ] Dequantizing quantized model to dense weights")
    print("=" * 80)
    model.dequantize(dtype=dense_dtype)
    model.to("cpu")

    model.config.torch_dtype = str(dense_dtype).replace("torch.", "")
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    print("=" * 80)
    print(f"[GPTQ] Saving dense dequantized export to: {out_dir}")
    print("=" * 80)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    if args.save_tokenizer:
        tokenizer.save_pretrained(out_dir)

    manifest["gptq_int3_dequant_dir"] = str(out_dir)
    return out_dir


# -----------------------------------------------------------------------------
# Baseline manifest only
# -----------------------------------------------------------------------------


def record_baseline(args: argparse.Namespace, manifest: dict) -> None:
    manifest["baseline_model_id"] = args.model_id
    manifest["baseline_note"] = (
        "Use the original HF model id directly for baseline evaluation. "
        "No export is required unless you want to snapshot the full fp16 model separately."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_id": args.model_id,
        "mode": args.mode,
        "dense_dtype": args.dense_dtype,
        "seed": args.seed,
    }

    record_baseline(args, manifest)

    if args.mode in {"naive", "all"}:
        export_naive_nf4_to_int3_dequant(args, manifest)

    if args.mode in {"gptq", "all"}:
        export_gptq_int3_to_dequant(args, manifest)

    manifest_path = output_root / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Done. Manifest written to: {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
