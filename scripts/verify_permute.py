#!/usr/bin/env python
"""
verify_permute.py — Stage-1 equivalence verification for permuted SQAT.

Applies the three OFFLINE equivalence transforms on a plain fp32 model and checks that the
permuted model (with runtime boundary gathers) reproduces the original logits:
  (A) residual-stream permutation P_k       (+ num_segments-1 boundary gathers)
  (B) MLP block-internal permutation P4_l    (down_proj salient-first)
  (C) per-head Hadamard rotation H on v/o_proj

No NF4, no LoRA, no training — this only validates that the permutation math is closed.

Usage:
  python scripts/verify_permute.py --model_name meta-llama/Llama-2-7b-hf --boundary_sizes 2 30
  python scripts/verify_permute.py --num_boundaries 4 --group_k 128 --n_samples 32
"""

import argparse
import importlib.util
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.qat_permute_sqat import build_and_verify_permutation_fp32


def _load_calibration(tokenizer, n_samples, seq_len, dataset):
    """Reuse analyze_boundary_salient_channels.load_calibration_data(tokenizer, n, seq, name)."""
    path = os.path.join(_ROOT, "analyze_boundary_salient_channels.py")
    spec = importlib.util.spec_from_file_location("_abc", path)
    abc  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abc)
    return abc.load_calibration_data(tokenizer, n_samples, seq_len, dataset)


def _build_test_batches(tokenizer, prompts, seq_len, device):
    return [
        {k: v.to(device)
         for k, v in tokenizer(p, return_tensors="pt", truncation=True,
                               max_length=seq_len, padding=False).items()}
        for p in prompts
    ]


def main():
    parser = argparse.ArgumentParser(description="Stage-1 permutation equivalence verification")
    parser.add_argument("--model_name",  type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--n_samples",   type=int,   default=512)
    parser.add_argument("--seq_len",     type=int,   default=2048)
    parser.add_argument("--dataset",     type=str,   default="wikitext",
                        choices=["wikitext", "metamath", "commonsense"])
    parser.add_argument("--group_k",     type=int,   default=128)
    parser.add_argument("--group_size",  type=int,   default=128)
    parser.add_argument("--top_k_ratio", type=float, default=0.01)
    parser.add_argument("--tol",         type=float, default=1e-3)
    bnd = parser.add_mutually_exclusive_group()
    bnd.add_argument("--boundary_sizes", type=int, nargs="+", metavar="N")
    bnd.add_argument("--num_boundaries", type=int, default=2)
    args = parser.parse_args()

    print(f"[verify_permute] Loading {args.model_name} in fp32 ...")
    tokenizer  = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    fp32_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map="auto"
    )
    fp32_model.eval()

    num_layers  = fp32_model.config.num_hidden_layers
    device      = next(fp32_model.parameters()).device

    if args.boundary_sizes is not None:
        boundary_sizes = args.boundary_sizes
        assert sum(boundary_sizes) == num_layers, (
            f"sum(boundary_sizes)={sum(boundary_sizes)} != num_layers={num_layers}"
        )
    else:
        nb = args.num_boundaries
        assert num_layers % nb == 0, f"num_layers={num_layers} not divisible by {nb}"
        boundary_sizes = [num_layers // nb] * nb

    print(f"[verify_permute] boundary_sizes={boundary_sizes}, "
          f"num_segments={len(boundary_sizes)}, group_k={args.group_k}, "
          f"group_size={args.group_size}")

    cal_samples = _load_calibration(tokenizer, args.n_samples, args.seq_len, args.dataset)
    cal_loader  = DataLoader(cal_samples, batch_size=1, shuffle=False)

    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 1969, Neil Armstrong became the first human to walk on the Moon.",
        "The transformer architecture was introduced in Attention Is All You Need.",
        "Machine learning models can learn patterns from large datasets.",
    ]
    test_batches = _build_test_batches(tokenizer, test_prompts, 64, device)

    build_and_verify_permutation_fp32(
        fp32_model,
        calibration_dataloader=cal_loader,
        boundary_sizes=boundary_sizes,
        test_inputs=test_batches,
        group_k=args.group_k,
        group_size=args.group_size,
        top_k_ratio=args.top_k_ratio,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
