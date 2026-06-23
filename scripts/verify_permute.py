#!/usr/bin/env python
"""
verify_permute.py — Stage-1 equivalence + AWQ-S fusion verification for permuted SQAT.

Stage-1 (permutation equivalence): applies the OFFLINE equivalence transforms on a plain
fp32 model and checks that the permuted model (with runtime boundary gathers) reproduces the
original logits:
  (A) residual-stream permutation P_k       (+ num_segments-1 boundary gathers)
  (B) MLP block-internal permutation P4_l    (down_proj salient-first)

Stage-1b (AWQ-S fusion, optional --awq_scale): builds the AWQ-style per-channel salient scales S
from the SAME calibration and checks, on the permuted weights, that the fusion is mathematically
correct:
  * grid:  the amplified-space TRAIN fakequant grid == the EXPORT quantize→dequant→/S grid on each
           salient slice (verify_permute_quant_consistency with awq_s ≈ 0). This is the property
           that makes the /S bake-back deploy bit-identically to training.
  * effect: S genuinely changes the quant grid vs no-scaling (so it is not a silent no-op).
  * shape:  S is [num_layers, group_k] per source, S ∈ [1, max], per-row min == 1.

No NF4, no LoRA, no training — this only validates that the permutation math is closed and that
the AWQ-S amplify/bake-back fusion is self-consistent.

Usage:
  python scripts/verify_permute.py --model_name meta-llama/Llama-2-7b-hf --boundary_sizes 2 30
  python scripts/verify_permute.py --boundary_sizes 2 30 --awq_scale --group_size 64
  python scripts/verify_permute.py --num_boundaries 4 --group_k 128 --no_awq_scale
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.qat_permute_sqat import (
    build_and_verify_permutation_fp32,
    compute_awq_scales,
    awq_s_for_module,
    expand_group_ids_to_indices,
    _layer_idx_from_module_name,
    verify_permute_quant_consistency,
    group_fakequant,
)

_AWQ_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


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


def _verify_awq_fusion(permuted_model, artifacts, boundary_sizes, group_k, group_size,
                       q_bits, symmetric, alpha, max_s, tol=1e-4):
    """
    Stage-1b: build S from the calibration intermediates and check the amplify/bake-back fusion is
    self-consistent on the (already permuted) fp32 weights. Returns (worst_grid, worst_effect).
    Raises SystemExit(1) on a grid mismatch (the train/export grids must be identical).
    """
    num_layers = artifacts["num_layers"]
    print("\n[verify_permute] Stage 1b: AWQ-S fusion check "
          f"(alpha={alpha}, max={max_s}, group_size={group_size}, "
          f"q_bits={q_bits}, symmetric={symmetric}) ...")

    awq_scales = compute_awq_scales(
        artifacts["second_moments"], artifacts["residual_salient"],
        artifacts["internal_salient"], boundary_sizes, num_layers, group_k, group_size,
        o_proj_salient_group_ids=artifacts.get("o_proj_salient_group_ids"),
        alpha=alpha, max_s=max_s,
    )

    # (1) shape / range / normalization of S
    for src in ("attn", "mlp", "down", "o"):
        S = awq_scales[src]
        assert S.shape == (num_layers, group_k), f"{src} S shape {tuple(S.shape)}"
        assert (S >= 1.0 - 1e-5).all() and (S <= max_s + 1e-5).all(), f"{src} S out of [1,{max_s}]"
        assert torch.allclose(S.min(dim=1).values, torch.ones(num_layers), atol=1e-4), \
            f"{src} S per-row min != 1"
    print(f"  [AWQ] scales OK: per-source shape [L={num_layers}, gk={group_k}], "
          f"S in [1, {max_s}], per-row min == 1")

    # (2) per-layer: amplified train grid == export quantize→dequant→/S grid; S is non-trivial
    worst_grid, worst_effect, n = 0.0, 0.0, 0
    for name, m in permuted_model.named_modules():
        if not isinstance(m, nn.Linear) or name.split(".")[-1] not in _AWQ_TARGETS:
            continue
        S = awq_s_for_module(awq_scales, name, group_k)
        if S is None:
            continue
        if name.split(".")[-1] == "o_proj":
            lidx = _layer_idx_from_module_name(name)
            if lidx is None:
                continue
            gids = artifacts["o_proj_salient_group_ids"][lidx]
            idx = expand_group_ids_to_indices(gids, group_size, device=m.weight.device)
            W = m.weight.detach().index_select(1, idx).float().cpu()
        else:
            W = m.weight.detach()[:, :group_k].float().cpu()
        s = S.view(1, -1)
        grid_err = verify_permute_quant_consistency(W, group_k, group_size, q_bits, symmetric, awq_s=S)
        amp   = group_fakequant(W * s, group_size, q_bits, symmetric) / s
        plain = group_fakequant(W,      group_size, q_bits, symmetric)
        worst_grid   = max(worst_grid,   grid_err)
        worst_effect = max(worst_effect, (amp - plain).abs().max().item())
        n += 1
    print(f"  [AWQ] checked {n} salient slices: "
          f"train<->export grid worst max|Δ|={worst_grid:.2e}, "
          f"S-vs-noS worst max|Δ|={worst_effect:.2e}")
    # ASCII-only machine-parseable summary (for run_validation.sh)
    print(f"  [AWQ] METRICS grid={worst_grid:.3e} effect={worst_effect:.3e}")

    if worst_grid >= tol:
        print(f"  [AWQ] ❌ FUSION FAILED: amplified-space train/export grid mismatch "
              f"(worst={worst_grid:.2e} >= {tol:.0e}). The /S bake-back will NOT deploy "
              f"bit-identically to training.")
        raise SystemExit(1)
    if worst_effect < tol:
        print("  [AWQ] ⚠ grid consistent but S ≈ identity (no amplification effect) — "
              "check alpha/max or the calibration.")
    else:
        print("  [AWQ] ✓ FUSION OK: amplified-space train == export grid, and S is non-trivial.")
    return worst_grid, worst_effect


def main():
    parser = argparse.ArgumentParser(description="Stage-1 permutation + AWQ-S fusion verification")
    parser.add_argument("--model_name",  type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--n_samples",   type=int,   default=512)
    parser.add_argument("--seq_len",     type=int,   default=2048)
    parser.add_argument("--dataset",     type=str,   default="wikitext",
                        choices=["wikitext", "metamath", "commonsense"])
    parser.add_argument("--group_k",     type=int,   default=128)
    parser.add_argument("--group_size",  type=int,   default=128)
    parser.add_argument("--top_k_ratio", type=float, default=0.01)
    parser.add_argument("--tol",         type=float, default=1e-3)
    # AWQ-S fusion check (Stage 1b)
    parser.add_argument("--awq_scale",    dest="awq_scale", action="store_true", default=True,
                        help="Run the Stage-1b AWQ-S amplify/bake-back fusion check (default on).")
    parser.add_argument("--no_awq_scale", dest="awq_scale", action="store_false",
                        help="Skip the AWQ-S fusion check (permutation equivalence only).")
    parser.add_argument("--awq_alpha",  type=float, default=0.5)
    parser.add_argument("--awq_max",    type=float, default=2.0)
    parser.add_argument("--q_bits",     type=int,   default=4, choices=[3, 4])
    parser.add_argument("--symmetric",  dest="symmetric", action="store_true", default=False,
                        help="AWQ fusion check uses symmetric quant (default: asymmetric).")
    parser.add_argument("--asymmetric", dest="symmetric", action="store_false")
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

    assert args.group_k % args.group_size == 0, \
        f"group_k={args.group_k} must be a multiple of group_size={args.group_size}"

    print(f"[verify_permute] boundary_sizes={boundary_sizes}, "
          f"num_segments={len(boundary_sizes)}, group_k={args.group_k}, "
          f"group_size={args.group_size}, awq_scale={args.awq_scale}")

    cal_samples = _load_calibration(tokenizer, args.n_samples, args.seq_len, args.dataset)
    cal_loader  = DataLoader(cal_samples, batch_size=1, shuffle=False)

    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 1969, Neil Armstrong became the first human to walk on the Moon.",
        "The transformer architecture was introduced in Attention Is All You Need.",
        "Machine learning models can learn patterns from large datasets.",
    ]
    test_batches = _build_test_batches(tokenizer, test_prompts, 64, device)

    max_err, artifacts = build_and_verify_permutation_fp32(
        fp32_model,
        calibration_dataloader=cal_loader,
        boundary_sizes=boundary_sizes,
        test_inputs=test_batches,
        group_k=args.group_k,
        group_size=args.group_size,
        top_k_ratio=args.top_k_ratio,
        tol=args.tol,
        return_artifacts=True,
    )

    # Stage 1b — AWQ-S amplify/bake-back fusion (uses the same calibration + permuted weights)
    if args.awq_scale:
        _verify_awq_fusion(
            fp32_model, artifacts, boundary_sizes,
            group_k=args.group_k, group_size=args.group_size,
            q_bits=args.q_bits, symmetric=args.symmetric,
            alpha=args.awq_alpha, max_s=args.awq_max,
        )

    print("\n[verify_permute] Equivalence PASSED "
          f"(max_abs_logit_err={max_err:.2e}; "
          f"AWQ-S fusion {'checked' if args.awq_scale else 'skipped'}).")


if __name__ == "__main__":
    main()
