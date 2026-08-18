#!/usr/bin/env python
"""
Correctness test for permute_common.fold_boundary_gathers_into_weights.

The fold is what makes a SALT-Q / SQAT-Permute export loadable by an external runtime: it
removes the BoundaryGatherHooks by putting the residual channels back in their original
order. Getting the index direction wrong there is silent — the model still runs, still
produces fluent-looking text, and just scores lower — so it is checked here against a
reference forward pass rather than by inspection.

Three assertions, on a tiny random Llama (seconds, CPU, fp32):

  1. permuted weights + boundary hooks  ==  the original model   (the property the export
     relies on; if this fails the permutation itself is broken, not the fold)
  2. permuted weights, hooks REMOVED    !=  the original model   (proves the hooks are load
     bearing, so that test 3 cannot pass vacuously)
  3. folded weights, no hooks           ==  the original model

Run: python scripts/test_unpermute_fold.py
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import LlamaConfig, LlamaForCausalLM

from src.permute_common import (
    apply_segment_permutation_fp32,
    fold_boundary_gathers_into_weights,
    register_boundary_gathers,
)

BOUNDARY_SIZES = [2, 1, 2]      # 3 segments over 5 layers -> 2 runtime gathers
D_MODEL = 64


def build_tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=D_MODEL,
        intermediate_size=128,
        num_hidden_layers=sum(BOUNDARY_SIZES),
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        # lm_head and embed_tokens take DIFFERENT permutations (P_0 vs P_last) in multi-segment
        # mode, so they must not share storage — apply_segment_permutation_fp32 refuses ties.
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model


def logits(model, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def main() -> int:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (2, 16))

    reference_model = build_tiny_model()
    reference = logits(reference_model, input_ids)

    model = copy.deepcopy(reference_model)

    # A distinct random permutation per segment — the general case the fold has to handle.
    segment_perms = {
        k: torch.randperm(D_MODEL).tolist() for k in range(len(BOUNDARY_SIZES))
    }
    boundary_perms = apply_segment_permutation_fp32(model, segment_perms, BOUNDARY_SIZES)
    meta = {
        "segment_perms": segment_perms,
        "boundary_sizes": BOUNDARY_SIZES,
        "d_model": D_MODEL,
        "boundary_perms": boundary_perms,
    }

    # --- 1. permuted + hooks == original ------------------------------------------------
    hooks = register_boundary_gathers(
        model, boundary_perms, [1, 2], D_MODEL,   # last layer index of each non-final segment
    )
    err_hooked = (logits(model, input_ids) - reference).abs().max().item()
    print(f"[1] permuted + boundary hooks      max|Δlogits| = {err_hooked:.3e}")
    assert err_hooked < 1e-4, f"permutation is not equivalence-preserving ({err_hooked:.3e})"

    # --- 2. permuted, hooks removed != original -----------------------------------------
    for h in hooks:
        h.remove()
    err_unhooked = (logits(model, input_ids) - reference).abs().max().item()
    print(f"[2] permuted, hooks removed        max|Δlogits| = {err_unhooked:.3e}  (must be large)")
    assert err_unhooked > 1e-2, (
        "removing the boundary hooks changed nothing, so test 3 would pass even if the fold "
        "were a no-op. The test model is not exercising multi-segment permutation."
    )

    # --- 3. folded, no hooks == original -------------------------------------------------
    fold_boundary_gathers_into_weights(model, meta)
    err_folded = (logits(model, input_ids) - reference).abs().max().item()
    print(f"[3] folded weights, no hooks       max|Δlogits| = {err_folded:.3e}")
    assert err_folded < 1e-4, f"fold is not equivalence-preserving ({err_folded:.3e})"

    print("\nPASS — the folded model needs no runtime hook and matches the hooked model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
