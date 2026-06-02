"""
sqat_permute.py — Segment-Shared Input-Channel Permutation + Selective QAT

Three orthogonal offline transforms (all equivalence transforms; model output unchanged):

  (A) Residual-stream permutation P_k (per-segment, boundary gathers required):
      Top group_k salient d_model channels → physical positions [0, group_k).
      Source: q_proj.input (attn) + gate_proj.input (mlp) per segment.
      Covers: q/k/v/gate/up_proj input cols + o/down_proj output rows + LNs + embed/lm_head.
      Cost: num_segments-1 runtime index_select gathers on residual (unavoidable).

  (B) MLP block-internal permutation P4_l (per-layer, no runtime gather):
      Top group_k salient channels in down_proj input space (intermediate_dim).
      Folded into: gate_proj + up_proj output rows + down_proj input cols.
      Source: down_proj.input[0] per layer.
      down_proj is then wrapped with PermutedSelectiveQATLinear (QAT on salient slice).

  (C) Attention Hadamard rotation H (per-layer per-head, no runtime gather):
      Normalized Walsh-Hadamard matrix H ∈ R^{head_dim × head_dim} applied per head.
      Folded into: v_proj output rows × H  (per-head block),
                   o_proj input cols  × H  (per-head block; H is self-inverse for WHT).
      NO QAT on o_proj — rotation alone guarantees PTQ lower bound (SpinQuant-style).
      Skipped for GQA models (v_proj.out_features ≠ o_proj.in_features).

Stage 1: --test_permute: verify all three transforms are closed on plain fp32 (no NF4/LoRA).
Stage 2: NF4 requant + QLoRA + PermutedSelectiveQATLinear on q/k/v/gate/up/down_proj.
         o_proj: Hadamard rotation only, no QAT (LoRA fine-tunes in rotated space).

TODO: KV cache inference: boundary gathers fire every decode token. TPOT impact not
evaluated here; profile separately before production deployment.

TODO: Tied lm_head/embed_tokens is safe for single-segment (P_0 == P_last). For
multi-segment it is detected and raises RuntimeError with instructions to untie.
"""

import argparse
import copy
import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.qat_base import (
    QATHandler,
    groupwise_asymmetric_fakequant,
)
from src.qat_sqat import dequantize_layer


# ============================================================================
# Part 1 — Calibration (one forward pass, all sources)
# ============================================================================

@torch.no_grad()
def _collect_second_moments(
    model: nn.Module,
    calibration_data,
    num_layers: int,
    device: torch.device,
    collect_internal: bool = True,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Collect per-channel E[x²] for residual-stream and (optionally) block-internal sources.

    Residual-stream sources (d_model dim):
      (l, 'attn')  — q_proj.input[0]     = input_layernorm output
      (l, 'mlp')   — gate_proj.input[0]  = post_attention_layernorm output

    Block-internal sources (intermediate dims, only when collect_internal=True):
      (l, 'o_proj')   — o_proj.input[0]    = reshaped attention output [num_heads*head_dim]
      (l, 'down_proj')— down_proj.input[0] = act_fn(gate)*up output  [intermediate_dim]

    Float32 accumulators on CPU; lazy initialisation on first token.
    Returns {key: tensor[feat_dim]}.
    """
    sum_sq:    Dict[Tuple[int, str], torch.Tensor] = {}
    tok_count: Dict[Tuple[int, str], int] = {}
    handles = []

    def _make_hook(key: Tuple[int, str]):
        def _hook(module, inp, out):
            x = inp[0].detach()
            feat = x.shape[-1]
            x_flat = x.reshape(-1, feat).float().cpu()
            if key not in sum_sq:
                sum_sq[key]    = torch.zeros(feat, dtype=torch.float32)
                tok_count[key] = 0
            sum_sq[key].add_(x_flat.pow(2).sum(dim=0))
            tok_count[key] += x_flat.shape[0]
        return _hook

    layers = model.model.layers
    for l in range(num_layers):
        handles.append(layers[l].self_attn.q_proj.register_forward_hook(
            _make_hook((l, "attn"))))
        handles.append(layers[l].mlp.gate_proj.register_forward_hook(
            _make_hook((l, "mlp"))))
        if collect_internal:
            handles.append(layers[l].self_attn.o_proj.register_forward_hook(
                _make_hook((l, "o_proj"))))
            handles.append(layers[l].mlp.down_proj.register_forward_hook(
                _make_hook((l, "down_proj"))))

    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(calibration_data, desc="[SegPerm] Calibrating E[x²]"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    return {key: sum_sq[key] / max(tok_count[key], 1) for key in sum_sq}


# ============================================================================
# Part 2 — Salient Channel Selection
# ============================================================================

def select_salient_channels(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    boundary_sizes: List[int],
    top_k_ratio: float = 0.01,
    group_k: int = 128,
    group_size: int = 128,
    outlier_log_sigma: float = 3.0,
) -> Dict[int, List[int]]:
    """
    Select top group_k salient RESIDUAL-STREAM channels per segment.

    Algorithm (replicates analyze_boundary_salient_channels.py):
      Sources: (l, 'attn') and (l, 'mlp') within each segment's layers.
      1. Per-source normalize: sm / sm.max()
      2. Per-source outlier set: log(sm) > mean + outlier_log_sigma * std
      3. Segment outlier = union of per-source outlier sets
      4. Aggregate = sum of normalized vectors
      5. Fill group_k: outliers first (desc agg), then non-outlier by agg

    Returns:
        {segment_idx: sorted List[int]} in d_model coordinate system.
    """
    assert group_k % group_size == 0, \
        f"group_k={group_k} must be a multiple of group_size={group_size}"

    top_k_per_source = max(1, round(top_k_ratio * hidden_size))
    b_offsets    = [0] + list(itertools.accumulate(boundary_sizes))
    num_segments = len(boundary_sizes)
    result: Dict[int, List[int]] = {}

    for seg in range(num_segments):
        b_start, b_end = b_offsets[seg], b_offsets[seg + 1]
        b_sources = [(l, s) for l in range(b_start, b_end) for s in ("attn", "mlp")]

        # Per-source normalize
        normalized = {}
        for key in b_sources:
            val = second_moments[key]
            mx  = val.max().item()
            normalized[key] = val / mx if mx > 0 else val.clone()

        # Per-source outlier sets
        outlier_sets = {}
        for key in b_sources:
            val   = second_moments[key]
            log_v = torch.log(val.clamp(min=1e-30))
            thr   = log_v.mean().item() + outlier_log_sigma * log_v.std().item()
            outlier_sets[key] = torch.where(log_v > thr)[0]

        # Aggregate score
        agg = torch.zeros(hidden_size, dtype=torch.float32)
        for key in b_sources:
            agg.add_(normalized[key])

        # Segment outlier union
        seg_outlier = torch.zeros(hidden_size, dtype=torch.bool)
        for key in b_sources:
            seg_outlier[outlier_sets[key]] = True

        sorted_by_score = torch.argsort(agg, descending=True)

        selected: List[int] = []
        sel_set:  set       = set()

        for idx in sorted_by_score.tolist():      # tier-1: outliers
            if len(selected) >= group_k:
                break
            if seg_outlier[idx].item() and idx not in sel_set:
                selected.append(idx); sel_set.add(idx)

        for idx in sorted_by_score.tolist():      # tier-2: top energy
            if len(selected) >= group_k:
                break
            if idx not in sel_set:
                selected.append(idx); sel_set.add(idx)

        salient = sorted(selected)
        result[seg] = salient

        sel_t      = torch.tensor(salient, dtype=torch.long)
        e_total    = sum(second_moments[k].sum().item() for k in b_sources)
        e_sel      = sum(second_moments[k][sel_t].sum().item() for k in b_sources)
        n_outliers = int(seg_outlier.sum().item())
        print(
            f"[SegPerm] Seg {seg} (L{b_start}-L{b_end-1}): "
            f"group_k={group_k}, outliers={n_outliers}, "
            f"energy_cov={e_sel/(e_total+1e-30)*100:.1f}%, "
            f"first10={salient[:10]}"
        )

    return result


def select_internal_salient_channels(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    num_layers: int,
    group_k: int = 128,
) -> Dict[Tuple[int, str], List[int]]:
    """
    Per-block salient channel selection for down_proj only → full permutations.

    down_proj (P4_l):
        Flat top-group_k by E[x^2] across all intermediate_dim channels.
        Arbitrary cross-channel permutation is valid (element-wise MLP, no multi-head).
        Returns full permutation of length down_proj_in_features (salient first).

    o_proj: handled by Hadamard rotation (see apply_hadamard_rotation_*).
            NOT included here — arbitrary permutation breaks multi-head attention,
            and within-head permutation gives only group_k//num_heads channels per head
            which is too few to improve quantization meaningfully.

    Returns:
        {(l, 'down_proj'): full permutation List[int]}  (length = intermediate_dim)
    """
    result: Dict[Tuple[int, str], List[int]] = {}
    for l in range(num_layers):
        key_d = (l, "down_proj")
        if key_d in second_moments:
            sm      = second_moments[key_d]
            down_in = sm.shape[0]
            k       = min(group_k, down_in)
            result[key_d] = _build_segment_perm(sm.topk(k).indices.tolist(), down_in)
    return result



# ============================================================================
# Part 3 — Permutation Construction
# ============================================================================

def _build_segment_perm(salient_channels: List[int], total_dim: int) -> List[int]:
    """
    Build permutation P: salient_channels first [0..group_k), then rest in original order.
    Works for both d_model (residual) and intermediate dims (o_proj/down_proj).
    """
    sal_set   = set(salient_channels)
    remaining = [c for c in range(total_dim) if c not in sal_set]
    return list(salient_channels) + remaining


def _compute_boundary_perm(
    P_k: List[int], P_kp1: List[int], d_model: int,
) -> torch.LongTensor:
    """
    Compute composite permutation P_{k+1} ∘ P_k^{-1} offline.
    Applied at runtime: hidden_states.index_select(-1, boundary_perm).

    The residual entering the boundary is in P_k order: physical position i holds
    original channel P_k[i]. We want output position i to hold original channel
    P_kp1[i]. Under index_select that means out[i] = hs[ inv_Pk[P_kp1[i]] ], i.e.

        boundary_perm = inv_Pk[P_kp1]          (gather indices into the P_k stream)

    NOTE: this is inv_Pk INDEXED BY P_kp1, *not* P_kp1 indexed by inv_Pk.
    The latter (P_kp1[inv_Pk]) is the inverse composition and silently corrupts
    the residual at every boundary — verified to blow Stage-1 logit error to O(10).
    """
    inv_Pk                    = torch.zeros(d_model, dtype=torch.long)
    inv_Pk[torch.tensor(P_k)] = torch.arange(d_model, dtype=torch.long)
    return inv_Pk[torch.tensor(P_kp1, dtype=torch.long)]


def _boundary_layer_indices(boundary_sizes: List[int]) -> List[int]:
    """Last decoder layer index of each non-final segment (for hook registration)."""
    cumsum = list(itertools.accumulate(boundary_sizes))
    return [c - 1 for c in cumsum[:-1]]


# ============================================================================
# Part 4 — Shared Permutation Helper
# ============================================================================

def _perm_tensor(t: torch.Tensor, perm: torch.Tensor, dim: int) -> torch.Tensor:
    idx = perm.to(t.device)
    return (t[idx] if dim == 0 else t[:, idx]).contiguous()


# ============================================================================
# Part 5 — Offline Weight Permutation: Stage 1 (plain fp32)
# ============================================================================

def _apply_residual_perm_fp32(
    model: nn.Module, seg: int, block_indices: range, P_k: torch.Tensor,
) -> None:
    """
    Apply residual-stream permutation P_k to one segment (fp32 model).

    Input cols (dim 1) permuted by P_k:  q/k/v_proj, gate_proj, up_proj
    Output rows (dim 0) permuted by P_k: o_proj, down_proj
    1-D vector permuted:                 input_layernorm, post_attention_layernorm
    """
    for l in block_indices:
        attn = model.model.layers[l].self_attn
        mlp  = model.model.layers[l].mlp
        ln   = model.model.layers[l]

        ln.input_layernorm.weight.data           = ln.input_layernorm.weight.data[P_k]
        ln.post_attention_layernorm.weight.data  = ln.post_attention_layernorm.weight.data[P_k]

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj,
                     mlp.gate_proj, mlp.up_proj):
            proj.weight.data = _perm_tensor(proj.weight.data, P_k, dim=1)   # input cols

        for proj in (attn.o_proj, mlp.down_proj):
            proj.weight.data = _perm_tensor(proj.weight.data, P_k, dim=0)   # output rows


def apply_segment_permutation_fp32(
    model: nn.Module,
    segment_perms: Dict[int, List[int]],
    boundary_sizes: List[int],
) -> List[torch.LongTensor]:
    """
    In-place residual-stream weight permutation for a PLAIN fp32 model (Stage 1).
    No BnB NF4, no LoRA — verifies permutation math only.

    Returns boundary_perms[k] = P_{k+1} ∘ P_k^{-1}, length = num_segments-1.
    """
    num_segments = len(boundary_sizes)
    d_model      = model.config.hidden_size
    b_offsets    = [0] + list(itertools.accumulate(boundary_sizes))

    if num_segments > 1 and hasattr(model, "lm_head"):
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError(
                "lm_head and embed_tokens are weight-tied but use different "
                "permutations (P_0 vs P_last) in multi-segment mode. "
                "Set tie_word_embeddings=False and clone lm_head.weight."
            )

    P = {k: torch.tensor(v, dtype=torch.long) for k, v in segment_perms.items()}
    P_0    = P[0]
    P_last = P[num_segments - 1]

    # embed_tokens: d_model cols (dim 1) → P_0
    model.model.embed_tokens.weight.data = _perm_tensor(
        model.model.embed_tokens.weight.data, P_0, dim=1
    )

    for seg in range(num_segments):
        _apply_residual_perm_fp32(
            model, seg, range(b_offsets[seg], b_offsets[seg + 1]), P[seg]
        )

    model.model.norm.weight.data = model.model.norm.weight.data[P_last]
    model.lm_head.weight.data    = _perm_tensor(model.lm_head.weight.data, P_last, dim=1)

    boundary_perms = [
        _compute_boundary_perm(segment_perms[k], segment_perms[k + 1], d_model)
        for k in range(num_segments - 1)
    ]
    assert len(boundary_perms) == num_segments - 1   # §11
    print(f"[SegPerm] Permuted fp32 residual stream: {num_segments} segs, "
          f"num_runtime_permutes={len(boundary_perms)}")
    return boundary_perms


def apply_block_internal_permutations_fp32(
    model: nn.Module,
    internal_salient: Dict[Tuple[int, str], List[int]],
) -> Dict[Tuple[int, str], List[int]]:
    """
    Apply MLP block-internal P4_l permutation in-place on plain fp32 model.

    P4_l (down_proj input permutation, fully offline fold):
      gate_proj output rows (dim 0) ← P4_l
      up_proj   output rows (dim 0) ← P4_l
      down_proj input cols  (dim 1) ← P4_l
    No runtime gather needed.

    o_proj is handled separately by apply_hadamard_rotation_* (no permutation here).

    Returns:
        applied_perms: {(l, 'down_proj'): P4_l}
    """
    num_layers    = model.config.num_hidden_layers
    applied_perms: Dict[Tuple[int, str], List[int]] = {}

    for l in range(num_layers):
        mlp   = model.model.layers[l].mlp
        key_d = (l, "down_proj")
        if key_d not in internal_salient:
            continue
        P4_l    = internal_salient[key_d]
        down_in = mlp.down_proj.weight.shape[1]
        if len(P4_l) != down_in:
            print(f"[SegPerm] Layer {l}: down_proj dim mismatch "
                  f"(perm_len={len(P4_l)} != down_in={down_in}). Skipping P4.")
            continue
        P4_t = torch.tensor(P4_l, dtype=torch.long)
        mlp.gate_proj.weight.data = _perm_tensor(mlp.gate_proj.weight.data, P4_t, dim=0)
        mlp.up_proj.weight.data   = _perm_tensor(mlp.up_proj.weight.data,   P4_t, dim=0)
        mlp.down_proj.weight.data = _perm_tensor(mlp.down_proj.weight.data, P4_t, dim=1)
        applied_perms[key_d] = P4_l

    print(f"[SegPerm] Applied P4 (down_proj) permutations: {len(applied_perms)} layers")
    return applied_perms


# ============================================================================
# Part 6 — Offline Weight Permutation: Stage 2 (NF4 + LoRA)
# ============================================================================

def _permute_peft_input_cols(module: nn.Module, P_t: torch.Tensor) -> None:
    """Permute NF4 base weight input cols (dim 1) and LoRA A input cols."""
    import bitsandbytes as bnb

    W = dequantize_layer(module).float()
    W_perm = W[:, P_t.to(W.device)].contiguous().float()

    if (hasattr(module, "base_layer") and
            hasattr(module.base_layer.weight, "quant_state") and
            module.base_layer.weight.quant_state is not None):
        W_q, qs = bnb.functional.quantize_4bit(
            W_perm, quant_type="nf4", compress_statistics=True
        )
        module.base_layer.weight = bnb.nn.Params4bit(
            data=W_q, requires_grad=False, quant_state=qs
        )
    else:
        base = module.base_layer if hasattr(module, "base_layer") else module
        base.weight.data = W_perm

    if hasattr(module, "lora_A"):
        for name in module.lora_A:
            A = module.lora_A[name].weight.data          # [rank, in]
            module.lora_A[name].weight.data = A[:, P_t.to(A.device)].contiguous()


def _permute_peft_output_rows(module: nn.Module, P_t: torch.Tensor) -> None:
    """Permute NF4 base weight output rows (dim 0) and LoRA B output rows."""
    import bitsandbytes as bnb

    W = dequantize_layer(module).float()
    W_perm = W[P_t.to(W.device), :].contiguous().float()

    if (hasattr(module, "base_layer") and
            hasattr(module.base_layer.weight, "quant_state") and
            module.base_layer.weight.quant_state is not None):
        W_q, qs = bnb.functional.quantize_4bit(
            W_perm, quant_type="nf4", compress_statistics=True
        )
        module.base_layer.weight = bnb.nn.Params4bit(
            data=W_q, requires_grad=False, quant_state=qs
        )
    else:
        base = module.base_layer if hasattr(module, "base_layer") else module
        base.weight.data = W_perm

    if hasattr(module, "lora_B"):
        for name in module.lora_B:
            B = module.lora_B[name].weight.data          # [out, rank]
            module.lora_B[name].weight.data = B[P_t.to(B.device), :].contiguous()


def apply_segment_permutation_nf4(
    model: nn.Module,
    segment_perms: Dict[int, List[int]],
    boundary_sizes: List[int],
) -> List[torch.LongTensor]:
    """
    In-place residual-stream permutation for NF4+LoRA PEFT model (Stage 2).
    NF4 base weights are dequantized, permuted, and re-quantized to NF4.
    Returns boundary_perms[k] = P_{k+1} ∘ P_k^{-1}.
    """
    num_segments = len(boundary_sizes)
    d_model      = model.config.hidden_size
    b_offsets    = [0] + list(itertools.accumulate(boundary_sizes))

    if num_segments > 1 and hasattr(model, "lm_head"):
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError(
                "Tied lm_head/embed_tokens with multi-segment: different permutations "
                "(P_0 vs P_last). Set tie_word_embeddings=False and clone lm_head.weight."
            )

    P = {k: torch.tensor(v, dtype=torch.long) for k, v in segment_perms.items()}
    P_0    = P[0]
    P_last = P[num_segments - 1]

    model.model.embed_tokens.weight.data = _perm_tensor(
        model.model.embed_tokens.weight.data, P_0, dim=1
    )

    for seg in range(num_segments):
        P_k   = P[seg]
        start = b_offsets[seg]
        end   = b_offsets[seg + 1]
        print(f"[SegPerm] NF4 residual perm seg {seg} (L{start}-L{end-1}) ...")
        for l in range(start, end):
            attn = model.model.layers[l].self_attn
            mlp  = model.model.layers[l].mlp
            ln   = model.model.layers[l]

            ln.input_layernorm.weight.data          = ln.input_layernorm.weight.data[P_k]
            ln.post_attention_layernorm.weight.data = ln.post_attention_layernorm.weight.data[P_k]

            for proj in (attn.q_proj, attn.k_proj, attn.v_proj,
                         mlp.gate_proj, mlp.up_proj):
                _permute_peft_input_cols(proj, P_k)    # input cols (dim 1)

            for proj in (attn.o_proj, mlp.down_proj):
                _permute_peft_output_rows(proj, P_k)   # output rows (dim 0)

    model.model.norm.weight.data = model.model.norm.weight.data[P_last]
    model.lm_head.weight.data    = _perm_tensor(model.lm_head.weight.data, P_last, dim=1)

    boundary_perms = [
        _compute_boundary_perm(segment_perms[k], segment_perms[k + 1], d_model)
        for k in range(num_segments - 1)
    ]
    assert len(boundary_perms) == num_segments - 1   # §11
    print(f"[SegPerm] NF4 residual perm done. num_runtime_permutes={len(boundary_perms)}")
    return boundary_perms


def apply_block_internal_permutations_nf4(
    model: nn.Module,
    internal_salient: Dict[Tuple[int, str], List[int]],
) -> Dict[Tuple[int, str], List[int]]:
    """
    Apply MLP block-internal P4_l permutation in-place on NF4+LoRA PEFT model.

    P4_l: gate_proj + up_proj output rows + down_proj input cols.
    o_proj is handled separately by apply_hadamard_rotation_nf4.

    Returns applied_perms: {(l, 'down_proj'): P4_l}
    """
    num_layers    = model.config.num_hidden_layers
    applied_perms: Dict[Tuple[int, str], List[int]] = {}

    for l in range(num_layers):
        mlp   = model.model.layers[l].mlp
        key_d = (l, "down_proj")
        if key_d not in internal_salient:
            continue
        P4_l   = internal_salient[key_d]
        base_d = mlp.down_proj.base_layer if hasattr(mlp.down_proj, "base_layer") else mlp.down_proj
        if len(P4_l) != base_d.in_features:
            print(f"[SegPerm] Layer {l}: down_proj dim mismatch. Skipping P4.")
            continue
        P4_t = torch.tensor(P4_l, dtype=torch.long)
        _permute_peft_output_rows(mlp.gate_proj, P4_t)
        _permute_peft_output_rows(mlp.up_proj,   P4_t)
        _permute_peft_input_cols(mlp.down_proj,  P4_t)
        applied_perms[key_d] = P4_l

    print(f"[SegPerm] NF4 P4 (down_proj) permutations: {len(applied_perms)} layers")
    return applied_perms


# ============================================================================
# Part 7 — Attention Hadamard Rotation (o_proj, per-layer per-head)
# ============================================================================

def _build_hadamard(n: int) -> torch.Tensor:
    """
    Normalized Walsh-Hadamard matrix of size n×n (n must be a power of 2).
    Satisfies H @ H = I (self-inverse), entries ±1/sqrt(n).
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be a power of 2"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return (H / math.sqrt(n)).float()


def apply_hadamard_rotation_fp32(
    model: nn.Module,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """
    Apply per-head Hadamard rotation to v_proj/o_proj in-place on plain fp32 model.

    Equivalence transform: for each KV head h,
      v_proj.weight[h*hd:(h+1)*hd, :] ← H @ v_proj.weight[...]   (H rotates v output)
      o_proj.weight[:, h*hd:(h+1)*hd] ← o_proj.weight[...] @ H   (H^{-1}=H cancels)

    H = normalized WHT of size head_dim (self-inverse, H^2 = I).
    Skipped for GQA (v_proj.out_features ≠ o_proj.in_features).

    Returns number of layers where rotation was applied.
    """
    H = _build_hadamard(head_dim).to(next(model.parameters()).device)
    rotated = 0
    for l in range(num_layers):
        attn  = model.model.layers[l].self_attn
        o_in  = attn.o_proj.weight.shape[1]   # [d_model, num_heads*head_dim]
        v_out = attn.v_proj.weight.shape[0]   # [num_kv_heads*head_dim, d_model]
        if v_out != o_in:
            continue   # GQA: skip
        H_l = H.to(attn.v_proj.weight.device)
        v_w  = attn.v_proj.weight.data.float()
        o_w  = attn.o_proj.weight.data.float()
        for h in range(num_kv_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            v_w[s:e, :] = H_l @ v_w[s:e, :]      # rotate v output rows per head
            o_w[:, s:e] = o_w[:, s:e] @ H_l       # H^{-1}=H, cancel rotation in o
        attn.v_proj.weight.data = v_w.float()
        attn.o_proj.weight.data = o_w.float()
        rotated += 1
    print(f"[SegPerm] Hadamard rotation applied to {rotated}/{num_layers} layers "
          f"(head_dim={head_dim}, num_kv_heads={num_kv_heads})")
    return rotated


def apply_hadamard_rotation_nf4(
    model: nn.Module,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """
    Apply per-head Hadamard rotation to v_proj/o_proj in NF4+LoRA PEFT model.
    Dequantizes NF4, applies H (per-head), re-quantizes to NF4.
    Also rotates LoRA A (v_proj) and LoRA B (o_proj) for consistency.
    Skipped for GQA.
    Returns number of layers rotated.
    """
    import bitsandbytes as bnb

    H = _build_hadamard(head_dim)
    rotated = 0

    for l in range(num_layers):
        attn = model.model.layers[l].self_attn

        base_v = attn.v_proj.base_layer if hasattr(attn.v_proj, "base_layer") else attn.v_proj
        base_o = attn.o_proj.base_layer if hasattr(attn.o_proj, "base_layer") else attn.o_proj
        if base_v.out_features != base_o.in_features:
            continue   # GQA

        H_l = H.to(next(model.parameters()).device)

        # ---- v_proj: rotate output rows per head ----
        Wv = dequantize_layer(attn.v_proj).float()
        for h in range(num_kv_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            Wv[s:e, :] = H_l @ Wv[s:e, :]
        Wv_h = Wv.float()
        if hasattr(attn.v_proj.base_layer.weight, "quant_state"):
            Wv_q, qs = bnb.functional.quantize_4bit(
                Wv_h, quant_type="nf4", compress_statistics=True
            )
            attn.v_proj.base_layer.weight = bnb.nn.Params4bit(
                data=Wv_q, requires_grad=False, quant_state=qs
            )
        else:
            attn.v_proj.base_layer.weight.data = Wv_h

        # LoRA B for v_proj: rotate output rows per head (B has shape [out, rank])
        if hasattr(attn.v_proj, "lora_B"):
            for name in attn.v_proj.lora_B:
                Bv = attn.v_proj.lora_B[name].weight.data.float()
                for h in range(num_kv_heads):
                    s, e = h * head_dim, (h + 1) * head_dim
                    Bv[s:e, :] = H_l @ Bv[s:e, :]
                attn.v_proj.lora_B[name].weight.data = Bv.float()

        # ---- o_proj: rotate input cols per head (H^{-1}=H) ----
        Wo = dequantize_layer(attn.o_proj).float()
        for h in range(num_kv_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            Wo[:, s:e] = Wo[:, s:e] @ H_l
        Wo_h = Wo.float()
        if hasattr(attn.o_proj.base_layer.weight, "quant_state"):
            Wo_q, qs = bnb.functional.quantize_4bit(
                Wo_h, quant_type="nf4", compress_statistics=True
            )
            attn.o_proj.base_layer.weight = bnb.nn.Params4bit(
                data=Wo_q, requires_grad=False, quant_state=qs
            )
        else:
            attn.o_proj.base_layer.weight.data = Wo_h

        # LoRA A for o_proj: rotate input cols per head (A has shape [rank, in])
        if hasattr(attn.o_proj, "lora_A"):
            for name in attn.o_proj.lora_A:
                Ao = attn.o_proj.lora_A[name].weight.data.float()
                for h in range(num_kv_heads):
                    s, e = h * head_dim, (h + 1) * head_dim
                    Ao[:, s:e] = Ao[:, s:e] @ H_l
                attn.o_proj.lora_A[name].weight.data = Ao.float()

        rotated += 1

    print(f"[SegPerm] NF4 Hadamard rotation: {rotated}/{num_layers} layers rotated")
    return rotated


# ============================================================================
# Part 8 — Boundary Gather Hooks (residual stream only)
# ============================================================================

class BoundaryGatherHook:
    """
    Persistent forward hook: permutes residual stream from P_k to P_{k+1} order
    after the last decoder layer of segment k.
    boundary_perm = P_{k+1} ∘ P_k^{-1}  (pre-computed offline)
    """

    def __init__(self, boundary_perm: torch.LongTensor, d_model: int):
        self._perm    = boundary_perm
        self._d_model = d_model
        self._handle  = None

    def register(self, decoder_layer: nn.Module) -> None:
        self._handle = decoder_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output):
        if isinstance(output, tuple):
            hs, *rest = output
        else:
            hs = output; rest = None

        assert hs.shape[-1] == self._d_model, (
            f"BoundaryGatherHook: expected hidden dim {self._d_model}, "
            f"got {hs.shape[-1]}. Check transformers version."
        )
        perm     = self._perm.to(hs.device)
        permuted = hs.index_select(-1, perm)
        return (permuted, *rest) if rest is not None else permuted

    def remove(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None


def register_boundary_gathers(
    model: nn.Module,
    boundary_perms: List[torch.LongTensor],
    boundary_layer_indices: List[int],
    d_model: int,
) -> List[BoundaryGatherHook]:
    """Register one hook per segment boundary (num_segments-1 total). Persistent."""
    assert len(boundary_perms) == len(boundary_layer_indices)
    hooks = []
    for k, (bp, li) in enumerate(zip(boundary_perms, boundary_layer_indices)):
        h = BoundaryGatherHook(bp, d_model)
        h.register(model.model.layers[li])
        hooks.append(h)
        print(f"[SegPerm] Boundary hook: seg {k}→{k+1} after layer {li}")
    print(f"[SegPerm] num_runtime_permutes={len(boundary_perms)}")
    assert len(boundary_perms) == len(boundary_layer_indices)   # §11
    return hooks


# ============================================================================
# Part 8 — Equivalence Verification (Stage 1, plain fp32 only)
# ============================================================================

@torch.no_grad()
def verify_permutation_equivalence(
    original_model: nn.Module,
    permuted_model: nn.Module,
    test_inputs: List[Dict[str, torch.Tensor]],
    tol: float = 1e-3,
    q_proj_tol: float = 1e-2,
) -> float:
    """
    Verify that permuted_model (all permutations applied + boundary gathers) is
    numerically equivalent to original_model on plain fp32.

    Checks:
      1. Final logits max-abs error < 0.1 (raises RuntimeError if violated)
      2. q_proj L0 output invariance: permutation must NOT change q output features

    NOTE on q_proj_tol: the plan called for a "bit-level" (<1e-5) q_proj check, but
    that is unachievable in fp32. P only touches q_proj INPUT cols, so q output is
    mathematically invariant — but the permuted path feeds q_proj a differently-
    ORDERED RMSNorm output (input_layernorm.weight and the residual are both in P_k
    order), and RMSNorm's mean(x²) reduction plus the 4096-dim matmul dot product are
    not order-invariant in fp32. This injects ~1e-3 rounding noise, the SAME source
    as the ~5e-2 final-logit noise. A genuine leak of P into output ROWS would mis-
    order q wholesale and produce O(1) error (empirically ~7.6), so q_proj_tol=1e-2
    cleanly separates "fp32 reduction noise" (pass) from "RoPE-breaking row leak"
    (fail) with several orders of magnitude of margin.

    Stage 1 ONLY (plain fp32 — NF4 requant noise would also break a tight q check).
    Returns max max-abs logit error across all inputs.
    """
    original_model.eval(); permuted_model.eval()
    device = next(original_model.parameters()).device

    q_orig_out: List[torch.Tensor] = []
    q_perm_out: List[torch.Tensor] = []

    def _qhook(store):
        def _h(m, i, o): store.append(o.detach().cpu())
        return _h

    h_o = original_model.model.layers[0].self_attn.q_proj.register_forward_hook(
        _qhook(q_orig_out))
    h_p = permuted_model.model.layers[0].self_attn.q_proj.register_forward_hook(
        _qhook(q_perm_out))

    max_err_global = 0.0
    try:
        for i, batch in enumerate(test_inputs):
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)

            lo = original_model(input_ids=ids, attention_mask=mask).logits.float().cpu()
            lp = permuted_model(input_ids=ids, attention_mask=mask).logits.float().cpu()

            err = (lo - lp).abs().max().item()
            max_err_global = max(max_err_global, err)
            status = "OK" if err < 0.1 else "FAIL"
            print(f"[SegPerm] Verify input {i}: max_abs_logit_err={err:.2e} [{status}]")
            if err > 0.1:
                raise RuntimeError(
                    f"Equivalence FAILED (input {i}): max_abs_err={err:.4f} > 0.1. "
                    "Permutation or boundary gather has a bug."
                )
    finally:
        h_o.remove(); h_p.remove()

    if q_orig_out and q_perm_out:
        qerr = (q_orig_out[0].float() - q_perm_out[0].float()).abs().max().item()
        print(f"[SegPerm] q_proj output max_abs_err={qerr:.2e} (tol={q_proj_tol:.2e})")
        if qerr > q_proj_tol:
            raise RuntimeError(
                f"q_proj output differs by {qerr:.4g} > {q_proj_tol:.4g}. "
                "P leaked into q_proj output rows — RoPE safety violated."
            )

    print(f"[SegPerm] Equivalence PASSED: max_logit_err={max_err_global:.2e} "
          f"over {len(test_inputs)} inputs")
    return max_err_global


# ============================================================================
# Part 9 — Stage 2: PermutedSelectiveQATLinear
# ============================================================================

class PermutedSelectiveQATLinear(nn.Module):
    """
    Selective QAT on physical input columns [0, group_k) of a permuted linear layer.

    Applied to ALL 7 target projections after permutation:
      Residual-stream permuted (input cols): q/k/v_proj, gate_proj, up_proj
      Block-internally permuted (input cols): o_proj, down_proj

    Forward:
      pass 1+2: full NF4 base + all-channel LoRA (original_module, unchanged)
      pass 3:   inject QAT residual for salient input slice [0:group_k] via STE

    Simpler than SelectiveSalientQATLinear: no base_max_group, no salient_gain —
    salient channels form complete quantization groups [0, group_k/group_size).
    Reuses groupwise_asymmetric_fakequant from qat_base.py.
    """

    def __init__(
        self,
        original_module: nn.Module,
        group_k: int,
        group_size: int,
        q_bits: int,
        lora_scaling: float,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"

        self.original_module = original_module
        self.group_k         = group_k
        self.group_size      = group_size
        self.q_max           = 2 ** q_bits - 1
        self.lora_scaling    = lora_scaling
        self.has_lora        = hasattr(original_module, "lora_A")

        W = dequantize_layer(original_module)
        self.register_buffer("W_base_salient", W[:, :group_k].detach())

    def _get_lora_A(self) -> torch.Tensor:
        name = list(self.original_module.lora_A.keys())[0]
        return self.original_module.lora_A[name].weight   # [rank, in]

    def _get_lora_B(self) -> torch.Tensor:
        name = list(self.original_module.lora_B.keys())[0]
        return self.original_module.lora_B[name].weight   # [out, rank]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        Y = self.original_module(x, *args, **kwargs)   # pass 1+2

        if not self.has_lora:
            return Y

        X_S = x[..., :self.group_k]
        A_S = self._get_lora_A()[:, :self.group_k]
        B   = self._get_lora_B()

        W_curr_S = (
            self.W_base_salient
            + (B.float() @ A_S.float()) * self.lora_scaling
        )

        W_fq_S  = groupwise_asymmetric_fakequant(
            W_curr_S.float(), self.group_size, self.q_max
        )
        delta_S = (W_fq_S - W_curr_S).to(x.dtype)

        return Y + F.linear(X_S, delta_S)


# ============================================================================
# Part 10 — QATHandler
# ============================================================================

class SegmentPermutedSelectiveQAT(QATHandler):
    """
    Segment-Shared Permutation QAT handler.

    prepare_model() orchestrates:
      1. One calibration pass → all E[x²] (residual + internal)
      2. select_salient_channels → segment perms P_k
      3. select_internal_salient_channels → per-block P2_l, P4_l
      4. apply_segment_permutation_nf4 → residual-stream weight permute
      5. apply_block_internal_permutations_nf4 → fold v/gate/up rows + o/down cols
      6. register_boundary_gathers → persistent hooks
      7. [stage>=2] Wrap all 7 projections with PermutedSelectiveQATLinear
    """

    def __init__(self):
        self.patched_layers:    Dict[str, PermutedSelectiveQATLinear] = {}
        self.boundary_hooks:    List[BoundaryGatherHook] = []
        self.boundary_perms:    List[torch.LongTensor] = []
        self.segment_perms:     Dict[int, List[int]] = {}
        self.block_internal_perms: Dict[Tuple[int, str], List[int]] = {}

    def prepare_model(
        self,
        model: nn.Module,
        cfg: dict,
        tokenizer=None,
        calibration_dataloader: Optional[DataLoader] = None,
        **kwargs,
    ) -> nn.Module:
        assert calibration_dataloader is not None, \
            "sqat_permute requires calibration_dataloader"

        sp_cfg         = cfg["qat"]["sqat_permute"]
        boundary_sizes = sp_cfg["boundary_sizes"]
        group_k        = sp_cfg.get("group_k", 128)
        group_size     = cfg["qat"].get("group_size", 128)
        top_k_ratio    = sp_cfg.get("top_k_ratio", 0.01)
        outlier_sigma  = sp_cfg.get("outlier_log_sigma", 3.0)
        stage          = sp_cfg.get("stage", 2)
        q_bits         = cfg["model"]["quant_bits"]
        lora_scaling   = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        target_modules = cfg["lora"]["target_modules"]

        device       = next(model.parameters()).device
        d_model      = model.config.hidden_size
        num_layers   = model.config.num_hidden_layers
        num_kv_heads = model.config.num_key_value_heads
        num_attn_heads = model.config.num_attention_heads
        head_dim     = d_model // num_attn_heads
        num_segments = len(boundary_sizes)

        # Step 1: One calibration pass — all sources
        second_moments = _collect_second_moments(
            model, calibration_dataloader, num_layers, device,
            collect_internal=True,
        )

        # Step 2a: Residual-stream salient channels (per-segment)
        residual_salient = select_salient_channels(
            second_moments, d_model, boundary_sizes,
            top_k_ratio=top_k_ratio, group_k=group_k,
            group_size=group_size, outlier_log_sigma=outlier_sigma,
        )
        self.segment_perms = {
            k: _build_segment_perm(residual_salient[k], d_model)
            for k in range(num_segments)
        }

        # Step 2b: down_proj internal salient channels (P4_l, flat)
        internal_salient = select_internal_salient_channels(
            second_moments, num_layers, group_k=group_k,
        )
        for (l, proj), perm in list(internal_salient.items())[:2]:
            print(f"[SegPerm] Internal P4 L{l} {proj}: perm_len={len(perm)}")

        # Step 3: Offline NF4 residual permutation
        self.boundary_perms = apply_segment_permutation_nf4(
            model, self.segment_perms, boundary_sizes
        )

        # Step 4: Offline NF4 MLP permutation (P4_l, down_proj)
        self.block_internal_perms = apply_block_internal_permutations_nf4(
            model, internal_salient
        )

        # Step 4b: Offline Hadamard rotation on v_proj/o_proj (SpinQuant-style)
        # Equivalence transform; no QAT on o_proj — rotation guarantees PTQ lower bound.
        apply_hadamard_rotation_nf4(model, num_layers, num_kv_heads, head_dim)

        # Step 5: Register boundary gathers (residual stream only)
        bli = _boundary_layer_indices(boundary_sizes)
        self.boundary_hooks = register_boundary_gathers(
            model, self.boundary_perms, bli, d_model
        )

        # Step 6: Stage 2 — wrap target projections with PermutedSelectiveQATLinear.
        # Residual-stream permuted (input cols [0:group_k] are salient):
        #   q/k/v_proj, gate_proj, up_proj
        # MLP block-internally permuted (input cols [0:group_k] are salient):
        #   down_proj
        # NOT wrapped: o_proj — Hadamard rotation only, no QAT (PTQ lower-bound only).
        _WRAP_PROJS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"}
        if stage >= 2:
            for name, module in model.named_modules():
                terminal = name.split(".")[-1]
                if terminal not in _WRAP_PROJS or terminal not in target_modules:
                    continue
                if not (hasattr(module, "base_layer") and hasattr(module, "lora_A")):
                    continue

                wrap = PermutedSelectiveQATLinear(
                    original_module=module,
                    group_k=group_k,
                    group_size=group_size,
                    q_bits=q_bits,
                    lora_scaling=lora_scaling,
                ).to(device)
                self.patched_layers[name] = wrap
                parts = name.rsplit(".", 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                setattr(parent, parts[-1], wrap)

            print(f"[SegPerm] Wrapped {len(self.patched_layers)} projections "
                  f"with PermutedSelectiveQATLinear")

        # Step 7: Attach export metadata
        model._sqat_permute_meta = {
            "boundary_perms":           [bp.cpu() for bp in self.boundary_perms],
            "boundary_layer_indices":   bli,
            "segment_perms":            dict(self.segment_perms),
            "block_internal_perms":     {
                f"{k[0]}_{k[1]}": v
                for k, v in self.block_internal_perms.items()
            },
            "group_k":                  group_k,
            "group_size":               group_size,
            "q_bits":                   q_bits,
            "boundary_sizes":           boundary_sizes,
            "d_model":                  d_model,
        }
        return model

    def on_train_begin(self, model: nn.Module): pass
    def on_step_end(self, model: nn.Module, step: int): pass
    def on_train_end(self, model: nn.Module): pass


# ============================================================================
# Part 11 — CLI (--test_permute = Stage 1 only)
# ============================================================================

def _build_test_batches(tokenizer, prompts, seq_len, device):
    return [
        {k: v.to(device)
         for k, v in tokenizer(
             p, return_tensors="pt", truncation=True,
             max_length=seq_len, padding=False
         ).items()}
        for p in prompts
    ]


def main():
    parser = argparse.ArgumentParser(description="SegPerm QAT — Stage 1 test + Stage 2 handler")
    parser.add_argument("--model_name",    type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--n_samples",     type=int,   default=512)
    parser.add_argument("--seq_len",       type=int,   default=2048)
    parser.add_argument("--dataset",       type=str,   default="wikitext",
                        choices=["wikitext", "metamath", "commonsense"])
    parser.add_argument("--group_k",       type=int,   default=128)
    parser.add_argument("--group_size",    type=int,   default=128)
    parser.add_argument("--top_k_ratio",   type=float, default=0.01)
    bnd = parser.add_mutually_exclusive_group()
    bnd.add_argument("--boundary_sizes",  type=int, nargs="+", metavar="N")
    bnd.add_argument("--num_boundaries",  type=int, default=2)
    parser.add_argument("--test_permute", action="store_true",
                        help="Stage 1 only: apply all permutations on fp32 and verify equivalence")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    import os as _os
    import importlib.util as _ilu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use analyze_boundary_salient_channels.load_calibration_data which accepts
    # (tokenizer, n_samples, seq_len, dataset_name) — same interface as the CLI args.
    _abc_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "analyze_boundary_salient_channels.py")
    _spec = _ilu.spec_from_file_location("_abc", _abc_path)
    _abc  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_abc)
    _load_cal = _abc.load_calibration_data

    print(f"[SegPerm] Loading {args.model_name} in fp32 ...")
    tokenizer  = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    fp32_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map="auto"
    )
    fp32_model.eval()

    num_layers   = fp32_model.config.num_hidden_layers
    hidden_size  = fp32_model.config.hidden_size
    device       = next(fp32_model.parameters()).device

    if args.boundary_sizes is not None:
        boundary_sizes = args.boundary_sizes
        assert sum(boundary_sizes) == num_layers, (
            f"sum(boundary_sizes)={sum(boundary_sizes)} != num_layers={num_layers}"
        )
    else:
        nb = args.num_boundaries
        assert num_layers % nb == 0, f"num_layers={num_layers} not divisible by {nb}"
        boundary_sizes = [num_layers // nb] * nb

    num_attn_heads = fp32_model.config.num_attention_heads
    head_dim       = hidden_size // num_attn_heads
    print(f"[SegPerm] boundary_sizes={boundary_sizes}, num_segments={len(boundary_sizes)}, "
          f"num_attn_heads={num_attn_heads}, head_dim={head_dim}")

    cal_samples = _load_cal(tokenizer, args.n_samples, args.seq_len, args.dataset)
    cal_loader  = DataLoader(cal_samples, batch_size=1, shuffle=False)

    if not args.test_permute:
        print("[SegPerm] For training, use SegmentPermutedSelectiveQAT via scripts/train.py")
        return

    # ---- Stage 1: fp32 only ----
    print("[SegPerm] Stage 1: collecting E[x²] (all sources) ...")
    second_moments = _collect_second_moments(
        fp32_model, cal_loader, num_layers, device, collect_internal=True
    )

    print("[SegPerm] Stage 1: selecting residual salient channels ...")
    residual_salient = select_salient_channels(
        second_moments, hidden_size, boundary_sizes,
        top_k_ratio=args.top_k_ratio, group_k=args.group_k,
        group_size=args.group_size,
    )
    segment_perms = {
        k: _build_segment_perm(residual_salient[k], hidden_size)
        for k in range(len(boundary_sizes))
    }

    print("[SegPerm] Stage 1: selecting down_proj internal salient channels (P4) ...")
    internal_salient = select_internal_salient_channels(
        second_moments, num_layers, group_k=args.group_k,
    )
    for (l, proj), perm in list(internal_salient.items())[:2]:
        print(f"  L{l} {proj}: perm_len={len(perm)}")

    print("[SegPerm] Stage 1: deep-copying fp32 model for comparison ...")
    original_fp32 = copy.deepcopy(fp32_model)

    print("[SegPerm] Stage 1: applying residual-stream fp32 permutation ...")
    boundary_perms = apply_segment_permutation_fp32(
        fp32_model, segment_perms, boundary_sizes
    )

    print("[SegPerm] Stage 1: applying MLP P4 fp32 permutations ...")
    applied_internal = apply_block_internal_permutations_fp32(
        fp32_model, internal_salient
    )

    num_kv_heads = fp32_model.config.num_key_value_heads
    print(f"[SegPerm] Stage 1: applying Hadamard rotation "
          f"(head_dim={head_dim}, num_kv_heads={num_kv_heads}) ...")
    apply_hadamard_rotation_fp32(fp32_model, num_layers, num_kv_heads, head_dim)

    bli   = _boundary_layer_indices(boundary_sizes)
    hooks = register_boundary_gathers(fp32_model, boundary_perms, bli, hidden_size)

    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 1969, Neil Armstrong became the first human to walk on the Moon.",
        "The transformer architecture was introduced in Attention Is All You Need.",
        "Machine learning models can learn patterns from large datasets.",
    ]
    test_batches = _build_test_batches(tokenizer, test_prompts, 64, device)

    print("[SegPerm] Stage 1: verifying equivalence (P_k + P4 + H all closed) ...")
    max_err = verify_permutation_equivalence(
        original_fp32, fp32_model, test_batches, tol=args.tol
    )

    print(
        f"\n[SegPerm] Stage 1 RESULT: "
        f"num_runtime_permutes={len(boundary_perms)}, "
        f"num_P4_perms={len(applied_internal)}, "
        f"num_H_layers={num_layers}, "
        f"max_abs_logit_err={max_err:.2e}"
    )
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()