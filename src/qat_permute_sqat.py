"""
qat_permute_sqat.py — Segment-shared input-channel permutation + block-level fused Selective QAT.

This single module owns the whole permuted-QLoRA Selective-QAT stack:

  Offline equivalence transforms (model output unchanged; applied on a clean fp16 base):
    (A) Residual-stream permutation P_k (per segment): top layer_group_k salient d_model channels →
        physical positions [0, layer_group_k). Covers q/k/v/gate/up input cols + o/down output rows
        + LNs + embed/lm_head. Needs num_segments-1 runtime boundary gathers on the residual.
    (B) MLP block-internal permutation P4_l (per layer): top layer_group_k salient down_proj input
        channels → [0, layer_group_k); folded into gate/up output rows + down input cols.
    (C) Per-head Hadamard rotation H on v_proj/o_proj (per layer): SpinQuant-style PTQ floor;
        no QAT on o_proj. Skipped for GQA.

  Stage-2 training (the from-scratch fused Selective-QAT forward):
    Fakequant is implemented HERE (not imported from qat_base): per-output-row, per-input-group,
    STE, on the permuted physical columns [0:layer_group_k] only. A `symmetric` flag selects the
    symmetric vs affine-asymmetric branch.

    Injection is BLOCK-LEVEL and FUSED via hooks (no forward replacement):
      - Attention: a forward_pre_hook on self_attn computes ONE fused delta for q/k/v
        (one BMM for B@A_S when shapes are uniform, one fused fakequant, one GEMM), splits it,
        and stashes the q/k/v slices; forward_hooks on q/k/v_proj add their slice.
      - MLP: gate/up_proj likewise → one fused delta, one GEMM, split + add.
      - down_proj: no sibling to fuse with → a single self-contained forward_hook (small GEMM).
    Net per block: 2 fused big GEMMs (QKV, GateUp) + 1 small GEMM (down) instead of 5 separate
    small delta GEMMs — fewer kernel launches, same QLoRA main path.

Hard invariants:
  * Never materialize the full merged weight W + B@A. Only the salient slice
    W_curr_S = W_base_salient + (B @ A_S) * lora_scaling  of shape [out, layer_group_k].
  * Never replace the BnB/QLoRA projection forward — deltas are ADDED via forward hooks.
  * layer_group_k % group_size == 0. No pre-permutation salient_idx, no index_select/gather on the
    salient slice. X_S = hidden_states[..., :layer_group_k].
  * QKV never share qparams; Gate/Up never share qparams — fakequant is per-output-row, so
    concatenating along the output dim before quantizing is identical to quantizing separately.

Quantize ONCE, not twice: the permute/fold runs on a clean fp16 base which is saved and reloaded
through the standard load_in_4bit path (build_permuted_fp16_checkpoint), so NF4 quantizes the
permuted weights exactly once (no dequant→permute→requant round-trip).

The boundary gather is NOT a weight transform — it is a runtime residual reorder that cannot be
folded offline (the skip connection has no weight carrier). It MUST be re-registered after reload
for training (prepare_model) AND on the exported model for inference (register_boundary_gathers_from_meta).

LoRA dropout: weight-level (W_base + B@A_S) injection equals the runtime LoRA forward only when
lora_dropout == 0. A non-zero dropout triggers a warning (recommend 0 for selective-QAT training).

Stage-1 equivalence verification (P_k + P4 + H all closed, plain fp32) lives in
scripts/verify_permute.py, which drives build_and_verify_permutation_fp32() here.
"""

import itertools
import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.qat_base import QATHandler
from src.qat_base import (
    groupwise_lsq_symmetric_fakequant as _lsq_sym_fq,
    groupwise_lsq_asym_fakequant as _lsq_asym_fq,
    init_lsq_scale_sym as _lsq_init_sym,
    init_lsq_scale_zp_asym as _lsq_init_asym,
    lsq_quantize_export_sym as _lsq_export_sym,
    lsq_quantize_export_asym as _lsq_export_asym,
)
from src.qat_sqat import dequantize_layer


# ============================================================================
# Part 0 — Model-structure resolver (robust to PEFT / CausalLM / base wrapping)
# ============================================================================

def _resolve_llama_model(model: nn.Module) -> nn.Module:
    """
    Return the inner module that owns `.layers` / `.embed_tokens` / `.norm` (the LlamaModel),
    unwrapping PEFT and the CausalLM head as needed.

    Handles:
      LlamaModel       → itself (already has .layers)
      LlamaForCausalLM → .model
      PeftModel        → .base_model.model(.model)  (LoraModel → CausalLM → LlamaModel)
    """
    obj = model
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model"):
        obj = obj.base_model.model          # PeftModel → LlamaForCausalLM
    for _ in range(4):
        if hasattr(obj, "layers"):
            return obj
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    raise AttributeError(
        f"Could not locate the decoder module (.layers) in model of type {type(model).__name__}"
    )


def _resolve_decoder_layers(model: nn.Module) -> nn.Module:
    """Return the decoder layer list (ModuleList of LlamaDecoderLayer)."""
    return _resolve_llama_model(model).layers


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

    layers = _resolve_decoder_layers(model)
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
# Part 2 — Salient channel selection
# ============================================================================

DEFAULT_GROUP_K_CANDIDATES: Tuple[int, ...] = (64, 128, 256)
TARGET_OUTLIER_CAPTURE = 1.0


def _normalize_group_k_candidates(
    group_k_candidates: Optional[Sequence[int]],
    group_size: int,
) -> List[int]:
    candidates = list(group_k_candidates or DEFAULT_GROUP_K_CANDIDATES)
    candidates = sorted({int(k) for k in candidates})
    if not candidates:
        raise ValueError("group_k_candidates must not be empty")
    bad = [k for k in candidates if k <= 0 or k % group_size != 0]
    if bad:
        raise ValueError(
            f"group_k_candidates={candidates} must be positive multiples of group_size={group_size}"
        )
    return candidates


def _boundary_offsets(boundary_sizes: Sequence[int]) -> List[int]:
    return [0] + list(itertools.accumulate(boundary_sizes))


def _unwrap_perm_model_meta(meta: Optional[dict]) -> Optional[dict]:
    """Accept both raw perm_meta and saved {"model": perm_meta} export metadata."""
    if not meta:
        return None
    if isinstance(meta, dict) and "boundary_perms" not in meta and "model" in meta:
        return meta["model"]
    return meta


def _expand_segment_group_ks(
    boundary_sizes: Sequence[int],
    segment_group_ks: Sequence[int],
) -> List[int]:
    assert len(boundary_sizes) == len(segment_group_ks), (
        f"len(boundary_sizes)={len(boundary_sizes)} != "
        f"len(segment_group_ks)={len(segment_group_ks)}"
    )
    out: List[int] = []
    for size, gk in zip(boundary_sizes, segment_group_ks):
        out.extend([int(gk)] * int(size))
    return out


def layer_group_ks_from_meta(meta: Optional[dict]) -> Optional[List[int]]:
    meta = _unwrap_perm_model_meta(meta)
    if not meta:
        return None
    if "layer_group_ks" in meta:
        return [int(x) for x in meta["layer_group_ks"]]
    if "boundary_sizes" in meta and "segment_group_ks" in meta:
        return _expand_segment_group_ks(meta["boundary_sizes"], meta["segment_group_ks"])
    if "boundary_sizes" in meta and "group_k" in meta:
        return [int(meta["group_k"])] * int(sum(meta["boundary_sizes"]))
    return None


def _layer_idx_from_module_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def group_k_for_module_name(
    name: str,
    perm_meta: Optional[dict] = None,
    default_group_k: Optional[int] = None,
) -> int:
    """Return the active salient slice width for a projection name."""
    terminal = name.split(".")[-1] if name else ""
    if terminal == "o_proj":
        return 0
    if default_group_k == 0:
        return 0
    layer_group_ks = layer_group_ks_from_meta(perm_meta)
    layer_idx = _layer_idx_from_module_name(name)
    if layer_group_ks is not None and layer_idx is not None and layer_idx < len(layer_group_ks):
        return int(layer_group_ks[layer_idx])
    if default_group_k is not None:
        return int(default_group_k)
    mm = _unwrap_perm_model_meta(perm_meta)
    if mm and "group_k" in mm:
        return int(mm["group_k"])
    return 0


def _normalized_residual_scores(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
) -> Dict[Tuple[int, str], torch.Tensor]:
    normalized: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, val in second_moments.items():
        if key[1] not in ("attn", "mlp"):
            continue
        mx = val.max().item()
        normalized[key] = val / mx if mx > 0 else val.clone()
    return normalized


def _source_outliers(val: torch.Tensor, outlier_log_sigma: float) -> torch.Tensor:
    log_v = torch.log(val.clamp(min=1e-30))
    thr = log_v.mean().item() + outlier_log_sigma * log_v.std(unbiased=False).item()
    return torch.where(log_v > thr)[0].to(torch.long)


def _residual_outlier_sets(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    outlier_log_sigma: float,
) -> Dict[Tuple[int, str], torch.Tensor]:
    return {
        key: _source_outliers(val, outlier_log_sigma)
        for key, val in second_moments.items()
        if key[1] in ("attn", "mlp")
    }


def _snap_group_k(required_count: int, candidates: Sequence[int]) -> int:
    for gk in candidates:
        if required_count <= gk:
            return int(gk)
    return int(candidates[-1])


def _segment_sources(start: int, end: int) -> List[Tuple[int, str]]:
    return [(l, s) for l in range(start, end) for s in ("attn", "mlp")]


def _segment_aggregate_score(
    b_sources: Sequence[Tuple[int, str]],
    normalized: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
) -> torch.Tensor:
    agg = torch.zeros(hidden_size, dtype=torch.float32)
    for key in b_sources:
        agg.add_(normalized[key])
    return agg


def _segment_outlier_mask(
    b_sources: Sequence[Tuple[int, str]],
    outlier_sets: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
) -> torch.Tensor:
    mask = torch.zeros(hidden_size, dtype=torch.bool)
    for key in b_sources:
        mask[outlier_sets[key]] = True
    return mask


def _select_bucket_from_mask(
    agg: torch.Tensor,
    outlier_mask: torch.Tensor,
    group_k: int,
) -> List[int]:
    sorted_by_score = torch.argsort(agg, descending=True).tolist()
    selected: List[int] = []
    selected_set = set()

    for idx in sorted_by_score:
        if len(selected) >= group_k:
            break
        if outlier_mask[idx].item() and idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)

    for idx in sorted_by_score:
        if len(selected) >= group_k:
            break
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)

    return sorted(selected)


def _segment_group_ks_for_boundaries(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    boundary_sizes: Sequence[int],
    group_k_candidates: Sequence[int],
    outlier_log_sigma: float,
) -> List[int]:
    outlier_sets = _residual_outlier_sets(second_moments, outlier_log_sigma)
    offsets = _boundary_offsets(boundary_sizes)
    out: List[int] = []
    for seg in range(len(boundary_sizes)):
        sources = _segment_sources(offsets[seg], offsets[seg + 1])
        mask = _segment_outlier_mask(sources, outlier_sets, hidden_size)
        union_count = int(mask.sum().item())
        gk = _snap_group_k(union_count, group_k_candidates)
        if union_count > gk:
            raise RuntimeError(
                f"Manual segment {seg} L{offsets[seg]}-{offsets[seg + 1] - 1} has "
                f"{union_count} true outliers, larger than max group_k candidate {gk}."
            )
        out.append(gk)
    return out


def auto_segment_by_outliers(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    num_layers: int,
    group_size: int,
    group_k_candidates: Optional[Sequence[int]] = None,
    max_segments: int = 4,
    outlier_log_sigma: float = 3.0,
) -> Tuple[List[int], List[int], Dict]:
    """
    Choose contiguous residual segments automatically.

    Hard constraint: every true per-source outlier, detected from that source's
    log(E[x^2]) distribution, must fit in its segment bucket. Objective under
    num_segments <= max_segments: minimize sum(segment_num_layers * group_k).
    """
    candidates = _normalize_group_k_candidates(group_k_candidates, group_size)
    max_segments = min(int(max_segments), int(num_layers))
    outlier_sets = _residual_outlier_sets(second_moments, outlier_log_sigma)

    seg_group_k: Dict[Tuple[int, int], int] = {}
    seg_union_count: Dict[Tuple[int, int], int] = {}
    feasible: Dict[Tuple[int, int], bool] = {}
    for start in range(num_layers):
        for end in range(start + 1, num_layers + 1):
            sources = _segment_sources(start, end)
            mask = _segment_outlier_mask(sources, outlier_sets, hidden_size)
            union_count = int(mask.sum().item())
            gk = _snap_group_k(union_count, candidates)
            key = (start, end)
            seg_group_k[key] = gk
            seg_union_count[key] = union_count
            feasible[key] = union_count <= gk

    inf = float("inf")
    dp = torch.full((max_segments + 1, num_layers + 1), inf, dtype=torch.float64)
    prev = torch.full((max_segments + 1, num_layers + 1), -1, dtype=torch.long)
    dp[0, 0] = 0.0
    for nseg in range(1, max_segments + 1):
        for end in range(1, num_layers + 1):
            for start in range(0, end):
                key = (start, end)
                if not feasible[key] or not torch.isfinite(dp[nseg - 1, start]):
                    continue
                cost = dp[nseg - 1, start] + (end - start) * seg_group_k[key]
                if cost < dp[nseg, end]:
                    dp[nseg, end] = cost
                    prev[nseg, end] = start

    curve = []
    best_nseg = -1
    best_cost = inf
    for nseg in range(1, max_segments + 1):
        ok = bool(torch.isfinite(dp[nseg, num_layers]))
        cost = float(dp[nseg, num_layers].item()) if ok else None
        curve.append({"num_segments": nseg, "feasible": ok, "selection_cost": cost})
        if ok and cost < best_cost - 1e-9:
            best_cost = float(cost)
            best_nseg = nseg
    if best_nseg < 0:
        raise RuntimeError(
            "No automatic SQAT segment partition can capture all true outliers with "
            f"max_segments={max_segments}, candidates={candidates}. Increase max_segments, "
            "increase group_k_candidates, or raise outlier_log_sigma."
        )

    ranges: List[Tuple[int, int]] = []
    end = num_layers
    for nseg in range(best_nseg, 0, -1):
        start = int(prev[nseg, end].item())
        if start < 0:
            raise RuntimeError("Automatic segment DP reconstruction failed.")
        ranges.append((start, end))
        end = start
    ranges.reverse()

    boundary_sizes = [end - start for start, end in ranges]
    segment_group_ks = [seg_group_k[(start, end)] for start, end in ranges]
    summary = {
        "rule": "minimize sum(segment_layers * group_k) subject to 100% true-outlier capture",
        "target_capture": TARGET_OUTLIER_CAPTURE,
        "outlier_log_sigma": float(outlier_log_sigma),
        "group_k_candidates": list(candidates),
        "max_segments": int(max_segments),
        "cost_curve": curve,
        "segments": [
            {
                "layers": [start, end - 1],
                "size": end - start,
                "group_k": seg_group_k[(start, end)],
                "outlier_union_count": seg_union_count[(start, end)],
                "headroom": seg_group_k[(start, end)] - seg_union_count[(start, end)],
            }
            for start, end in ranges
        ],
    }
    return boundary_sizes, segment_group_ks, summary

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

    Returns {segment_idx: sorted List[int]} in d_model coordinate system.
    """
    assert group_k % group_size == 0, \
        f"group_k={group_k} must be a multiple of group_size={group_size}"
    return select_salient_channels_variable(
        second_moments=second_moments,
        hidden_size=hidden_size,
        boundary_sizes=boundary_sizes,
        segment_group_ks=[int(group_k)] * len(boundary_sizes),
        group_size=group_size,
        outlier_log_sigma=outlier_log_sigma,
    )


def select_salient_channels_variable(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    boundary_sizes: List[int],
    segment_group_ks: Sequence[int],
    group_size: int = 128,
    outlier_log_sigma: float = 3.0,
) -> Dict[int, List[int]]:
    """Select salient residual channels with a potentially different group_k per segment."""
    assert len(boundary_sizes) == len(segment_group_ks), (
        f"len(boundary_sizes)={len(boundary_sizes)} != "
        f"len(segment_group_ks)={len(segment_group_ks)}"
    )
    bad = [int(k) for k in segment_group_ks if int(k) <= 0 or int(k) % group_size != 0]
    assert not bad, f"segment_group_ks must be positive multiples of group_size={group_size}: {bad}"

    b_offsets    = _boundary_offsets(boundary_sizes)
    result: Dict[int, List[int]] = {}
    normalized_all = _normalized_residual_scores(second_moments)
    outlier_sets_all = _residual_outlier_sets(second_moments, outlier_log_sigma)

    for seg in range(len(boundary_sizes)):
        b_start, b_end = b_offsets[seg], b_offsets[seg + 1]
        b_sources = _segment_sources(b_start, b_end)
        group_k = int(segment_group_ks[seg])

        agg = _segment_aggregate_score(b_sources, normalized_all, hidden_size)
        seg_outlier = _segment_outlier_mask(b_sources, outlier_sets_all, hidden_size)
        salient = _select_bucket_from_mask(agg, seg_outlier, group_k)
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
        if n_outliers > group_k:
            print(
                f"[SegPerm] WARNING: seg {seg} has {n_outliers} true outliers but "
                f"group_k={group_k}; only the highest aggregate-score outliers fit."
            )

    return result


def select_internal_salient_channels(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    num_layers: int,
    group_k: int = 128,
    layer_group_ks: Optional[Sequence[int]] = None,
) -> Dict[Tuple[int, str], List[int]]:
    """
    Per-block salient channel selection for down_proj only → full permutations.

    down_proj (P4_l): flat top-group_k by E[x^2] across all intermediate_dim channels.
    Arbitrary cross-channel permutation is valid (element-wise MLP, no multi-head).
    Returns full permutation of length down_proj_in_features (salient first).

    o_proj is handled by Hadamard rotation (apply_hadamard_rotation_fp32), not here.

    Returns {(l, 'down_proj'): full permutation List[int]} (length = intermediate_dim).
    """
    result: Dict[Tuple[int, str], List[int]] = {}
    for l in range(num_layers):
        key_d = (l, "down_proj")
        if key_d in second_moments:
            sm      = second_moments[key_d]
            down_in = sm.shape[0]
            layer_k = int(layer_group_ks[l]) if layer_group_ks is not None else int(group_k)
            k       = min(layer_k, down_in)
            result[key_d] = _build_segment_perm(sm.topk(k).indices.tolist(), down_in)
    return result


def compute_awq_scales(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    residual_salient: Dict[int, List[int]],
    internal_salient: Dict[Tuple[int, str], List[int]],
    boundary_sizes: List[int],
    num_layers: int,
    group_k: int,
    layer_group_ks: Optional[Sequence[int]] = None,
    alpha: float = 0.5,
    max_s: float = 2.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    AWQ-style per-input-channel scale S on the salient slice [0:layer_group_k], per (layer, source):
      attn (q/k/v share)   ← (l,'attn')      E[x²] at the segment's salient channels
      mlp  (gate/up share) ← (l,'mlp')       E[x²] at the same salient channels
      down (down_proj)     ← (l,'down_proj') E[x²] at the P4 salient channels
    S_j = (E[x²]_j)^alpha, normalized so min over the slice = 1, clamped to [1, max_s] (so S≥1,
    i.e. salient channels are only ever amplified). Returns {"attn"/"mlp"/"down": [L, max_group_k]}
    float32 (1.0 where a source is unavailable or where a layer's active group_k is smaller than
    max_group_k). Indexed by PERMUTED position (matches the slice).
    """
    b_off = [0] + list(itertools.accumulate(boundary_sizes))
    if layer_group_ks is None:
        layer_group_ks = [int(group_k)] * num_layers
    else:
        layer_group_ks = [int(x) for x in layer_group_ks]
        assert len(layer_group_ks) == num_layers, (
            f"len(layer_group_ks)={len(layer_group_ks)} != num_layers={num_layers}"
        )
    max_group_k = int(group_k)

    def seg_of(l: int) -> int:
        for s in range(len(boundary_sizes)):
            if b_off[s] <= l < b_off[s + 1]:
                return s
        return len(boundary_sizes) - 1

    def _scale_from(e: torch.Tensor) -> torch.Tensor:
        d = e.clamp(min=eps).pow(alpha)
        d = d / d.min()
        return d.clamp(max=max_s).float()

    attn = torch.ones(num_layers, max_group_k)
    mlp  = torch.ones(num_layers, max_group_k)
    down = torch.ones(num_layers, max_group_k)
    for l in range(num_layers):
        gk_l = min(int(layer_group_ks[l]), max_group_k)
        sal = residual_salient.get(seg_of(l))
        if sal is not None and gk_l > 0:
            idx = torch.as_tensor(list(sal)[:gk_l], dtype=torch.long)
            if (l, "attn") in second_moments:
                attn[l, :gk_l] = _scale_from(second_moments[(l, "attn")][idx])
            if (l, "mlp") in second_moments:
                mlp[l, :gk_l] = _scale_from(second_moments[(l, "mlp")][idx])
        dperm = internal_salient.get((l, "down_proj"))
        if dperm is not None and (l, "down_proj") in second_moments and gk_l > 0:
            didx = torch.as_tensor(list(dperm)[:gk_l], dtype=torch.long)
            down[l, :gk_l] = _scale_from(second_moments[(l, "down_proj")][didx])
    return {"attn": attn, "mlp": mlp, "down": down}


# ============================================================================
# Part 3 — Permutation construction
# ============================================================================

def _build_segment_perm(salient_channels: List[int], total_dim: int) -> List[int]:
    """
    Build permutation P: salient_channels first [0..group_k), then rest in original order.
    Works for both d_model (residual) and intermediate dims (down_proj).
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

    The residual entering the boundary is in P_k order: physical position i holds original
    channel P_k[i]. We want output position i to hold original channel P_kp1[i]. Under
    index_select that means out[i] = hs[ inv_Pk[P_kp1[i]] ], i.e.

        boundary_perm = inv_Pk[P_kp1]          (gather indices into the P_k stream)

    NOTE: inv_Pk INDEXED BY P_kp1, *not* P_kp1 indexed by inv_Pk. The latter is the inverse
    composition and silently corrupts the residual at every boundary (blows up logit error).
    """
    inv_Pk                    = torch.zeros(d_model, dtype=torch.long)
    inv_Pk[torch.tensor(P_k)] = torch.arange(d_model, dtype=torch.long)
    return inv_Pk[torch.tensor(P_kp1, dtype=torch.long)]


def _boundary_layer_indices(boundary_sizes: List[int]) -> List[int]:
    """Last decoder layer index of each non-final segment (for hook registration)."""
    cumsum = list(itertools.accumulate(boundary_sizes))
    return [c - 1 for c in cumsum[:-1]]


def _perm_tensor(t: torch.Tensor, perm: torch.Tensor, dim: int) -> torch.Tensor:
    """Permute tensor `t` along `dim` (0=rows, 1=cols) by index `perm`; returns contiguous."""
    idx = perm.to(t.device)
    return (t[idx] if dim == 0 else t[:, idx]).contiguous()


# ============================================================================
# Part 4 — Offline weight permutation (operates on plain .weight.data; dtype-preserving)
# ============================================================================

def _apply_residual_perm_fp32(
    model: nn.Module, block_indices: range, P_k: torch.Tensor,
) -> None:
    """
    Apply residual-stream permutation P_k to one segment.

    Input cols (dim 1) permuted by P_k:  q/k/v_proj, gate_proj, up_proj
    Output rows (dim 0) permuted by P_k: o_proj, down_proj
    1-D vector permuted:                 input_layernorm, post_attention_layernorm
    """
    for l in block_indices:
        attn = model.model.layers[l].self_attn
        mlp  = model.model.layers[l].mlp
        ln   = model.model.layers[l]

        ln.input_layernorm.weight.data          = ln.input_layernorm.weight.data[P_k]
        ln.post_attention_layernorm.weight.data = ln.post_attention_layernorm.weight.data[P_k]

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj):
            proj.weight.data = _perm_tensor(proj.weight.data, P_k, dim=1)   # input cols

        for proj in (attn.o_proj, mlp.down_proj):
            proj.weight.data = _perm_tensor(proj.weight.data, P_k, dim=0)   # output rows


def apply_segment_permutation_fp32(
    model: nn.Module,
    segment_perms: Dict[int, List[int]],
    boundary_sizes: List[int],
) -> List[torch.LongTensor]:
    """
    In-place residual-stream weight permutation (dense fp16/fp32 model; no BnB, no LoRA).
    Returns boundary_perms[k] = P_{k+1} ∘ P_k^{-1}, length = num_segments-1.
    """
    num_segments = len(boundary_sizes)
    d_model      = model.config.hidden_size
    b_offsets    = [0] + list(itertools.accumulate(boundary_sizes))

    if num_segments > 1 and hasattr(model, "lm_head"):
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError(
                "lm_head and embed_tokens are weight-tied but use different permutations "
                "(P_0 vs P_last) in multi-segment mode. "
                "Set tie_word_embeddings=False and clone lm_head.weight."
            )

    P      = {k: torch.tensor(v, dtype=torch.long) for k, v in segment_perms.items()}
    P_0    = P[0]
    P_last = P[num_segments - 1]

    # embed_tokens: d_model cols (dim 1) → P_0
    model.model.embed_tokens.weight.data = _perm_tensor(
        model.model.embed_tokens.weight.data, P_0, dim=1
    )

    for seg in range(num_segments):
        _apply_residual_perm_fp32(model, range(b_offsets[seg], b_offsets[seg + 1]), P[seg])

    model.model.norm.weight.data = model.model.norm.weight.data[P_last]
    model.lm_head.weight.data    = _perm_tensor(model.lm_head.weight.data, P_last, dim=1)

    boundary_perms = [
        _compute_boundary_perm(segment_perms[k], segment_perms[k + 1], d_model)
        for k in range(num_segments - 1)
    ]
    assert len(boundary_perms) == num_segments - 1
    print(f"[SegPerm] Permuted residual stream: {num_segments} segs, "
          f"num_runtime_permutes={len(boundary_perms)}")
    return boundary_perms


def apply_block_internal_permutations_fp32(
    model: nn.Module,
    internal_salient: Dict[Tuple[int, str], List[int]],
) -> Dict[Tuple[int, str], List[int]]:
    """
    Apply MLP block-internal P4_l permutation in-place (fully offline fold):
      gate_proj output rows (dim 0) ← P4_l
      up_proj   output rows (dim 0) ← P4_l
      down_proj input cols  (dim 1) ← P4_l
    o_proj is handled separately by apply_hadamard_rotation_fp32.
    Returns {(l, 'down_proj'): P4_l}.
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
# Part 5 — Attention Hadamard rotation (o_proj, per-layer per-head)
# ============================================================================

def _build_hadamard(n: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix n×n (n a power of 2); H @ H = I, entries ±1/sqrt(n)."""
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
    Apply per-head Hadamard rotation to v_proj/o_proj in-place (equivalence transform):
      v_proj.weight[h*hd:(h+1)*hd, :] ← H @ v_proj.weight[...]   (rotate v output rows per head)
      o_proj.weight[:, h*hd:(h+1)*hd] ← o_proj.weight[...] @ H   (H^{-1}=H cancels)

    Rotation runs in fp32 then casts back to the param dtype, so this is correct for both the
    fp32 Stage-1 model and the fp16 Stage-2 base. Skipped for GQA (v_out != o_in).
    Returns number of layers rotated.
    """
    H = _build_hadamard(head_dim).to(next(model.parameters()).device)
    rotated = 0
    for l in range(num_layers):
        attn  = model.model.layers[l].self_attn
        o_in  = attn.o_proj.weight.shape[1]
        v_out = attn.v_proj.weight.shape[0]
        if v_out != o_in:
            continue   # GQA: skip
        H_l = H.to(attn.v_proj.weight.device)
        v_dtype = attn.v_proj.weight.dtype
        o_dtype = attn.o_proj.weight.dtype
        v_w  = attn.v_proj.weight.data.float()
        o_w  = attn.o_proj.weight.data.float()
        for h in range(num_kv_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            v_w[s:e, :] = H_l @ v_w[s:e, :]
            o_w[:, s:e] = o_w[:, s:e] @ H_l
        attn.v_proj.weight.data = v_w.to(v_dtype)
        attn.o_proj.weight.data = o_w.to(o_dtype)
        rotated += 1
    print(f"[SegPerm] Hadamard rotation applied to {rotated}/{num_layers} layers "
          f"(head_dim={head_dim}, num_kv_heads={num_kv_heads})")
    return rotated


# ============================================================================
# Part 6 — Boundary gather hooks (residual stream; train + inference)
# ============================================================================

class BoundaryGatherHook:
    """
    Persistent forward hook: permutes the residual stream from P_k to P_{k+1} order after the
    last decoder layer of segment k.  boundary_perm = P_{k+1} ∘ P_k^{-1} (pre-computed offline).
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
    layers = _resolve_decoder_layers(model)
    hooks = []
    for k, (bp, li) in enumerate(zip(boundary_perms, boundary_layer_indices)):
        h = BoundaryGatherHook(bp, d_model)
        h.register(layers[li])
        hooks.append(h)
        print(f"[SegPerm] Boundary hook: seg {k}→{k+1} after layer {li}")
    print(f"[SegPerm] num_runtime_permutes={len(boundary_perms)}")
    return hooks


# ============================================================================
# Part 7 — Equivalence verification (Stage 1, plain fp32)
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
    Verify that permuted_model (all transforms applied + boundary gathers) is numerically
    equivalent to original_model on plain fp32.

    Checks:
      1. Final logits max-abs error < 0.1 (raises if violated).
      2. q_proj L0 output invariance: permutation must NOT change q output features.

    q_proj_tol=1e-2 separates fp32 reduction noise (~1e-3, from re-ordered RMSNorm/matmul
    reductions) from a real P leak into output ROWS (O(1), RoPE-breaking).
    Returns the max max-abs logit error across all inputs.
    """
    original_model.eval(); permuted_model.eval()
    device = next(original_model.parameters()).device

    q_orig_out: List[torch.Tensor] = []
    q_perm_out: List[torch.Tensor] = []

    def _qhook(store):
        def _h(m, i, o): store.append(o.detach().cpu())
        return _h

    h_o = original_model.model.layers[0].self_attn.q_proj.register_forward_hook(_qhook(q_orig_out))
    h_p = permuted_model.model.layers[0].self_attn.q_proj.register_forward_hook(_qhook(q_perm_out))

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


@torch.no_grad()
def build_and_verify_permutation_fp32(
    model: nn.Module,
    calibration_dataloader: DataLoader,
    boundary_sizes: List[int],
    test_inputs: List[Dict[str, torch.Tensor]],
    group_k: int = 128,
    group_size: int = 128,
    top_k_ratio: float = 0.01,
    outlier_log_sigma: float = 3.0,
    tol: float = 1e-3,
    return_artifacts: bool = False,
):
    """
    Stage-1 driver (used by scripts/verify_permute.py): on a plain fp32 `model`, calibrate →
    select salient → deep-copy the original → apply P_k + P4 + Hadamard → register boundary
    gathers → verify equivalence. Returns the max logit error. Mutates `model` in place.

    return_artifacts=True additionally returns the calibration/saliency intermediates
    (second_moments, residual_salient, internal_salient, ...) so callers can build and verify the
    AWQ-style salient scales S against the SAME permuted weights (Stage-1b fusion check).
    """
    import copy

    device         = next(model.parameters()).device
    d_model        = model.config.hidden_size
    num_layers     = model.config.num_hidden_layers
    num_kv_heads   = model.config.num_key_value_heads
    num_attn_heads = model.config.num_attention_heads
    head_dim       = d_model // num_attn_heads
    num_segments   = len(boundary_sizes)

    print("[SegPerm] Stage 1: collecting E[x²] (all sources) ...")
    second_moments = _collect_second_moments(
        model, calibration_dataloader, num_layers, device, collect_internal=True
    )

    print("[SegPerm] Stage 1: selecting salient channels ...")
    residual_salient = select_salient_channels(
        second_moments, d_model, boundary_sizes,
        top_k_ratio=top_k_ratio, group_k=group_k,
        group_size=group_size, outlier_log_sigma=outlier_log_sigma,
    )
    segment_perms = {
        k: _build_segment_perm(residual_salient[k], d_model) for k in range(num_segments)
    }
    internal_salient = select_internal_salient_channels(second_moments, num_layers, group_k=group_k)

    print("[SegPerm] Stage 1: deep-copying fp32 model for comparison ...")
    original = copy.deepcopy(model)

    print("[SegPerm] Stage 1: applying P_k + P4 + Hadamard ...")
    boundary_perms   = apply_segment_permutation_fp32(model, segment_perms, boundary_sizes)
    applied_internal = apply_block_internal_permutations_fp32(model, internal_salient)
    apply_hadamard_rotation_fp32(model, num_layers, num_kv_heads, head_dim)

    bli   = _boundary_layer_indices(boundary_sizes)
    hooks = register_boundary_gathers(model, boundary_perms, bli, d_model)
    try:
        print("[SegPerm] Stage 1: verifying equivalence (P_k + P4 + H all closed) ...")
        max_err = verify_permutation_equivalence(original, model, test_inputs, tol=tol)
    finally:
        for h in hooks:
            h.remove()

    print(
        f"\n[SegPerm] Stage 1 RESULT: num_runtime_permutes={len(boundary_perms)}, "
        f"num_P4_perms={len(applied_internal)}, num_H_layers={num_layers}, "
        f"max_abs_logit_err={max_err:.2e}"
    )
    if return_artifacts:
        artifacts = {
            "second_moments":   second_moments,
            "residual_salient": residual_salient,
            "internal_salient": internal_salient,
            "segment_perms":    segment_perms,
            "num_layers":       num_layers,
            "d_model":          d_model,
        }
        return max_err, artifacts
    return max_err


# ============================================================================
# Part 8 — Fresh STE group fakequant (input-column groups; per output row, per group)
# ============================================================================

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward = round, backward = identity."""
    return (torch.round(x) - x).detach() + x


def _asym_q_max(q_bits: int) -> int:
    """Affine (asymmetric) upper level: 2**bits - 1 (15 for INT4, 7 for INT3)."""
    return 2 ** q_bits - 1


def _sym_q_max(q_bits: int) -> int:
    """Symmetric clamp bound: 2**(bits-1) - 1 (7 for INT4, 3 for INT3)."""
    return 2 ** (q_bits - 1) - 1


# ----------------------------------------------------------------------------
# Canonical per-output-row, per-input-group quantization parameters.
#
# THIS IS THE SINGLE SOURCE OF TRUTH for the SQAT-permute grid. Training fakequant,
# the export PTQ, and the grid verifier ALL derive scale/zero_point from these two
# helpers, so the training-time grid and the deployment-time grid are identical by
# construction (the earlier regression was a min/max-vs-pos/neg formula mismatch
# between training and export — never reintroduce a second formula).
#
# Asymmetric (affine) convention:
#   scale = (wmax - wmin) / q_max ;  zp = round(-wmin/scale) clamped to [0, q_max]
#   quantize:   q  = round(clamp(w/scale + zp, 0, q_max))
#   dequantize: w' = (q - zp) * scale
# Symmetric convention:
#   scale = amax / q_max ;  quantize q = round(clamp(w/scale, -q_max, q_max)) ; w' = q*scale
# ----------------------------------------------------------------------------

def _asym_qparams(Wg: torch.Tensor, q_max: int, eps: float = 1e-8):
    """Wg: [..., group_size] (last dim = a quant group). Returns (scale, zp), each [..., 1]."""
    wmin = Wg.amin(dim=-1, keepdim=True)
    wmax = Wg.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / q_max).clamp(min=eps)
    zp    = torch.round(-wmin / scale).clamp(0, q_max)
    return scale, zp


def _sym_scale(Wg: torch.Tensor, q_max: int, eps: float = 1e-8) -> torch.Tensor:
    """Wg: [..., group_size]. Returns scale [..., 1]."""
    return (Wg.abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=eps)


def groupwise_symmetric_fakequant(
    W: torch.Tensor, group_size: int, q_max: int, eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric per-row, per-group fakequant with STE. W: [out, group_k] → same shape."""
    out_f, gk = W.shape
    assert gk % group_size == 0, f"group_k={gk} must be a multiple of group_size={group_size}"
    Wg    = W.reshape(out_f, gk // group_size, group_size)
    scale = _sym_scale(Wg, q_max, eps)
    q     = round_ste(torch.clamp(Wg / scale, -q_max, q_max))
    return (q * scale).reshape(out_f, gk)


def groupwise_asymmetric_fakequant(
    W: torch.Tensor, group_size: int, q_max: int, eps: float = 1e-8,
) -> torch.Tensor:
    """Affine asymmetric per-row, per-group fakequant with STE. W: [out, group_k] → same shape."""
    out_f, gk = W.shape
    assert gk % group_size == 0, f"group_k={gk} must be a multiple of group_size={group_size}"
    Wg        = W.reshape(out_f, gk // group_size, group_size)
    scale, zp = _asym_qparams(Wg, q_max, eps)
    q  = round_ste(torch.clamp(Wg / scale + zp, 0, q_max))
    return ((q - zp) * scale).reshape(out_f, gk)


def group_fakequant(
    W: torch.Tensor, group_size: int, q_bits: int, symmetric: bool,
    fixed_scale=None,
) -> torch.Tensor:
    """
    Training fakequant dispatch (STE). Returns dequantized W (same shape).

    fixed_scale (LSQ): when given, the per-step min-max scale is replaced by a learned scale[, zp]
    on the LSQ grid (the qat_base single-source-of-truth functions). sym → a scale tensor [out, ng];
    asym → a (scale[out, ng], zp[out, ng]) tuple. The LSQ grid differs from the min-max grid
    (sym Qn=-2^(b-1) vs the min-max -q_max), so train and export MUST both go through these LSQ
    functions — they do (group_quantize/group_dequantize below take the same fixed_scale).
    """
    if fixed_scale is not None:
        if symmetric:
            return _lsq_sym_fq(W, fixed_scale, group_size, q_bits)
        scale, zp = fixed_scale
        return _lsq_asym_fq(W, scale, zp, group_size, q_bits)
    if symmetric:
        return groupwise_symmetric_fakequant(W, group_size, _sym_q_max(q_bits))
    return groupwise_asymmetric_fakequant(W, group_size, _asym_q_max(q_bits))


# ----------------------------------------------------------------------------
# Export-side real quantize / dequantize (NO STE) — share the qparams above.
# group_dequantize(group_quantize(W)) == group_fakequant(W) exactly (verified).
# ----------------------------------------------------------------------------

@torch.no_grad()
def group_quantize(
    W: torch.Tensor, group_size: int, q_bits: int, symmetric: bool, eps: float = 1e-8,
    fixed_scale=None,
):
    """
    Real group quantization for export. Pads in_features to a group multiple internally.

    Returns:
        W_int:  [out, in_features] int levels (float tensor of integers, trimmed to in_features)
        scale:  [out, num_groups]
        zp:     [out, num_groups]   (all zeros for the symmetric branch)

    fixed_scale (LSQ): when given (sym: scale[out, ng]; asym: (scale, zp)), skip the min-max
    amax/_asym_qparams and quantize with the LEARNED scale[, zp] on the LSQ grid via the qat_base
    export quantizers. This is the export half of the train↔export single-source-of-truth: the
    returned (scale, zp) are exactly what group_dequantize/group_fakequant(fixed_scale=...) use.
    """
    out_f, in_f = W.shape
    if fixed_scale is not None:
        if symmetric:
            scale = fixed_scale.float()
            W_int = _lsq_export_sym(W, scale, group_size, q_bits)
            zp = torch.zeros_like(scale)
        else:
            scale, zp_in = fixed_scale
            scale = scale.float()
            W_int, z_int = _lsq_export_asym(W, scale, zp_in.float(), group_size, q_bits)
            zp = z_int
        return W_int.contiguous(), scale, zp
    ng  = math.ceil(in_f / group_size)
    pad = ng * group_size - in_f
    Wp  = F.pad(W, (0, pad)) if pad > 0 else W
    Wg  = Wp.reshape(out_f, ng, group_size)
    if symmetric:
        q_max = _sym_q_max(q_bits)
        scale = _sym_scale(Wg, q_max, eps)
        q     = torch.round(torch.clamp(Wg / scale, -q_max, q_max))
        zp    = torch.zeros_like(scale)
    else:
        q_max     = _asym_q_max(q_bits)
        scale, zp = _asym_qparams(Wg, q_max, eps)
        q = torch.round(torch.clamp(Wg / scale + zp, 0, q_max))
    W_int = q.reshape(out_f, -1)[:, :in_f].contiguous()
    return W_int, scale.squeeze(-1), zp.squeeze(-1)


@torch.no_grad()
def group_dequantize(
    W_int: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor,
    group_size: int, in_features: int, symmetric: bool,
) -> torch.Tensor:
    """Inverse of group_quantize. Returns dequantized W [out, in_features]."""
    out_f = W_int.shape[0]
    ng    = scale.shape[1]
    pad   = ng * group_size - in_features
    qf    = W_int.float()
    qp    = F.pad(qf, (0, pad)) if pad > 0 else qf
    Wg    = qp.reshape(out_f, ng, group_size)
    s     = scale.unsqueeze(-1).float()
    if symmetric:
        Wdq = Wg * s
    else:
        Wdq = (Wg - zp.unsqueeze(-1).float()) * s
    return Wdq.reshape(out_f, -1)[:, :in_features]


def _strip_peft_prefix(name: str) -> str:
    """Strip the PEFT wrapper prefix so training-time names match export dense-model names."""
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


@torch.no_grad()
def collect_lsq_scales_from_model(model: nn.Module) -> Dict[str, dict]:
    """
    Walk the model for proj modules carrying learned LSQ params (lsq_w_scale[, lsq_w_zp]) and
    return {stripped_proj_name: {"scale": [out, n_sal_g], "zp": [out, n_sal_g]}}.

    Keyed by the EXPORT (PEFT-prefix-stripped) name so export's per-module lookup is direct. The
    scales are in the AMPLIFIED space when AWQ is on (training learned them there); export quantizes
    W*S with these scales, then bakes /S into the dense weight — same as the training fakequant.
    """
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lsq_w_scale"):
            entry = {"scale": mod.lsq_w_scale.detach().float().cpu().clone()}
            if hasattr(mod, "lsq_w_zp"):
                entry["zp"] = mod.lsq_w_zp.detach().float().cpu().clone()
            out[_strip_peft_prefix(name)] = entry
    return out


@torch.no_grad()
def verify_permute_quant_consistency(
    W: torch.Tensor, group_k: int, group_size: int, q_bits: int, symmetric: bool,
    awq_s: Optional[torch.Tensor] = None,
    fixed_scale=None,
) -> float:
    """
    Assert the SALIENT slice's training grid == export grid. Returns max|Δ| over [out, group_k]
    between training fakequant and export quantize→dequant on the salient slice. With the shared
    qparams this must be ~0 (fp round-off only). If AWQ-style scaling is used, pass the per-channel
    `awq_s` the slice is quantized in (the amplify+de-amplify cancels, so it stays a self-check of
    the quantizer formulas in the amplified space).

    fixed_scale (LSQ): when given, both the training fakequant and the export quant→dequant use the
    learned scale[, zp] (LSQ grid) — the check then confirms the LSQ train↔export grids match.
    """
    W_s = W[:, :group_k].float()
    if awq_s is not None:
        s = awq_s.to(torch.float32).view(1, -1)
        fq = group_fakequant(W_s * s, group_size, q_bits, symmetric, fixed_scale=fixed_scale) / s
        wi, sc, zp = group_quantize(W_s * s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        dq = group_dequantize(wi, sc, zp, group_size, group_k, symmetric) / s
    else:
        fq = group_fakequant(W_s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        wi, sc, zp = group_quantize(W_s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        dq = group_dequantize(wi, sc, zp, group_size, group_k, symmetric)
    return (fq - dq).abs().max().item()


# ============================================================================
# Part 8b — GPTQ (Optimal Brain Quantization) for the NON-salient columns.
#
# Improvement over plain RTN export: the ~97% non-salient columns are quantized with OBS error
# compensation instead of round-to-nearest. The salient slice [0:layer_group_k] is the QAT-protected
# part and MUST keep the EXACT canonical group_quantize grid the LoRA was trained against (else
# the QAT benefit does not transfer — the earlier min/max-vs-pos/neg regression). So GPTQ here:
#   * fixes columns [0:layer_group_k] to the canonical RTN grid (training-consistent), and
#   * GPTQ-quantizes columns [layer_group_k:] with OBS compensation (same group_size and asym/sym as QAT).
# o_proj carries no salient slice (group_k=0) → it is fully GPTQ-quantized.
#
# The Hessian H = X^T X must be in the SAME (permuted) basis as the weight columns — collect it on
# the permuted base with the boundary gathers registered (gptq_quantize_model_sequential does this).
# ============================================================================

def _gptq_cholesky_inv_upper(H: torch.Tensor, percdamp: float, max_tries: int = 5) -> torch.Tensor:
    """
    Return the upper-triangular Cholesky factor U of H^{-1} (H^{-1} = U^T U), the form GPTQ's
    sequential update consumes. Dead (zero-activation) columns are made invertible; damping is
    escalated until H+damp is positive-definite.
    """
    cols = H.shape[0]
    H = H.clone()
    diagH = torch.diagonal(H)
    dead = diagH == 0.0
    if dead.any():
        H[dead, dead] = 1.0
        diagH = torch.diagonal(H)
    live = ~dead
    mean_diag = diagH[live].mean() if live.any() else H.new_tensor(1.0)
    idx = torch.arange(cols, device=H.device)
    base = percdamp * mean_diag
    for t in range(max_tries):
        Hd = H.clone()
        Hd[idx, idx] += base * (1.0 + t)            # escalate damping if not PD
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError:
            continue
    raise RuntimeError("GPTQ: Hessian Cholesky failed even after damping escalation.")


@torch.no_grad()
def gptq_quantize_layer(
    W: torch.Tensor,            # [out, in] dense weight (permuted basis)
    H: torch.Tensor,            # [in, in]  input Hessian (X^T X) in the SAME basis
    group_k: int,               # leading salient columns held at the canonical grid (0 = none)
    group_size: int,
    q_bits: int,
    symmetric: bool,
    percdamp: float = 0.01,
    blocksize: int = 128,
    eps: float = 1e-8,
    awq_s: Optional[torch.Tensor] = None,   # [group_k] AWQ scale for the salient slice, or None
    keep_salient_fp16: bool = False,        # ablation: leave the salient slice as fp16 (no quant)
    fixed_scale=None,                       # LSQ learned scale[, zp] for the salient slice, or None
):
    """
    Quantize the salient slice [0:group_k] to the canonical SQAT grid (training-consistent) and
    GPTQ-quantize ONLY the non-salient block [group_k:]. Returns (W_int [out,in], scale [out,ng],
    zp [out,ng]) in the EXACT layout of group_quantize, so the existing group_dequantize path
    reconstructs the deployed weight unchanged.

    IMPORTANT — the salient slice's quantization error is NOT propagated into the non-salient
    columns. The QAT/LoRA was trained to tolerate the salient slice's quant error (training saw
    the fakequant'd salient + the un-requantized fp16 non-salient), so the deployed non-salient
    must approximate that SAME fp16 weight W_n — not "absorb" the salient error (doing so shifts
    the deployed output away from the training-time output and degrades accuracy). GPTQ therefore
    runs as an INDEPENDENT OBS problem on the non-salient block with the non-salient sub-Hessian
    H[group_k:, group_k:], minimizing ||(W_q_n - W_n) X_n|| — strictly an improvement over RTN.

    AWQ-style scaling: if `awq_s` is given, the salient slice is quantized in the amplified space
    (W_S * S). The STORED ints/scale/zp stay in the amplified space — the caller bakes the `/S`
    back into the salient columns of the dequantized dense weight (see _unscale_salient_cols in
    export.py), matching the training fakequant W_fq/S exactly. AWQ only touches the salient slice.
    """
    dev = W.device
    out_f, in_f = W.shape
    assert in_f % group_size == 0, \
        f"GPTQ requires in_features ({in_f}) divisible by group_size ({group_size})"
    assert group_k % group_size == 0, \
        f"group_k ({group_k}) must be a multiple of group_size ({group_size})"
    ng = in_f // group_size
    q_max = _sym_q_max(q_bits) if symmetric else _asym_q_max(q_bits)

    # block size aligned to group_size so a quant group never straddles a block boundary
    if blocksize < group_size:
        blocksize = group_size
    blocksize = (blocksize // group_size) * group_size

    W = W.clone().float()
    H = H.float().to(dev)

    W_int = torch.zeros(out_f, in_f, device=dev)
    scale = torch.zeros(out_f, ng, device=dev)
    zp    = torch.zeros(out_f, ng, device=dev)

    # ---- 1) salient slice [0:group_k]: fixed canonical grid (amplified if AWQ). NO propagation. ----
    # ABLATION (keep_salient_fp16): the salient slice is NOT quantized at all — it is deployed at
    # fp16 (the methodology upper bound: "what if the QAT-protected slice were full precision?").
    # We leave W_int/scale/zp at zero for those leading columns; the caller restores the original
    # fp16 weight into the dequantized dense weight. The non-salient GPTQ block below is identical
    # either way (it already targets the fp16 W_n with the independent non-salient sub-Hessian).
    n_sal_g = group_k // group_size
    if group_k > 0 and not keep_salient_fp16:
        W_sal = W[:, :group_k]
        # LSQ: fixed_scale carries the learned scale[, zp]; group_quantize then uses the LSQ grid
        # for the salient slice (identical to the training fakequant). Move to W's device.
        fs = fixed_scale
        if fs is not None:
            fs = fs.to(dev) if torch.is_tensor(fs) else (fs[0].to(dev), fs[1].to(dev))
        if awq_s is not None:
            s = awq_s.to(torch.float32).view(1, -1).to(dev)
            wi_s, sc_s, zp_s = group_quantize(W_sal * s, group_size, q_bits, symmetric, eps,
                                              fixed_scale=fs)
        else:
            wi_s, sc_s, zp_s = group_quantize(W_sal, group_size, q_bits, symmetric, eps,
                                              fixed_scale=fs)
        W_int[:, :group_k] = wi_s.to(dev)
        scale[:, :n_sal_g]  = sc_s.to(dev)
        zp[:, :n_sal_g]     = zp_s.to(dev)

    if group_k >= in_f:                          # nothing non-salient to GPTQ (shouldn't happen)
        return W_int, scale, zp

    # ---- 2) GPTQ on the NON-salient block [group_k:] only (target the fp16 W_n). ----
    Wn = W[:, group_k:]                          # [out, in_n]  (never touched by the salient slice)
    Hn = H[group_k:, group_k:]
    in_n = in_f - group_k

    # STATIC groups: precompute every group's scale/zp from the ORIGINAL weights and keep them
    # FIXED during the sweep. GPTQ updates the not-yet-quantized columns (error compensation), so
    # a grid recomputed mid-sweep from the partially-updated weights is "stale" for the rest of the
    # group — that breaks OBS's fixed-grid assumption and makes the compensation INCREASE the output
    # error (GPTQ worse than RTN), worsening as the group grows / bits shrink. A fixed grid restores
    # OBS optimality (GPTQ ≤ RTN). (Standard GPTQ "static_groups".)
    ng_n = in_n // group_size
    for gi in range(ng_n):
        Wg = Wn[:, gi * group_size:(gi + 1) * group_size]
        if symmetric:
            s = _sym_scale(Wg, q_max, eps); z = torch.zeros_like(s)
        else:
            s, z = _asym_qparams(Wg, q_max, eps)
        scale[:, n_sal_g + gi] = s.squeeze(-1)
        zp[:, n_sal_g + gi]    = z.squeeze(-1)

    Hinv = _gptq_cholesky_inv_upper(Hn, percdamp)

    for i1 in range(0, in_n, blocksize):
        i2 = min(i1 + blocksize, in_n)
        W1    = Wn[:, i1:i2].clone()
        Err1  = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(i2 - i1):
            col = i1 + j                         # local column within the non-salient block
            g   = n_sal_g + col // group_size    # global group index
            w   = W1[:, j]
            d   = Hinv1[j, j]

            s = scale[:, g].unsqueeze(-1)        # FIXED grid (precomputed above)
            z = zp[:, g].unsqueeze(-1)
            if symmetric:
                qi = torch.round(torch.clamp(w.unsqueeze(-1) / s, -q_max, q_max))
                q  = (qi * s).squeeze(-1)
            else:
                qi = torch.round(torch.clamp(w.unsqueeze(-1) / s + z, 0, q_max))
                q  = ((qi - z) * s).squeeze(-1)
            W_int[:, group_k + col] = qi.squeeze(-1)

            err = (w - q) / d
            W1[:, j:] -= err.unsqueeze(-1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err

        Wn[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W_int, scale, zp


# ----------------------------------------------------------------------------
# AWQ-style per-input-channel scale S for the salient slice (Improvement 2).
# S is per (decoder layer, projection group): q/k/v share the "attn" S, gate/up share "mlp",
# down_proj has its own "down" S. Stored in perm_meta["awq_scales"] = {src: [num_layers, max_group_k]}.
# ----------------------------------------------------------------------------

_AWQ_SOURCE = {
    "q_proj": "attn", "k_proj": "attn", "v_proj": "attn",
    "gate_proj": "mlp", "up_proj": "mlp",
    "down_proj": "down",
}


def awq_s_for_module(
    awq_scales: Optional[dict],
    name: str,
    group_k: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return the [group_k] AWQ scale for a linear by full name, or None (o_proj / disabled)."""
    if not awq_scales:
        return None
    src = _AWQ_SOURCE.get(name.split(".")[-1])
    if src is None or src not in awq_scales:
        return None
    l = _layer_idx_from_module_name(name)
    if l is None:
        return None
    s = awq_scales[src]
    s = s[l] if not torch.is_tensor(s) or s.dim() == 2 else s
    s = torch.as_tensor(s, dtype=torch.float32)
    if group_k is not None:
        s = s[:int(group_k)]
    return s


def lsq_scale_for_module(lsq_scales: Optional[dict], name: str, symmetric: bool):
    """
    Return the learned LSQ fixed_scale for a proj by name, or None.

    sym  → scale tensor [out, n_sal_g]
    asym → (scale [out, n_sal_g], zp [out, n_sal_g])

    `lsq_scales` is keyed by the export (PEFT-prefix-stripped) name. Names are matched by exact
    key, falling back to a suffix match (handles any residual prefix differences).
    """
    if not lsq_scales:
        return None
    entry = lsq_scales.get(name)
    if entry is None:
        # suffix fallback (e.g. caller passes a name with an extra prefix)
        for k, v in lsq_scales.items():
            if name.endswith(k) or k.endswith(name):
                entry = v
                break
    if entry is None:
        return None
    scale = entry["scale"].float()
    if symmetric:
        return scale
    return scale, entry["zp"].float()


class _GPTQCatcherStop(Exception):
    """Raised by the layer-0 catcher to stop the forward after capturing the first input."""


@torch.no_grad()
def gptq_quantize_model_sequential(
    model: nn.Module,
    calibration_dataloader: DataLoader,
    target_terminals: Sequence[str],
    perm_group_k: int,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    device: torch.device,
    perm_meta=None,
    percdamp: float = 0.01,
    blocksize: int = 128,
    nsamples: int = 128,
    awq_scales: Optional[dict] = None,
    keep_salient_fp16: bool = False,
    lsq_scales: Optional[dict] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    In-place sequential GPTQ on a dense (permuted) fp16 model. For every nn.Linear whose terminal
    name is in `target_terminals`:
       * q/k/v/gate/up/down_proj → columns [0:layer_group_k] fixed to the canonical SQAT grid, the
         rest GPTQ-quantized. layer_group_k is read from perm_meta when present, otherwise
         perm_group_k is used for backward compatibility;
       * o_proj                  → no salient slice (group_k=0) → fully GPTQ.
    Each decoder layer is quantized using the ALREADY-quantized previous layers' outputs (true
    cross-layer sequential GPTQ): weights are replaced in place with their quantize→dequant values,
    and the per-layer (W_int, scale, zp) are returned (on CPU) in the group_quantize layout.

    The model MUST be the permuted base; boundary gathers from `perm_meta` are registered for the
    duration so the captured activations are in the deployment basis (and per-layer inputs are
    re-ordered at segment boundaries exactly as at inference).
    """
    target_terminals = set(target_terminals)
    name_of = {m: n for n, m in model.named_modules()}
    layers = _resolve_decoder_layers(model)
    num_layers = len(layers)

    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    model.eval()

    gather_hooks = register_boundary_gathers_from_meta(model, perm_meta) if perm_meta else []

    # ---- capture layer-0 input + per-batch kwargs (attention_mask / position_embeddings / ...) ----
    inps: List[torch.Tensor] = []
    kwargs_list: List[dict] = []
    orig_layer0 = layers[0]

    class _Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach().to("cpu"))
            kwargs_list.append(
                {k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in kw.items()}
            )
            raise _GPTQCatcherStop()

    layers[0] = _Catcher(orig_layer0)
    seen = 0
    for batch in calibration_dataloader:
        if seen >= nsamples:
            break
        ids = batch["input_ids"].to(device)
        am  = batch.get("attention_mask")
        am  = am.to(device) if am is not None else None
        try:
            model(input_ids=ids, attention_mask=am)
        except _GPTQCatcherStop:
            pass
        seen += ids.shape[0]
    layers[0] = orig_layer0
    print(f"[GPTQ] Captured {len(inps)} calibration batches ({seen} sequences).")

    def _kw_to_dev(kw):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}

    quantized_layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for L in tqdm(range(num_layers), desc="[GPTQ] Sequential quantize"):
        layer = layers[L]
        subs = {}                                   # full_name -> (module, group_k)
        for sub in layer.modules():
            if isinstance(sub, nn.Linear) and name_of[sub].split(".")[-1] in target_terminals:
                term = name_of[sub].split(".")[-1]
                nm = name_of[sub]
                gk = 0 if term == "o_proj" else group_k_for_module_name(
                    nm, perm_meta=perm_meta, default_group_k=perm_group_k,
                )
                subs[nm] = (sub, gk)
        if not subs:
            continue

        # 1) accumulate input Hessians for this layer's sublayers (fp16 weights)
        Hs: Dict[str, torch.Tensor] = {}
        handles = []

        def _mk(nm):
            def _h(mod, inp, out):
                x = inp[0].detach()
                x = x.reshape(-1, x.shape[-1]).float()
                xtx = x.t() @ x
                Hs[nm] = xtx if nm not in Hs else Hs[nm].add_(xtx)
            return _h

        for nm, (mod, _) in subs.items():
            handles.append(mod.register_forward_hook(_mk(nm)))
        for i in range(len(inps)):
            layer(inps[i].to(device), **_kw_to_dev(kwargs_list[i]))
        for h in handles:
            h.remove()

        # 2) GPTQ each sublayer; replace its weight with quantize->dequant
        for nm, (mod, gk) in subs.items():
            W = mod.weight.data.float()
            awq_s = awq_s_for_module(awq_scales, nm, gk) if gk > 0 else None
            # LSQ: salient slice uses the learned scale[, zp] (o_proj gk=0 → none).
            fixed_scale = (lsq_scale_for_module(lsq_scales, nm, symmetric)
                           if (lsq_scales and gk > 0) else None)
            W_int, sc, zp = gptq_quantize_layer(
                W, Hs[nm], gk, group_size, q_bits, symmetric,
                percdamp=percdamp, blocksize=blocksize, awq_s=awq_s,
                keep_salient_fp16=keep_salient_fp16,
                fixed_scale=fixed_scale,
            )
            W_deq = group_dequantize(W_int, sc, zp, group_size, W.shape[1], symmetric)
            if keep_salient_fp16 and gk > 0:
                # Ablation: restore the un-quantized fp16 salient slice (group_dequantize left it 0).
                # The cross-layer propagation below then sees the salient slice at full precision.
                W_deq[:, :gk] = W[:, :gk]
            if awq_s is not None:
                # bake 1/S back into the salient columns so the in-place (and exported) dense
                # weight is the deployed value W_fq/S — matching the training fakequant.
                W_deq[:, :gk] = W_deq[:, :gk] / awq_s.view(1, -1).to(W_deq.device)
            mod.weight.data.copy_(W_deq.to(mod.weight.dtype))
            quantized_layers[nm] = (W_int.cpu(), sc.cpu(), zp.cpu())
            Hs[nm] = None

        # 3) recompute inputs for the next layer using the QUANTIZED layer
        if L < num_layers - 1:
            for i in range(len(inps)):
                out = layer(inps[i].to(device), **_kw_to_dev(kwargs_list[i]))
                out = out[0] if isinstance(out, tuple) else out
                inps[i] = out.detach().to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in gather_hooks:
        h.remove()
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache

    return quantized_layers


# ============================================================================
# Part 9 — Fused QAT residual (pure math — unit-testable without a real model)
# ============================================================================

def _fused_BA(A_S_list: Sequence[torch.Tensor],
              B_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Concatenated B @ A_S over sibling projections, stacked along the output dim.

    When every (A_S, B) shares the same shape (MHA q/k/v, or gate/up), this is a single batched
    matmul (one kernel). Otherwise (e.g. GQA, where q has more output rows than k/v) it falls
    back to per-projection matmuls concatenated along dim 0 — the fused fakequant and the single
    injection GEMM downstream are unaffected either way.

    Returns: [sum_i out_i, group_k]
    """
    n = len(A_S_list)
    uniform = (
        n > 1
        and all(A.shape == A_S_list[0].shape for A in A_S_list)
        and all(B.shape == B_list[0].shape for B in B_list)
    )
    if uniform:
        Bs = torch.stack(list(B_list), dim=0)              # [n, out, rank]
        As = torch.stack(list(A_S_list), dim=0)            # [n, rank, group_k]
        BA = torch.bmm(Bs, As)                             # [n, out, group_k]
        return BA.reshape(n * BA.shape[1], BA.shape[2])    # rows ordered proj0,proj1,...
    return torch.cat([B @ A for B, A in zip(B_list, A_S_list)], dim=0)


def fused_qat_residual_outputs(
    W_base_salient: torch.Tensor,        # [sum_out, group_k]  (frozen NF4 base salient slice)
    A_S_list: Sequence[torch.Tensor],    # each [rank, group_k]
    B_list: Sequence[torch.Tensor],      # each [out_i, rank]
    out_splits: Sequence[int],           # [out_0, out_1, ...]
    X_S: torch.Tensor,                   # [..., group_k]  (= hidden_states[..., :group_k])
    group_size: int,
    q_bits: int,
    symmetric: bool,
    lora_scaling: float,
    awq_s: Optional[torch.Tensor] = None,   # [group_k] per-input-channel AWQ scale, or None
    lsq_scale: Optional[torch.Tensor] = None,   # [sum_out, n_sal_g] learned LSQ scale, or None
    lsq_zp: Optional[torch.Tensor] = None,      # [sum_out, n_sal_g] learned LSQ zp (asym), or None
) -> List[torch.Tensor]:
    """
    Compute the per-projection QAT residual outputs to be ADDED to each projection's output.

    W_curr = W_base_salient + (B @ A_S) * lora_scaling          [sum_out, group_k]
    delta  = fakequant(W_curr) - W_curr                          (STE residual, original space)
    Y      = F.linear(X_S, delta)                                [..., sum_out]   (ONE GEMM)
    return   Y.split(out_splits, dim=-1)

    AWQ-style per-channel scaling: if `awq_s` (a per-input-channel vector of length group_k, shared
    across the fused siblings) is given, the salient slice is quantized in the AMPLIFIED space —
    the salient input channels' weight columns are scaled up by S before quantization so the
    high-activation channels survive the shared group grid, then divided back out:

        W_fq = fakequant(W_curr * S) ;  delta = W_fq / S - W_curr ;  Y = F.linear(X_S, delta)

    The activation X_S is UNCHANGED — the 1/S is baked into the (dense) weight at export, so this is
    purely a better quantization grid (no runtime activation scaling, output bit-identical to a
    fold-1/S-into-the-preceding-LN deployment). The main NF4+LoRA path is untouched.

    LSQ: if `lsq_scale` (+ `lsq_zp` for asym) is given, the inner fakequant uses the LEARNED scale[,
    zp] on the LSQ grid instead of per-step min-max. It composes with AWQ — the scale/zp are learned
    in the SAME amplified space the fakequant runs in (W_curr*S), so the outer *S … /S is untouched.
    The scale rows are concatenated over the fused siblings in the SAME order as W_base_salient.

    The salient slice is the ONLY weight materialized. Quantization runs in fp32; the residual
    is cast back to X_S.dtype.
    """
    BA     = _fused_BA(A_S_list, B_list).to(torch.float32)         # [sum_out, group_k]
    W_curr = W_base_salient.to(torch.float32) + BA * lora_scaling  # [sum_out, group_k]
    if lsq_scale is not None:
        fixed = lsq_scale.float() if symmetric else (lsq_scale.float(), lsq_zp.float())
    else:
        fixed = None
    if awq_s is not None:
        s     = awq_s.to(torch.float32).view(1, -1)                # [1, group_k]
        W_fq  = group_fakequant(W_curr * s, group_size, q_bits, symmetric, fixed_scale=fixed)
        delta = (W_fq / s - W_curr).to(X_S.dtype)                  # STE residual (original space)
    else:
        W_fq  = group_fakequant(W_curr, group_size, q_bits, symmetric, fixed_scale=fixed)
        delta = (W_fq - W_curr).to(X_S.dtype)
    Y      = F.linear(X_S, delta)                                  # [..., sum_out], one GEMM
    return list(torch.split(Y, list(out_splits), dim=-1))


# ============================================================================
# Part 10 — PEFT projection helpers
# ============================================================================

def _has_lora(proj: nn.Module) -> bool:
    return hasattr(proj, "base_layer") and hasattr(proj, "lora_A") and len(proj.lora_A) > 0


def _adapter_name(proj: nn.Module) -> str:
    return list(proj.lora_A.keys())[0]


def _lora_A_S(proj: nn.Module, group_k: int) -> torch.Tensor:
    """A[:, :group_k]: [rank, group_k] — a view into the global LoRA A (no copy)."""
    return proj.lora_A[_adapter_name(proj)].weight[:, :group_k]


def _lora_B(proj: nn.Module) -> torch.Tensor:
    """B: [out_features, rank]."""
    return proj.lora_B[_adapter_name(proj)].weight


def _lora_dropout_p(proj: nn.Module) -> float:
    if not hasattr(proj, "lora_dropout") or len(proj.lora_dropout) == 0:
        return 0.0
    d = proj.lora_dropout[_adapter_name(proj)]
    return float(getattr(d, "p", 0.0) or 0.0)


def _warn_if_lora_dropout(projs: Sequence[nn.Module], where: str) -> None:
    ps = [_lora_dropout_p(p) for p in projs]
    if any(p > 0.0 for p in ps):
        warnings.warn(
            f"[qat_permute_sqat] {where}: lora_dropout={max(ps):.3g} > 0. Weight-level QAT "
            "injection (W_base + B@A_S) is NOT equivalent to the dropout-applied LoRA forward; "
            "results will be biased. Set lora_dropout=0 for selective-QAT training.",
            RuntimeWarning,
        )


def _dequant_base_salient(proj: nn.Module, group_k: int) -> torch.Tensor:
    """Frozen dequantized NF4 base, salient slice [out, group_k], fp32 on the proj's device."""
    W = dequantize_layer(proj)                              # [out, in]
    return W[:, :group_k].detach().to(torch.float32).contiguous()


# ============================================================================
# Part 11 — Block-level fused injectors (hook-based; original forwards untouched)
# ============================================================================

class _FusedSiblingQATInjector(nn.Module):
    """
    Shared logic for fusing sibling projections that read the SAME block input (q/k/v reading
    the input_layernorm output, or gate/up reading the post_attention_layernorm output).

    A forward_pre_hook on the parent block (`self_attn` / `mlp`) computes the fused residual
    ONCE from the block input and stashes per-projection slices; a forward_hook on each
    projection adds its slice to that projection's output. The parent block's forward and the
    projections' BnB/LoRA forwards are never replaced.
    """

    def __init__(
        self,
        block: nn.Module,
        projs: Sequence[nn.Module],
        group_k: int,
        group_size: int,
        q_bits: int,
        symmetric: bool,
        lora_scaling: float,
        where: str,
        awq_s: Optional[torch.Tensor] = None,
        enable_lsq: bool = False,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"
        self.projs        = list(projs)
        self.group_k      = group_k
        self.group_size   = group_size
        self.q_bits       = q_bits
        self.symmetric    = symmetric
        self.lora_scaling = lora_scaling
        self.where        = where
        self.enable_lsq   = bool(enable_lsq)

        _warn_if_lora_dropout(self.projs, where)

        bases = [_dequant_base_salient(p, group_k) for p in self.projs]
        self.out_splits = tuple(b.shape[0] for b in bases)
        # Frozen fused base salient slice [sum_out, group_k]; not saved to checkpoints.
        self.register_buffer("W_base_salient", torch.cat(bases, dim=0), persistent=False)

        # AWQ-style per-input-channel scale S for the salient slice (shared by the fused siblings,
        # since they read the same block input). Quantizing W_S*S protects the high-activation
        # channels; the /S is baked into the dense weight at export (no runtime activation scaling).
        # On the buffer's device — the injector is not a model submodule, so model.to() won't move it.
        if awq_s is not None:
            self.register_buffer(
                "awq_s",
                awq_s.to(torch.float32).view(-1).to(self.W_base_salient.device),
                persistent=False,
            )
        else:
            self.awq_s = None

        # LSQ: register a learned scale[, zp] PER PROJ (nn.Parameter on the proj module, so the HF
        # Trainer optimizer collects it and export can look it up by the proj's module name). Shape
        # [out_i, n_sal_g]. Init current_minmax from the base salient slice IN THE SAME (amplified)
        # space the fakequant runs in, so the initial grid matches the old min-max grid (no jump).
        if self.enable_lsq:
            n_sal_g = group_k // group_size
            for p, b in zip(self.projs, bases):
                W0 = b.to(torch.float32)                           # [out_i, group_k]
                if self.awq_s is not None:
                    W0 = W0 * self.awq_s.view(1, -1).to(W0.device)
                if symmetric:
                    s0 = _lsq_init_sym(W0, group_size, q_bits)     # [out_i, n_sal_g]
                    p.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                else:
                    s0, z0 = _lsq_init_asym(W0, group_size, q_bits)
                    p.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                    p.lsq_w_zp = nn.Parameter(z0.to(W0.device), requires_grad=True)
                assert p.lsq_w_scale.shape[1] == n_sal_g

        self._deltas: Optional[List[torch.Tensor]] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._handles.append(block.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for i, p in enumerate(self.projs):
            self._handles.append(p.register_forward_hook(self._make_add_hook(i)))

    @staticmethod
    def _extract_hidden_states(args, kwargs) -> torch.Tensor:
        if len(args) > 0 and torch.is_tensor(args[0]):
            return args[0]
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            return kwargs["hidden_states"]
        raise RuntimeError(
            "FusedSiblingQATInjector: could not locate hidden_states in the block forward args."
        )

    def _pre_hook(self, module, args, kwargs):
        hidden_states = self._extract_hidden_states(args, kwargs)
        X_S = hidden_states[..., :self.group_k]
        A_S_list = [_lora_A_S(p, self.group_k) for p in self.projs]
        B_list   = [_lora_B(p) for p in self.projs]
        lsq_scale = lsq_zp = None
        if self.enable_lsq:
            # Concatenate the per-proj learned scale[, zp] in the SAME order as W_base_salient.
            lsq_scale = torch.cat([p.lsq_w_scale for p in self.projs], dim=0)
            if not self.symmetric:
                lsq_zp = torch.cat([p.lsq_w_zp for p in self.projs], dim=0)
        self._deltas = fused_qat_residual_outputs(
            self.W_base_salient, A_S_list, B_list, self.out_splits, X_S,
            self.group_size, self.q_bits, self.symmetric, self.lora_scaling,
            awq_s=self.awq_s, lsq_scale=lsq_scale, lsq_zp=lsq_zp,
        )
        return None  # do not modify the block inputs

    def _make_add_hook(self, idx: int):
        def _add(module, inp, out):
            # _deltas is populated by the parent pre-hook earlier in the same forward.
            if self._deltas is None:
                return out
            result = out + self._deltas[idx]
            # Fix C: drop the Python reference to this (activation-sized) delta as
            # soon as it is consumed. The add result + autograd graph keep what
            # backward needs; without this the whole layer's deltas stay pinned on
            # self until the NEXT step's pre-hook overwrites them (~hundreds of MB
            # per layer x num_layers held idle between steps). Clearing per-index is
            # order-independent (does not assume sibling execution order) and the
            # pre-hook repopulates _deltas before these add-hooks run again
            # (including under gradient-checkpoint recompute), so it is safe.
            self._deltas[idx] = None
            return result
        return _add

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._deltas = None


class FusedAttnQATInjector(_FusedSiblingQATInjector):
    """Fuse q_proj/k_proj/v_proj. Handles GQA (q_out != kv_out) via the cat fallback."""

    def __init__(self, attn, q_proj, k_proj, v_proj, **kw):
        super().__init__(attn, [q_proj, k_proj, v_proj], where="self_attn(qkv)", **kw)


class FusedMLPQATInjector(_FusedSiblingQATInjector):
    """Fuse gate_proj/up_proj (always equal output dims = intermediate_size)."""

    def __init__(self, mlp, gate_proj, up_proj, **kw):
        super().__init__(mlp, [gate_proj, up_proj], where="mlp(gate,up)", **kw)


class DownProjQATInjector(nn.Module):
    """
    Single-projection QAT residual for down_proj (no sibling to fuse with). A self-contained
    forward_hook reads the projection input (the act(gate)*up activation, already permuted by P4
    so the salient channels are at [0:layer_group_k]) and adds one small delta GEMM to the output.
    """

    def __init__(
        self,
        down_proj: nn.Module,
        group_k: int,
        group_size: int,
        q_bits: int,
        symmetric: bool,
        lora_scaling: float,
        awq_s: Optional[torch.Tensor] = None,
        enable_lsq: bool = False,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"
        self.down_proj    = down_proj
        self.group_k      = group_k
        self.group_size   = group_size
        self.q_bits       = q_bits
        self.symmetric    = symmetric
        self.lora_scaling = lora_scaling
        self.enable_lsq   = bool(enable_lsq)

        _warn_if_lora_dropout([down_proj], "mlp(down)")
        self.register_buffer(
            "W_base_salient", _dequant_base_salient(down_proj, group_k), persistent=False
        )
        # AWQ scale for down_proj's own salient (P4-permuted) intermediate input slice.
        if awq_s is not None:
            self.register_buffer(
                "awq_s",
                awq_s.to(torch.float32).view(-1).to(self.W_base_salient.device),
                persistent=False,
            )
        else:
            self.awq_s = None
        self.out_splits = (self.W_base_salient.shape[0],)

        # LSQ: learned scale[, zp] for down_proj's salient slice (registered on the proj module).
        if self.enable_lsq:
            W0 = self.W_base_salient.to(torch.float32)
            if self.awq_s is not None:
                W0 = W0 * self.awq_s.view(1, -1).to(W0.device)
            if symmetric:
                s0 = _lsq_init_sym(W0, group_size, q_bits)
                down_proj.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
            else:
                s0, z0 = _lsq_init_asym(W0, group_size, q_bits)
                down_proj.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                down_proj.lsq_w_zp = nn.Parameter(z0.to(W0.device), requires_grad=True)

        self._handles = [down_proj.register_forward_hook(self._hook)]

    def _hook(self, module, inp, out):
        X_S = inp[0][..., :self.group_k]
        A_S = _lora_A_S(self.down_proj, self.group_k)
        B   = _lora_B(self.down_proj)
        lsq_scale = lsq_zp = None
        if self.enable_lsq:
            lsq_scale = self.down_proj.lsq_w_scale
            if not self.symmetric:
                lsq_zp = self.down_proj.lsq_w_zp
        (delta_out,) = fused_qat_residual_outputs(
            self.W_base_salient, [A_S], [B], self.out_splits, X_S,
            self.group_size, self.q_bits, self.symmetric, self.lora_scaling,
            awq_s=self.awq_s, lsq_scale=lsq_scale, lsq_zp=lsq_zp,
        )
        return out + delta_out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# ============================================================================
# Part 12 — Install / remove fused injectors
# ============================================================================

def install_fused_selective_qat(
    model: nn.Module,
    group_k: int,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    lora_scaling: float,
    target_modules: Sequence[str],
    include_down_proj: bool = True,
    awq_scales: Optional[dict] = None,
    layer_group_ks: Optional[Sequence[int]] = None,
    enable_lsq: bool = False,
) -> List[nn.Module]:
    """
    Install block-level fused Selective-QAT injectors on every decoder layer.

    - q/k/v_proj   → one FusedAttnQATInjector per layer (pre-hook on self_attn + add-hooks).
    - gate/up_proj → one FusedMLPQATInjector per layer.
    - down_proj    → one DownProjQATInjector per layer (single small GEMM, no fusion).

    awq_scales: if given ({"attn"/"mlp"/"down": [num_layers, max_group_k]}), the salient slice of each
    projection group is quantized in the AWQ-amplified space (q/k/v share the layer's "attn" S,
    gate/up share "mlp", down_proj uses "down"). o_proj is never injected (per-head Hadamard only).
    The /S is baked into the dense weight at export — a better quant grid, bit-identical output.

    Only projections that are LoRA-wrapped AND listed in `target_modules` are injected.
    Returns the list of injectors (keep a reference; call .remove() on each to uninstall).
    """
    layers = list(_resolve_decoder_layers(model))
    if layer_group_ks is None:
        layer_group_ks = [int(group_k)] * len(layers)
    else:
        layer_group_ks = [int(x) for x in layer_group_ks]
        assert len(layer_group_ks) == len(layers), (
            f"len(layer_group_ks)={len(layer_group_ks)} != num_layers={len(layers)}"
        )

    def _s(src: str, l: int, gk: int) -> Optional[torch.Tensor]:
        if not awq_scales or src not in awq_scales:
            return None
        return torch.as_tensor(awq_scales[src][l][:gk], dtype=torch.float32)

    tset = set(target_modules)
    injectors: List[nn.Module] = []

    for l, layer in enumerate(layers):
        attn = layer.self_attn
        mlp  = layer.mlp
        gk_l = int(layer_group_ks[l])
        common = dict(
            group_k=gk_l, group_size=group_size, q_bits=q_bits,
            symmetric=symmetric, lora_scaling=lora_scaling, enable_lsq=enable_lsq,
        )

        if {"q_proj", "k_proj", "v_proj"} <= tset and all(
            _has_lora(getattr(attn, n)) for n in ("q_proj", "k_proj", "v_proj")
        ):
            injectors.append(FusedAttnQATInjector(
                attn, attn.q_proj, attn.k_proj, attn.v_proj,
                awq_s=_s("attn", l, gk_l), **common
            ))

        if {"gate_proj", "up_proj"} <= tset and all(
            _has_lora(getattr(mlp, n)) for n in ("gate_proj", "up_proj")
        ):
            injectors.append(FusedMLPQATInjector(
                mlp, mlp.gate_proj, mlp.up_proj,
                awq_s=_s("mlp", l, gk_l), **common
            ))

        if include_down_proj and "down_proj" in tset and _has_lora(mlp.down_proj):
            injectors.append(DownProjQATInjector(mlp.down_proj, awq_s=_s("down", l, gk_l), **common))

    print(
        f"[qat_permute_sqat] Installed fused Selective-QAT injectors: "
        f"{sum(isinstance(i, FusedAttnQATInjector) for i in injectors)} attn, "
        f"{sum(isinstance(i, FusedMLPQATInjector) for i in injectors)} mlp, "
        f"{sum(isinstance(i, DownProjQATInjector) for i in injectors)} down  "
        f"(group_k_by_layer={min(layer_group_ks)}..{max(layer_group_ks)}, "
        f"group_size={group_size}, symmetric={symmetric}, "
        f"awq_scale={'on' if awq_scales else 'off'}, enable_lsq={enable_lsq})"
    )
    return injectors


def remove_fused_selective_qat(injectors: Sequence[nn.Module]) -> None:
    """Remove all hooks installed by install_fused_selective_qat."""
    for inj in injectors:
        inj.remove()


# ============================================================================
# Part 13 — Stage-2 orchestration: permute in fp16 → save → reload as NF4 ONCE
# ============================================================================

PERM_META_FILENAME = "sqat_permute_meta.pt"


@torch.no_grad()
def build_permuted_fp16_checkpoint(
    model_name: str,
    tokenizer,
    calibration_dataloader: DataLoader,
    boundary_sizes: Optional[List[int]],
    save_dir: str,
    group_k: Optional[int] = None,
    group_size: int = 128,
    top_k_ratio: float = 0.01,
    outlier_log_sigma: float = 3.0,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    awq_alpha: float = 0.5,
    awq_max: float = 2.0,
    group_k_candidates: Optional[Sequence[int]] = None,
    max_segments: int = 4,
) -> dict:
    """
    Stage-2 pre-quantization step — run on ONE process only (rank 0).

      1. Load `model_name` in `dtype` (fp16/bf16): no BNB, no LoRA.
      2. Calibrate E[x²] (residual + down_proj) on `calibration_dataloader`.
      3. Apply the three equivalence transforms IN fp16 (dtype-preserving):
           residual-stream P_k  → apply_segment_permutation_fp32
           MLP block P4_l        → apply_block_internal_permutations_fp32
           per-head Hadamard H   → apply_hadamard_rotation_fp32
      4. save_pretrained(save_dir) + tokenizer.save_pretrained(save_dir).

    Quantization is intentionally NOT done here. The caller sets cfg["model"]["name"]=save_dir
    and reloads through load_model_and_tokenizer, so NF4 quantizes the *permuted* weights
    exactly once (no dequant→permute→requant).

    Returns perm_meta (also written to save_dir/sqat_permute_meta.pt).
    """
    import os
    import gc
    from transformers import AutoModelForCausalLM

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SegPerm] Stage-2 pre-quant: loading {model_name} in {dtype} (no BNB) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    d_model        = model.config.hidden_size
    num_layers     = model.config.num_hidden_layers
    num_kv_heads   = model.config.num_key_value_heads
    num_attn_heads = model.config.num_attention_heads
    head_dim       = d_model // num_attn_heads

    # ---- 1) calibrate ----
    second_moments = _collect_second_moments(
        model, calibration_dataloader, num_layers, device, collect_internal=True,
    )

    # ---- 2) automatic/manual segment + group_k resolution ----
    candidates = _normalize_group_k_candidates(group_k_candidates, group_size)
    auto_summary = None
    if boundary_sizes is None:
        boundary_sizes, segment_group_ks, auto_summary = auto_segment_by_outliers(
            second_moments=second_moments,
            hidden_size=d_model,
            num_layers=num_layers,
            group_size=group_size,
            group_k_candidates=candidates,
            max_segments=max_segments,
            outlier_log_sigma=outlier_log_sigma,
        )
        print(
            f"[SegPerm] Auto segments: boundary_sizes={boundary_sizes}, "
            f"segment_group_ks={segment_group_ks}"
        )
    else:
        boundary_sizes = [int(x) for x in boundary_sizes]
        assert sum(boundary_sizes) == num_layers, (
            f"sum(boundary_sizes)={sum(boundary_sizes)} != num_hidden_layers={num_layers}"
        )
        if group_k is None:
            segment_group_ks = _segment_group_ks_for_boundaries(
                second_moments=second_moments,
                hidden_size=d_model,
                boundary_sizes=boundary_sizes,
                group_k_candidates=candidates,
                outlier_log_sigma=outlier_log_sigma,
            )
            print(
                f"[SegPerm] Manual segments + auto group_k: boundary_sizes={boundary_sizes}, "
                f"segment_group_ks={segment_group_ks}"
            )
        else:
            assert int(group_k) % group_size == 0, (
                f"group_k={group_k} must be a multiple of group_size={group_size}"
            )
            segment_group_ks = [int(group_k)] * len(boundary_sizes)
            print(
                f"[SegPerm] Manual segments + fixed group_k: boundary_sizes={boundary_sizes}, "
                f"group_k={int(group_k)}"
            )

    num_segments = len(boundary_sizes)
    layer_group_ks = _expand_segment_group_ks(boundary_sizes, segment_group_ks)
    max_group_k = int(max(segment_group_ks))

    # ---- 2b) salient selection ----
    residual_salient = select_salient_channels_variable(
        second_moments, d_model, boundary_sizes, segment_group_ks,
        group_size=group_size, outlier_log_sigma=outlier_log_sigma,
    )
    segment_perms = {
        k: _build_segment_perm(residual_salient[k], d_model) for k in range(num_segments)
    }
    internal_salient = select_internal_salient_channels(
        second_moments, num_layers, group_k=max_group_k, layer_group_ks=layer_group_ks,
    )

    # ---- 2b) AWQ-style per-channel salient scales S (always computed; usage gated by config) ----
    awq_scales = compute_awq_scales(
        second_moments, residual_salient, internal_salient,
        boundary_sizes, num_layers, max_group_k, layer_group_ks=layer_group_ks,
        alpha=awq_alpha, max_s=awq_max,
    )

    # ---- 3) apply the three transforms IN fp16 (dtype-preserving, equivalence-preserving) ----
    boundary_perms = apply_segment_permutation_fp32(model, segment_perms, boundary_sizes)
    block_internal = apply_block_internal_permutations_fp32(model, internal_salient)
    apply_hadamard_rotation_fp32(model, num_layers, num_kv_heads, head_dim)

    # ---- 4) save permuted fp16 base + tokenizer ----
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SegPerm] Saving permuted fp16 base → {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    bli = _boundary_layer_indices(boundary_sizes)
    perm_meta = {
        "boundary_perms":         [bp.cpu() for bp in boundary_perms],
        "boundary_layer_indices": bli,
        "segment_perms":          {k: list(v) for k, v in segment_perms.items()},
        "block_internal_perms":   {f"{k[0]}_{k[1]}": v for k, v in block_internal.items()},
        # Backward-compatible max group_k. New code reads layer_group_ks/segment_group_ks.
        "group_k":                max_group_k,
        "segment_group_ks":       list(segment_group_ks),
        "layer_group_ks":         list(layer_group_ks),
        "group_k_candidates":     list(candidates),
        "group_size":             group_size,
        "boundary_sizes":         list(boundary_sizes),
        "d_model":                d_model,
        "permuted_base_dir":      os.path.abspath(save_dir),
        "auto_segments":          auto_summary,
        # AWQ-style per-channel salient scales (used only when awq_scale is enabled in cfg).
        "awq_scales":             {k: v.cpu() for k, v in awq_scales.items()},
        "awq_alpha":              awq_alpha,
        "awq_max":                awq_max,
    }
    torch.save(perm_meta, os.path.join(save_dir, PERM_META_FILENAME))
    print(f"[SegPerm] perm_meta saved → {os.path.join(save_dir, PERM_META_FILENAME)} "
          f"(num_runtime_permutes={len(boundary_perms)}, num_P4={len(block_internal)})")

    # free the fp16 base so the caller's NF4 load has headroom
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return perm_meta


def load_perm_meta(meta_dir_or_path: str) -> dict:
    """Load perm_meta from a directory (expects sqat_permute_meta.pt) or a direct .pt path."""
    import os
    path = (meta_dir_or_path if meta_dir_or_path.endswith(".pt")
            else os.path.join(meta_dir_or_path, PERM_META_FILENAME))
    return torch.load(path, map_location="cpu")


def register_boundary_gathers_from_meta(
    model: nn.Module, meta,
) -> List[BoundaryGatherHook]:
    """
    Register the runtime boundary gathers on a (typically exported / reloaded) model so the
    residual stream is re-ordered P_k → P_{k+1} at each segment boundary. REQUIRED for any
    forward pass of a multi-segment permuted model: training re-registers these in
    prepare_model; inference must call this (see eval scripts).

    `meta` may be a perm_meta dict, the export wrapper {"layers":..., "model": perm_meta}, or a
    path/dir to sqat_permute_meta.pt.
    """
    if isinstance(meta, str):
        meta = load_perm_meta(meta)
    if isinstance(meta, dict) and "boundary_perms" not in meta and "model" in meta:
        meta = meta["model"]          # unwrap export {"layers", "model"} container
    if not meta or not meta.get("boundary_perms"):
        print("[SegPerm] register_boundary_gathers_from_meta: no boundaries "
              "(single segment) — nothing to register")
        return []
    d_model        = meta["d_model"]
    boundary_perms = [torch.as_tensor(bp, dtype=torch.long) for bp in meta["boundary_perms"]]
    bli            = meta["boundary_layer_indices"]
    return register_boundary_gathers(model, boundary_perms, bli, d_model)


def maybe_build_gather_aware_hflm(
    model_path: str,
    dtype: str = "float16",
    batch_size=8,
    peft: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Inference glue for lm-eval-harness. If `model_path` contains sqat_permute_meta.pt, build an
    HFLM explicitly and register the boundary gathers on its underlying HF model, then return
    the HFLM (pass it as `model=` to lm_eval.simple_evaluate). Returns None when there is no
    perm meta, so callers fall back to the plain `pretrained={path}` string path.
    """
    import os
    meta_path = os.path.join(model_path, PERM_META_FILENAME)
    if not os.path.exists(meta_path):
        return None

    from lm_eval.models.huggingface import HFLM
    hflm = HFLM(
        pretrained=model_path,
        dtype=dtype,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        peft=peft,
    )
    hf_model = getattr(hflm, "model", None) or getattr(hflm, "_model", None)
    if hf_model is None:
        raise RuntimeError("Could not access underlying HF model from HFLM to register gathers")
    hooks = register_boundary_gathers_from_meta(hf_model, meta_path)
    print(f"[SegPerm][eval] Registered {len(hooks)} boundary gather(s) for {model_path}")
    return hflm


def lm_eval_model_kwargs(
    model_path: str,
    dtype: str = "float16",
    batch_size=8,
    adapter_path: Optional[str] = None,
    trust_remote_code: bool = True,
) -> dict:
    """
    Return the kwargs to splat into ``lm_eval.simple_evaluate(...)`` for ``model_path``,
    transparently handling SQAT-Permute exports:

      - if ``model_path`` contains sqat_permute_meta.pt → build an HFLM with the boundary
        gathers registered and return {"model": hflm};
      - otherwise → return the standard {"model": "hf", "model_args": "pretrained=..."} form.

    A permute export is HARD-required to get its gathers: if the meta is present but the HFLM
    cannot be built, this raises rather than silently evaluating an incorrect model.
    """
    import os
    if os.path.exists(os.path.join(model_path, PERM_META_FILENAME)):
        hflm = maybe_build_gather_aware_hflm(
            model_path, dtype=dtype, batch_size=batch_size,
            peft=adapter_path, trust_remote_code=trust_remote_code,
        )
        if hflm is None:
            raise RuntimeError(
                f"{model_path} contains {PERM_META_FILENAME} but the gather-aware HFLM could "
                "not be built; refusing to evaluate without the boundary gather."
            )
        return {"model": hflm}

    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code={trust_remote_code}"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    return {"model": "hf", "model_args": model_args}


# ============================================================================
# Part 14 — QATHandler
# ============================================================================

class SegmentPermutedSelectiveQAT(QATHandler):
    """
    Segment-Shared Permutation QAT handler (Stage 2).

    The permute/fold happens BEFORE NF4 quantization, on a clean fp16 base, inside
    build_permuted_fp16_checkpoint() (called by scripts/train.py on rank 0). That base is saved
    and reloaded through the standard load_in_4bit path, so NF4 quantizes the permuted weights
    exactly once (no dequant→permute→requant round-trip).

    prepare_model() therefore receives the ALREADY-permuted, freshly-NF4-quantized PEFT model
    plus `perm_meta`, and only:
      1. Re-registers the runtime boundary gathers (residual reorder; cannot be folded).
      2. [stage>=2] Installs block-level fused Selective-QAT injectors (hooks) on
         q/k/v/gate/up/down_proj — QKV and Gate/Up each fuse into ONE big GEMM, down_proj is one
         small GEMM. o_proj is NOT injected (Hadamard-rotated only).
      3. Attaches _sqat_permute_meta for export.
    """

    def __init__(self):
        self.injectors:            List[nn.Module] = []
        self.boundary_hooks:       List[BoundaryGatherHook] = []
        self.boundary_perms:       List[torch.LongTensor] = []
        self.segment_perms:        Dict[int, List[int]] = {}
        self.block_internal_perms: Dict[str, List[int]] = {}
        self.enable_lsq:           bool = False
        self._lsq_proj_names:      Dict[int, Tuple[str, nn.Module]] = {}

    def prepare_model(
        self,
        model: nn.Module,
        cfg: dict,
        tokenizer=None,
        perm_meta: Optional[dict] = None,
        calibration_dataloader: Optional[DataLoader] = None,
        **kwargs,
    ) -> nn.Module:
        assert perm_meta is not None, (
            "sqat_permute prepare_model requires perm_meta from "
            "build_permuted_fp16_checkpoint(). The permute/fold happens in fp16 BEFORE NF4 "
            "(see scripts/train.py), not in this handler."
        )

        sp_cfg         = cfg["qat"]["sqat_permute"]
        stage          = sp_cfg.get("stage", 2)
        group_k        = perm_meta["group_k"]
        group_size     = perm_meta["group_size"]
        layer_group_ks = layer_group_ks_from_meta(perm_meta) or [int(group_k)] * int(sum(perm_meta["boundary_sizes"]))
        q_bits         = cfg["model"]["quant_bits"]
        symmetric      = cfg["qat"].get("symmetric", True)
        awq_enabled    = bool((sp_cfg.get("awq_scale", {}) or {}).get("enabled", False))
        enable_lsq     = bool(cfg["qat"].get("lsq", {}).get("enabled", False))
        lora_scaling   = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        target_modules = cfg["lora"]["target_modules"]
        d_model        = perm_meta["d_model"]
        awq_scales     = perm_meta.get("awq_scales") if awq_enabled else None

        # ---- 1) Runtime boundary gathers (residual reorder P_k → P_{k+1}). ----
        # MUST exist for every forward — at training (here) AND inference (eval scripts call
        # register_boundary_gathers_from_meta). Skip connections carry no weight, so this cannot
        # be folded offline; it is the one unavoidable runtime cost.
        self.boundary_perms = [
            torch.as_tensor(bp, dtype=torch.long) for bp in perm_meta["boundary_perms"]
        ]
        bli                       = perm_meta["boundary_layer_indices"]
        self.segment_perms        = perm_meta.get("segment_perms", {})
        self.block_internal_perms = perm_meta.get("block_internal_perms", {})
        self.boundary_hooks = register_boundary_gathers(
            model, self.boundary_perms, bli, d_model
        )

        # ---- 2) Stage 2 — install block-level fused Selective-QAT injectors. ----
        self.enable_lsq = enable_lsq
        if stage >= 2:
            self.injectors = install_fused_selective_qat(
                model,
                group_k=group_k,
                group_size=group_size,
                q_bits=q_bits,
                symmetric=symmetric,
                lora_scaling=lora_scaling,
                target_modules=target_modules,
                include_down_proj=True,
                awq_scales=awq_scales,
                layer_group_ks=layer_group_ks,
                enable_lsq=enable_lsq,
            )

        # ---- 3) Attach export metadata (perm_meta + q_bits + symmetric + awq flag). ----
        model._sqat_permute_meta = {
            **perm_meta, "q_bits": q_bits, "symmetric": symmetric,
            "awq_scale": awq_enabled, "lsq": enable_lsq,
        }
        # Keep a reference so collect (after training) can read the learned scales off the live
        # proj modules. The proj→name map lets us key lsq_scales by the export (stripped) name.
        self._lsq_proj_names = {}
        if enable_lsq:
            for name, mod in model.named_modules():
                if hasattr(mod, "lsq_w_scale"):
                    self._lsq_proj_names[id(mod)] = (name, mod)

        print(f"[SegPerm] prepare_model done: stage={stage}, "
              f"num_runtime_permutes={len(self.boundary_perms)}, "
              f"group_k_by_layer={min(layer_group_ks)}..{max(layer_group_ks)} "
              f"(max={group_k}), group_size={group_size}, symmetric={symmetric}, "
              f"awq_scale={awq_enabled}, enable_lsq={enable_lsq}")
        return model

    def on_train_begin(self, model: nn.Module): pass
    def on_step_end(self, model: nn.Module, step: int): pass
    def on_train_end(self, model: nn.Module, output_dir=None): pass
