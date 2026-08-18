"""
permute_common.py — the shared offline "permute the salient channels to the front" stack.

Used by BOTH quantization methods in this repo:
  * `qat_permute_sqat` (Permuted SQAT: NF4 base + LoRA, fakequant on the salient slice)
  * `qat_saltq`        (SALT-Q: trainable salient weights + frozen codes with trainable (s, z))

What lives here (everything that is method-agnostic):

  Calibration
    _collect_second_moments        per-channel E[x^2] for residual + block-internal sources

  Saliency & segmentation
    auto_segment_by_outliers / auto_segment_with_fixed_group_k     contiguous-segment DP
    select_salient_channels[_variable], select_internal_salient_channels
    compute_fakequant_param_stats  how many weights the salient slice actually covers

  Offline equivalence transforms (model output unchanged; applied on a clean fp16 base)
    (A) apply_segment_permutation_fp32        residual-stream P_k, per segment
    (B) apply_block_internal_permutations_fp32 MLP P4_l, per layer (gate/up rows + down cols)
    (C) apply_hadamard_rotation_fp32          per-head Hadamard on v_proj/o_proj (skipped for GQA)

  The one thing that CANNOT be folded offline
    BoundaryGatherHook / register_boundary_gathers[_from_meta]
    The skip connection carries no weight, so the residual must be re-ordered at runtime at each
    segment boundary (num_segments-1 index_selects). It MUST be registered in all three places:
    training (the QAT handler), the exported model, and lm-eval (lm_eval_model_kwargs).

  Orchestration & metadata
    build_permuted_fp16_checkpoint  permute a clean fp16 base ONCE and save it, so the downstream
                                    quantizer (NF4 for SQAT, GPTQ for SALT-Q) never sees a
                                    dequant->permute->requant round trip
    load_perm_meta / layer_group_ks_from_meta / group_k_for_module_name / AWQ-style scales

`o_proj` never gets a contiguous salient slice (per-head structure forbids cross-head permutation),
so `group_k_for_module_name` returns 0 for it and it is protected by the Hadamard rotation only.
"""

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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

TARGET_OUTLIER_CAPTURE = 1.0
QAT_FAKEQUANT_TERMINALS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
PERMUTE_TARGET_TERMINALS = QAT_FAKEQUANT_TERMINALS + ("o_proj",)


def _round_group_k_to_group_size(
    required_count: int,
    group_size: int,
    dim: int,
) -> int:
    """Smallest positive full quant group count that can hold required_count channels."""
    group_size = int(group_size)
    dim = int(dim)
    required_count = int(required_count)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    max_group_k = (dim // group_size) * group_size
    if max_group_k <= 0:
        raise ValueError(f"dim={dim} is smaller than group_size={group_size}")
    if required_count > max_group_k:
        raise RuntimeError(
            f"{required_count} outliers cannot fit in dim={dim} with group_size={group_size}; "
            f"largest valid group_k is {max_group_k}."
        )
    groups = max(1, math.ceil(required_count / group_size))
    return int(groups * group_size)


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


def down_layer_group_ks_from_meta(meta: Optional[dict]) -> Optional[List[int]]:
    meta = _unwrap_perm_model_meta(meta)
    if not meta:
        return None
    if "down_layer_group_ks" in meta:
        return [int(x) for x in meta["down_layer_group_ks"]]
    return layer_group_ks_from_meta(meta)


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
    layer_group_ks = (
        down_layer_group_ks_from_meta(perm_meta)
        if terminal == "down_proj" else layer_group_ks_from_meta(perm_meta)
    )
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


def _topk_ratio_mask(values: torch.Tensor, top_k_ratio: float) -> torch.Tensor:
    k = int(math.ceil(values.numel() * float(top_k_ratio)))
    k = min(values.numel(), max(1, k))
    idx = torch.topk(values.float(), k=k).indices
    mask = torch.zeros(values.numel(), dtype=torch.bool)
    mask[idx.cpu()] = True
    return mask


def _segment_group_ks_for_boundaries(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    boundary_sizes: Sequence[int],
    group_size: int,
    outlier_log_sigma: float,
) -> List[int]:
    outlier_sets = _residual_outlier_sets(second_moments, outlier_log_sigma)
    offsets = _boundary_offsets(boundary_sizes)
    out: List[int] = []
    for seg in range(len(boundary_sizes)):
        sources = _segment_sources(offsets[seg], offsets[seg + 1])
        mask = _segment_outlier_mask(sources, outlier_sets, hidden_size)
        union_count = int(mask.sum().item())
        gk = _round_group_k_to_group_size(union_count, group_size, hidden_size)
        if union_count > gk:
            raise RuntimeError(
                f"Manual segment {seg} L{offsets[seg]}-{offsets[seg + 1] - 1} has "
                f"{union_count} true outliers, larger than group_k={gk}."
            )
        out.append(gk)
    return out


def _down_layer_group_ks(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    num_layers: int,
    group_size: int,
    outlier_log_sigma: float,
) -> List[int]:
    out: List[int] = []
    for l in range(num_layers):
        sm = second_moments.get((l, "down_proj"))
        if sm is None:
            out.append(int(group_size))
            continue
        n_outliers = int(_source_outliers(sm, outlier_log_sigma).numel())
        gk = _round_group_k_to_group_size(n_outliers, group_size, sm.numel())
        if n_outliers > gk:
            raise RuntimeError(
                f"Layer {l} down_proj has {n_outliers} true outliers, larger than group_k={gk}."
            )
        out.append(gk)
    return out


def _module_weight_shape(module: nn.Module) -> Optional[Tuple[int, int]]:
    weight = getattr(module, "weight", None)
    if weight is None and hasattr(module, "base_layer"):
        weight = getattr(module.base_layer, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    return int(weight.shape[0]), int(weight.shape[1])


def _pct(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def compute_fakequant_param_stats(
    model: nn.Module,
    layer_group_ks: Sequence[int],
    down_layer_group_ks: Sequence[int],
    target_modules: Optional[Sequence[str]] = None,
    include_down_proj: bool = True,
) -> Dict:
    """
    Count weight parameters that will pass through Selective-QAT fakequant.

    fakequant_params counts only the salient input columns [0:group_k] for modules
    that actually use Selective-QAT fakequant. o_proj is kept as a denominator-only
    target when present in target_modules because it is Hadamard-rotated but has no
    contiguous salient slice.
    """
    layers = _resolve_decoder_layers(model)
    target_set = set(target_modules or PERMUTE_TARGET_TERMINALS)
    layer_group_ks = [int(x) for x in layer_group_ks]
    down_layer_group_ks = [int(x) for x in down_layer_group_ks]
    total_model_params = int(sum(p.numel() for p in model.parameters()))

    fakequant_params = 0
    qat_target_weight_params = 0
    lora_target_weight_params = 0
    by_projection: Dict[str, Dict[str, int]] = {}

    paths = (
        ("self_attn", "q_proj", "residual"),
        ("self_attn", "k_proj", "residual"),
        ("self_attn", "v_proj", "residual"),
        ("self_attn", "o_proj", "none"),
        ("mlp", "gate_proj", "residual"),
        ("mlp", "up_proj", "residual"),
        ("mlp", "down_proj", "down"),
    )
    for layer_idx, layer in enumerate(layers):
        for parent_name, terminal, group_source in paths:
            parent = getattr(layer, parent_name, None)
            module = getattr(parent, terminal, None) if parent is not None else None
            shape = _module_weight_shape(module) if module is not None else None
            if shape is None:
                continue

            out_features, in_features = shape
            weight_params = out_features * in_features
            if terminal in target_set:
                lora_target_weight_params += weight_params

            if terminal not in QAT_FAKEQUANT_TERMINALS or terminal not in target_set:
                continue
            if terminal == "down_proj" and not include_down_proj:
                continue

            group_k = (
                down_layer_group_ks[layer_idx]
                if group_source == "down" else layer_group_ks[layer_idx]
            )
            active_cols = min(int(group_k), in_features)
            active_params = out_features * active_cols
            fakequant_params += active_params
            qat_target_weight_params += weight_params

            proj = by_projection.setdefault(
                terminal,
                {"fakequant_params": 0, "weight_params": 0, "active_cols_sum": 0},
            )
            proj["fakequant_params"] += active_params
            proj["weight_params"] += weight_params
            proj["active_cols_sum"] += active_cols

    stats = {
        "fakequant_params": int(fakequant_params),
        "qat_target_weight_params": int(qat_target_weight_params),
        "lora_target_weight_params": int(lora_target_weight_params),
        "model_total_params": int(total_model_params),
        "ratio_of_qat_target_weights": _pct(fakequant_params, qat_target_weight_params),
        "ratio_of_lora_target_weights": _pct(fakequant_params, lora_target_weight_params),
        "ratio_of_model_params": _pct(fakequant_params, total_model_params),
        "by_projection": by_projection,
        "note": (
            "fakequant_params=sum(out_features*group_k) over q/k/v/gate/up/down; "
            "o_proj is denominator-only when it appears in target_modules."
        ),
    }
    return stats


def format_fakequant_param_stats(stats: Dict) -> str:
    return (
        f"fakequant_params={_format_count(stats['fakequant_params'])}; "
        f"of_qat_target_weights={stats['ratio_of_qat_target_weights'] * 100:.2f}% "
        f"({_format_count(stats['qat_target_weight_params'])} params); "
        f"of_lora_target_weights={stats['ratio_of_lora_target_weights'] * 100:.2f}% "
        f"({_format_count(stats['lora_target_weight_params'])} params); "
        f"of_model_params={stats['ratio_of_model_params'] * 100:.2f}% "
        f"({_format_count(stats['model_total_params'])} params)"
    )


def auto_segment_by_outliers(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    num_layers: int,
    group_size: int,
    max_segments: int = 4,
    outlier_log_sigma: float = 3.0,
) -> Tuple[List[int], List[int], Dict]:
    """
    Choose contiguous residual segments automatically.

    Hard constraint: every true per-source outlier, detected from that source's
    log(E[x^2]) distribution, must fit in its segment bucket. Objective under
    num_segments <= max_segments: minimize sum(segment_num_layers * group_k).
    """
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
            gk = _round_group_k_to_group_size(union_count, group_size, hidden_size)
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
            f"max_segments={max_segments}, group_size={group_size}. Increase max_segments "
            "or relax the outlier threshold."
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
        "group_size": int(group_size),
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


def auto_segment_with_fixed_group_k(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    num_layers: int,
    group_k: int,
    max_segments: int = 4,
    outlier_log_sigma: float = 3.0,
) -> Tuple[List[int], List[int], Dict]:
    """Choose contiguous residual segments while forcing every segment to use group_k."""
    group_k = int(group_k)
    max_segments = min(int(max_segments), int(num_layers))
    outlier_sets = _residual_outlier_sets(second_moments, outlier_log_sigma)

    seg_union_count: Dict[Tuple[int, int], int] = {}
    seg_overflow: Dict[Tuple[int, int], int] = {}
    for start in range(num_layers):
        for end in range(start + 1, num_layers + 1):
            sources = _segment_sources(start, end)
            mask = _segment_outlier_mask(sources, outlier_sets, hidden_size)
            union_count = int(mask.sum().item())
            key = (start, end)
            seg_union_count[key] = union_count
            seg_overflow[key] = max(0, union_count - group_k)

    inf = float("inf")
    dp = torch.full((max_segments + 1, num_layers + 1), inf, dtype=torch.float64)
    prev = torch.full((max_segments + 1, num_layers + 1), -1, dtype=torch.long)
    dp[0, 0] = 0.0
    for nseg in range(1, max_segments + 1):
        for end in range(1, num_layers + 1):
            for start in range(0, end):
                key = (start, end)
                if not torch.isfinite(dp[nseg - 1, start]):
                    continue
                cand = dp[nseg - 1, start] + seg_overflow[key]
                if cand < dp[nseg, end]:
                    dp[nseg, end] = cand
                    prev[nseg, end] = start

    curve = []
    best_nseg = -1
    best_overflow = inf
    for nseg in range(1, max_segments + 1):
        ok = bool(torch.isfinite(dp[nseg, num_layers]))
        overflow = int(dp[nseg, num_layers].item()) if ok else None
        curve.append({
            "num_segments": nseg,
            "feasible": ok,
            "overflow_outliers": overflow,
            "selection_cost": (float(num_layers * group_k + overflow) if ok else None),
        })
        if ok and (overflow < best_overflow or (overflow == best_overflow and best_nseg < 0)):
            best_overflow = overflow
            best_nseg = nseg

    if best_nseg < 0:
        raise RuntimeError("Fixed-group_k segment DP reconstruction failed before any candidate was found.")

    ranges: List[Tuple[int, int]] = []
    end = num_layers
    for nseg in range(best_nseg, 0, -1):
        start = int(prev[nseg, end].item())
        if start < 0:
            raise RuntimeError("Fixed-group_k segment DP reconstruction failed.")
        ranges.append((start, end))
        end = start
    ranges.reverse()

    if best_overflow > 0:
        print(
            f"[SegPerm] WARNING: fixed group_k={group_k} cannot cover all true residual "
            f"outliers with <= {max_segments} segments; selected minimum-overflow split "
            f"with {int(best_overflow)} outliers outside the protected buckets."
        )

    boundary_sizes = [end - start for start, end in ranges]
    segment_group_ks = [group_k] * len(ranges)
    summary = {
        "rule": "fixed group_k; choose <=max_segments split with minimum residual-outlier overflow",
        "target_capture": TARGET_OUTLIER_CAPTURE,
        "outlier_log_sigma": float(outlier_log_sigma),
        "fixed_group_k": group_k,
        "max_segments": int(max_segments),
        "overflow_outliers": int(best_overflow),
        "cost_curve": curve,
        "segments": [
            {
                "layers": [start, end - 1],
                "size": end - start,
                "group_k": group_k,
                "outlier_union_count": seg_union_count[(start, end)],
                "headroom": group_k - seg_union_count[(start, end)],
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
    Legacy residual-stream salient selection for fixed manual experiments.

    For each segment:
      1. Per-source normalize: sm / sm.max().
      2. Take top ceil(hidden_size * top_k_ratio) channels per source.
      3. Segment candidate set = union of those per-source top-ratio sets.
      4. Fill fixed group_k: union-first by aggregate score, then non-union by aggregate score.

    Returns {segment_idx: sorted List[int]} in d_model coordinate system.
    """
    assert group_k % group_size == 0, \
        f"group_k={group_k} must be a multiple of group_size={group_size}"

    b_offsets = _boundary_offsets(boundary_sizes)
    result: Dict[int, List[int]] = {}
    normalized_all = _normalized_residual_scores(second_moments)

    for seg in range(len(boundary_sizes)):
        b_start, b_end = b_offsets[seg], b_offsets[seg + 1]
        b_sources = _segment_sources(b_start, b_end)
        agg = _segment_aggregate_score(b_sources, normalized_all, hidden_size)
        union_mask = torch.zeros(hidden_size, dtype=torch.bool)
        for key in b_sources:
            union_mask |= _topk_ratio_mask(normalized_all[key], top_k_ratio)

        salient = _select_bucket_from_mask(agg, union_mask, int(group_k))
        result[seg] = salient

        sel_t = torch.tensor(salient, dtype=torch.long)
        e_total = sum(second_moments[k].sum().item() for k in b_sources)
        e_sel = sum(second_moments[k][sel_t].sum().item() for k in b_sources)
        n_union = int(union_mask.sum().item())
        print(
            f"[SegPerm] Legacy seg {seg} (L{b_start}-L{b_end-1}): "
            f"group_k={group_k}, top_k_ratio={top_k_ratio:g}, "
            f"union_topk={n_union}, energy_cov={e_sel/(e_total+1e-30)*100:.1f}%, "
            f"first10={salient[:10]}"
        )
        if n_union > group_k:
            print(
                f"[SegPerm] WARNING: legacy seg {seg} has top-ratio union size {n_union} "
                f"> group_k={group_k}; only the highest aggregate-score union channels fit."
            )

    return result


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
    down_layer_group_ks: Optional[Sequence[int]] = None,
) -> Dict[Tuple[int, str], List[int]]:
    """
    Per-block salient channel selection for down_proj only → full permutations.

    down_proj (P4_l): per-layer top-down_layer_group_k by E[x^2] across all intermediate_dim channels.
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
            layer_k = int(down_layer_group_ks[l]) if down_layer_group_ks is not None else int(group_k)
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
    down_layer_group_ks: Optional[Sequence[int]] = None,
    alpha: float = 0.5,
    max_s: float = 2.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    AWQ-style per-input-channel scale S on the salient slice [0:group_k], per (layer, source):
      attn (q/k/v share)   ← (l,'attn')      E[x²] at the segment's salient channels
      mlp  (gate/up share) ← (l,'mlp')       E[x²] at the same salient channels
      down (down_proj)     ← (l,'down_proj') E[x²] at the layer-local P4 salient channels
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
    if down_layer_group_ks is None:
        down_layer_group_ks = list(layer_group_ks)
    else:
        down_layer_group_ks = [int(x) for x in down_layer_group_ks]
        assert len(down_layer_group_ks) == num_layers, (
            f"len(down_layer_group_ks)={len(down_layer_group_ks)} != num_layers={num_layers}"
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
        residual_gk_l = min(int(layer_group_ks[l]), max_group_k)
        down_gk_l = min(int(down_layer_group_ks[l]), max_group_k)
        sal = residual_salient.get(seg_of(l))
        if sal is not None and residual_gk_l > 0:
            idx = torch.as_tensor(list(sal)[:residual_gk_l], dtype=torch.long)
            if (l, "attn") in second_moments:
                attn[l, :residual_gk_l] = _scale_from(second_moments[(l, "attn")][idx])
            if (l, "mlp") in second_moments:
                mlp[l, :residual_gk_l] = _scale_from(second_moments[(l, "mlp")][idx])
        dperm = internal_salient.get((l, "down_proj"))
        if dperm is not None and (l, "down_proj") in second_moments and down_gk_l > 0:
            didx = torch.as_tensor(list(dperm)[:down_gk_l], dtype=torch.long)
            down[l, :down_gk_l] = _scale_from(second_moments[(l, "down_proj")][didx])
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


def reorder_salient_by_post_awq_magnitude(
    model: nn.Module,
    residual_salient: Dict[int, List[int]],
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    boundary_sizes: Sequence[int],
    *,
    alpha: float = 0.5,
    max_s: float = 2.0,
    eps: float = 1e-12,
) -> Dict[int, List[int]]:
    """
    Reorder each segment's salient channel LIST (not its membership) so that channels which will
    end up with similar magnitude after the AWQ fold land in the same quantization group.

    WHY THIS IS FREE. `_build_segment_perm` lays the salient channels out at [0, group_k) in
    whatever order the list happens to be in, and that order is otherwise arbitrary — it is folded
    offline into the weights like the rest of P_k and costs nothing at runtime. Which channels are
    salient is unchanged, so group_k, the outlier capture and the boundary gathers are all
    unaffected; only WHICH 32 channels share a scale changes.

    WHY POST-AWQ MAGNITUDE IS THE RIGHT KEY. A group's scale is s = (max - min)/(2^b - 1) over its
    32 columns, so what a shared scale costs is the SPREAD inside the group. Before the AWQ fold
    the salient columns' weights are fairly homogeneous — saliency is a property of the
    activations, not of the weights — so there is little spread to exploit and regrouping buys
    almost nothing. The fold is what creates the spread: it multiplies column j by
    S_j = E[x_j^2]^alpha (normalized to min 1, clamped to max_s), deliberately paying a larger
    shared scale to buy smaller relative error on the highest-E[x^2] channels. Sorting by the
    POST-fold magnitude |W_j| * S_j therefore isolates the amplified channels together, so the
    scale they inflate is one they share with each other rather than with quiet columns. Sorting
    on E[x^2] alone, or on |W| alone, each captures only one of the two factors.

    THE COMPROMISE, STATED. P_k is shared by every layer in the segment AND by both the attn
    (q/k/v) and mlp (gate/up) consumers, while |W_j| is per-(layer, projection) and E[x_j^2]
    differs between the attn and mlp sources. One ordering therefore cannot be optimal for all of
    them; the key below is the mean of |W_j| * S_j over the segment's layers and over both
    sources, i.e. the best single compromise rather than a per-layer optimum. down_proj is NOT
    handled here — it permutes per layer via `internal_salient`, so it can be ordered exactly.

    Returns a new {segment: [channels]} dict; the input is not modified.
    """
    b_off = [0] + list(itertools.accumulate(boundary_sizes))
    num_segments = len(boundary_sizes)
    out: Dict[int, List[int]] = {}

    for k in range(num_segments):
        chans = list(residual_salient.get(k, []))
        if len(chans) <= 1:
            out[k] = chans
            continue
        idx = torch.tensor(chans, dtype=torch.long)
        acc = torch.zeros(len(chans), dtype=torch.float64)
        n_acc = 0

        for l in range(b_off[k], min(b_off[k + 1], model.config.num_hidden_layers)):
            layer = model.model.layers[l]
            for src, projs in (
                ("attn", (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)),
                ("mlp", (layer.mlp.gate_proj, layer.mlp.up_proj)),
            ):
                sm = second_moments.get((l, src))
                if sm is None:
                    continue
                h = torch.as_tensor(sm, dtype=torch.float64).flatten()[idx].clamp(min=eps)
                S = h.pow(alpha)
                S = (S / S.min().clamp(min=eps)).clamp(max=max_s)   # same normalization as
                                                                    # compute_awq_scales
                for p in projs:
                    w = p.weight.data
                    if w.shape[1] <= int(idx.max()):
                        continue
                    # RMS over output rows: a per-column magnitude comparable across columns.
                    m = w[:, idx.to(w.device)].to(torch.float64).pow(2).mean(0).sqrt().cpu()
                    acc += m * S
                    n_acc += 1

        if n_acc == 0:
            out[k] = chans
            continue
        key = acc / n_acc
        order = torch.argsort(key, descending=True)
        out[k] = [chans[i] for i in order.tolist()]

    print(f"[SegPerm] Reordered salient channels within {len(out)} segments by post-AWQ "
          f"magnitude (membership and group_k unchanged)")
    return out


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


def apply_awq_folding_fp32(
    model: nn.Module,
    awq_scales: Optional[dict],
    layer_group_ks: Sequence[int],
    down_layer_group_ks: Sequence[int],
) -> int:
    """
    Fold the AWQ per-input-channel scale S into the weights, offline and EXACTLY.

    This is the fourth equivalence-preserving transform, and it runs AFTER the permutations so
    that the salient channels already occupy [0, group_k). For every linear whose input carries a
    salient slice we scale that slice UP by S and divide the PRODUCER of those activations by S:

        q/k/v  cols[:gk] *= S_attn   <->  input_layernorm.weight[:gk]          /= S_attn
        gate/up cols[:gk] *= S_mlp   <->  post_attention_layernorm.weight[:gk] /= S_mlp
        down    cols[:gk] *= S_down  <->  up_proj.weight[rows :gk, :]          /= S_down

    Each pairing is exact and has exactly one consumer, which is what makes the fold safe:
      * Llama's input_layernorm feeds ONLY q/k/v, and post_attention_layernorm ONLY gate/up, so
        rescaling an RMSNorm weight channel rescales precisely the activations we scaled up. The
        residual stream bypasses both norms and is untouched.
      * down_proj's input is silu(gate(x)) * up(x), elementwise, so dividing up_proj's OUTPUT row j
        divides input channel j of down_proj. up_proj's rows are consumed by nothing else.
      * q/k/v share S_attn and gate/up share S_mlp (compute_awq_scales emits one vector per
        (layer, source)), so a single norm rescale serves both consumers.
      * o_proj has no salient slice (group_k = 0) and is never touched.

    WHY FOLD HERE rather than at export. Once S lives in the weights, the quantizer simply sees a
    better-conditioned matrix and NOTHING downstream needs to know about AWQ: GPTQ, the SALT-Q
    training forward and the merge-free export are all unchanged, and `deployed == trained` still
    holds by construction. The alternative — carrying S through training and dividing it out at
    export — cannot work for SALT-Q, because 1/S is per-COLUMN while a deployed INT-b checkpoint
    only has a per-(row, group) scale; that path forces a dense export and destroys the property
    that the codes ARE the deployment.

    NOTE the direction: compute_awq_scales normalizes min(S) = 1 and clamps to [1, max_s], so S
    only ever AMPLIFIES. It does not compress the within-group range — it deliberately spends
    range (a larger shared s for the whole group) to buy smaller relative error on the channels
    with the largest E[x^2]. Grouping the amplified channels together afterwards is what contains
    that cost, which is why any salient reordering must be keyed on the POST-folding magnitudes.

    Returns the number of layers folded. No-op (returns 0) when awq_scales is falsy.
    """
    if not awq_scales:
        return 0

    num_layers = model.config.num_hidden_layers
    n_folded = 0

    for l in range(num_layers):
        layer = model.model.layers[l]
        gk = int(layer_group_ks[l]) if l < len(layer_group_ks) else 0
        gk_d = int(down_layer_group_ks[l]) if l < len(down_layer_group_ks) else 0

        # ---- attention: q/k/v input cols <-> input_layernorm ----
        if gk > 0 and "attn" in awq_scales:
            S = torch.as_tensor(awq_scales["attn"][l][:gk], dtype=torch.float32)
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                w = proj.weight.data
                w32 = w[:, :gk].to(torch.float32) * S.view(1, -1).to(w.device)
                w[:, :gk] = w32.to(w.dtype)
            ln = layer.input_layernorm.weight.data
            ln[:gk] = (ln[:gk].to(torch.float32) / S.to(ln.device)).to(ln.dtype)

        # ---- MLP: gate/up input cols <-> post_attention_layernorm ----
        if gk > 0 and "mlp" in awq_scales:
            S = torch.as_tensor(awq_scales["mlp"][l][:gk], dtype=torch.float32)
            for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
                w = proj.weight.data
                w32 = w[:, :gk].to(torch.float32) * S.view(1, -1).to(w.device)
                w[:, :gk] = w32.to(w.dtype)
            ln = layer.post_attention_layernorm.weight.data
            ln[:gk] = (ln[:gk].to(torch.float32) / S.to(ln.device)).to(ln.dtype)

        # ---- down_proj input cols <-> up_proj output rows ----
        if gk_d > 0 and "down" in awq_scales:
            S = torch.as_tensor(awq_scales["down"][l][:gk_d], dtype=torch.float32)
            w = layer.mlp.down_proj.weight.data
            w32 = w[:, :gk_d].to(torch.float32) * S.view(1, -1).to(w.device)
            w[:, :gk_d] = w32.to(w.dtype)
            wu = layer.mlp.up_proj.weight.data
            wu[:gk_d, :] = (wu[:gk_d, :].to(torch.float32)
                            / S.view(-1, 1).to(wu.device)).to(wu.dtype)

        n_folded += 1

    print(f"[SegPerm] Folded AWQ per-channel scales into {n_folded} layers "
          f"(salient cols amplified, producers divided — model output unchanged)")
    return n_folded


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


@torch.no_grad()
def fold_boundary_gathers_into_weights(model: nn.Module, meta) -> int:
    """Undo the residual-stream permutation in the weights so the model needs NO runtime hook.

    A multi-segment permuted model is only correct with BoundaryGatherHooks registered: the
    residual stream is in P_k order inside segment k and has to be re-gathered to P_{k+1} at
    each boundary. That is a Python forward hook on the HF module tree, which exists only in
    this repo's own eval path — an external runtime (vLLM) loads the checkpoint, sees a plain
    Llama, and silently produces garbage.

    The permutation is pure relabelling of the residual channels, so it can simply be undone.
    This applies P_k^{-1} to every tensor apply_segment_permutation_fp32 touched, which puts
    the whole model back in the ORIGINAL channel order; the boundary gathers then compose to
    the identity and are no longer needed. It is exact — a reindex, not an approximation —
    and it does not undo the quantization: the VALUES are the trained/deployed ones, only
    their position in the channel axis moves back.

    The block-internal (P4) permutation and the Hadamard rotation are deliberately left alone.
    Both are confined inside a block (gate/up rows <-> down cols; v rows <-> o cols) and never
    reach the residual stream, so neither needs a runtime op in the first place.

    `meta` may be a perm_meta dict, the export wrapper {"layers":..., "model": perm_meta}, or
    a path/dir to sqat_permute_meta.pt. Returns the number of boundaries eliminated.
    """
    if isinstance(meta, str):
        meta = load_perm_meta(meta)
    meta = _unwrap_perm_model_meta(meta) or meta
    segment_perms = meta.get("segment_perms") or {}
    boundary_sizes = list(meta.get("boundary_sizes") or [])
    if not segment_perms or not boundary_sizes:
        raise ValueError(
            "fold_boundary_gathers_into_weights needs segment_perms + boundary_sizes in the "
            f"perm meta; got keys {sorted(meta.keys())}."
        )

    num_segments = len(boundary_sizes)
    b_offsets = _boundary_offsets(boundary_sizes)
    # inv[P[i]] = i, so t_permuted[inv] == t_original (t_permuted[i] == t[P[i]]).
    inv = {
        int(k): torch.argsort(torch.as_tensor(v, dtype=torch.long))
        for k, v in segment_perms.items()
    }
    missing = [k for k in range(num_segments) if k not in inv]
    if missing:
        raise ValueError(f"perm meta has no segment_perms entry for segment(s) {missing}.")

    inner = _resolve_llama_model(model)
    inner.embed_tokens.weight.data = _perm_tensor(
        inner.embed_tokens.weight.data, inv[0], dim=1
    )
    for seg in range(num_segments):
        _apply_residual_perm_fp32(
            model, range(b_offsets[seg], b_offsets[seg + 1]), inv[seg]
        )
    inner.norm.weight.data = inner.norm.weight.data[inv[num_segments - 1]]
    model.lm_head.weight.data = _perm_tensor(
        model.lm_head.weight.data, inv[num_segments - 1], dim=1
    )

    n_boundaries = num_segments - 1
    print(f"[SegPerm] Folded the residual permutation back into the weights: "
          f"{num_segments} segments, {n_boundaries} runtime gather(s) eliminated")
    return n_boundaries


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
    target_modules: Optional[Sequence[str]] = None,
    group_k: Optional[int] = None,
    group_size: int = 128,
    top_k_ratio: float = 0.01,
    outlier_log_sigma: float = 3.0,
    down_outlier_log_sigma: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    awq_alpha: float = 0.5,
    awq_max: float = 2.0,
    max_segments: int = 4,
    fold_awq: bool = False,
    reorder_salient: bool = False,
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
    if down_outlier_log_sigma is None:
        down_outlier_log_sigma = outlier_log_sigma
    down_outlier_log_sigma = float(down_outlier_log_sigma)

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
    fixed_group_k: Optional[int] = None
    if group_k is not None:
        fixed_group_k = int(group_k)
        if fixed_group_k <= 0:
            raise ValueError(f"group_k must be positive, got {fixed_group_k}")
        if fixed_group_k % group_size != 0:
            raise ValueError(
                f"group_k={fixed_group_k} must be a multiple of group_size={group_size}"
            )
        max_residual_group_k = (d_model // group_size) * group_size
        if fixed_group_k > max_residual_group_k:
            raise ValueError(
                f"group_k={fixed_group_k} does not fit hidden_size={d_model} "
                f"with group_size={group_size}; max valid group_k is {max_residual_group_k}."
            )

    # ---- 1) calibrate ----
    second_moments = _collect_second_moments(
        model, calibration_dataloader, num_layers, device, collect_internal=True,
    )

    # ---- 2) automatic/manual segment + group_k resolution ----
    auto_summary = None
    legacy_topk_ratio_mode = boundary_sizes is not None and fixed_group_k is not None
    if boundary_sizes is None:
        if fixed_group_k is None:
            boundary_sizes, segment_group_ks, auto_summary = auto_segment_by_outliers(
                second_moments=second_moments,
                hidden_size=d_model,
                num_layers=num_layers,
                group_size=group_size,
                max_segments=max_segments,
                outlier_log_sigma=outlier_log_sigma,
            )
            print(
                f"[SegPerm] Auto segments: boundary_sizes={boundary_sizes}, "
                f"segment_group_ks={segment_group_ks}"
            )
        else:
            boundary_sizes, segment_group_ks, auto_summary = auto_segment_with_fixed_group_k(
                second_moments=second_moments,
                hidden_size=d_model,
                num_layers=num_layers,
                group_k=fixed_group_k,
                max_segments=max_segments,
                outlier_log_sigma=outlier_log_sigma,
            )
            print(
                f"[SegPerm] Auto segments + fixed global group_k: "
                f"boundary_sizes={boundary_sizes}, group_k={fixed_group_k}"
            )
    else:
        boundary_sizes = [int(x) for x in boundary_sizes]
        assert sum(boundary_sizes) == num_layers, (
            f"sum(boundary_sizes)={sum(boundary_sizes)} != num_hidden_layers={num_layers}"
        )
        if fixed_group_k is None:
            segment_group_ks = _segment_group_ks_for_boundaries(
                second_moments=second_moments,
                hidden_size=d_model,
                boundary_sizes=boundary_sizes,
                group_size=group_size,
                outlier_log_sigma=outlier_log_sigma,
            )
            print(
                f"[SegPerm] Manual segments + auto group_k: boundary_sizes={boundary_sizes}, "
                f"segment_group_ks={segment_group_ks}"
            )
        else:
            outlier_sets = _residual_outlier_sets(second_moments, outlier_log_sigma)
            offsets = _boundary_offsets(boundary_sizes)
            for seg_idx in range(len(boundary_sizes)):
                sources = _segment_sources(offsets[seg_idx], offsets[seg_idx + 1])
                mask = _segment_outlier_mask(sources, outlier_sets, d_model)
                union_count = int(mask.sum().item())
                if union_count > fixed_group_k:
                    print(
                        "[SegPerm] WARNING: "
                        f"Manual segment {seg_idx} L{offsets[seg_idx]}-{offsets[seg_idx + 1] - 1} "
                        f"has {union_count} true residual outliers, larger than fixed "
                        f"group_k={fixed_group_k}; only the selected top-{fixed_group_k} "
                        "channels will be protected."
                    )
            segment_group_ks = [fixed_group_k] * len(boundary_sizes)
            print(
                f"[SegPerm] Manual segments + fixed global group_k + legacy top_k_ratio selection: "
                f"boundary_sizes={boundary_sizes}, group_k={fixed_group_k}, "
                f"top_k_ratio={top_k_ratio:g}"
            )

    num_segments = len(boundary_sizes)
    layer_group_ks = _expand_segment_group_ks(boundary_sizes, segment_group_ks)
    if fixed_group_k is None:
        down_layer_group_ks = _down_layer_group_ks(
            second_moments=second_moments,
            num_layers=num_layers,
            group_size=group_size,
            outlier_log_sigma=down_outlier_log_sigma,
        )
    else:
        for layer_idx in range(num_layers):
            sm = second_moments.get((layer_idx, "down_proj"))
            if sm is None:
                continue
            max_down_group_k = (int(sm.numel()) // group_size) * group_size
            if fixed_group_k > max_down_group_k:
                raise ValueError(
                    f"group_k={fixed_group_k} does not fit layer {layer_idx} down_proj "
                    f"input dim={int(sm.numel())} with group_size={group_size}; "
                    f"max valid group_k is {max_down_group_k}."
                )
            n_outliers = int(_source_outliers(sm, down_outlier_log_sigma).numel())
            if n_outliers > fixed_group_k:
                print(
                    "[SegPerm] WARNING: "
                    f"Layer {layer_idx} down_proj has {n_outliers} true outliers "
                    f"(sigma={down_outlier_log_sigma:g}), larger than fixed group_k={fixed_group_k}. "
                    f"Only the top-{fixed_group_k} down_proj channels will be protected."
                )
        down_layer_group_ks = [fixed_group_k] * num_layers
    max_group_k = int(max(max(segment_group_ks), max(down_layer_group_ks)))
    if fixed_group_k is None:
        print(
            f"[SegPerm] Down-proj per-layer group_k (sigma={down_outlier_log_sigma:g}): "
            f"min={min(down_layer_group_ks)}, max={max(down_layer_group_ks)}, "
            f"values={down_layer_group_ks}"
        )
    else:
        print(
            f"[SegPerm] Down-proj fixed global group_k: "
            f"group_k={fixed_group_k}, values={down_layer_group_ks}"
        )
    fakequant_param_stats = compute_fakequant_param_stats(
        model=model,
        layer_group_ks=layer_group_ks,
        down_layer_group_ks=down_layer_group_ks,
        target_modules=target_modules,
        include_down_proj=True,
    )
    print(
        "[SegPerm] Selective-QAT fakequant parameter coverage: "
        f"{format_fakequant_param_stats(fakequant_param_stats)}"
    )

    # ---- 2b) salient selection ----
    if legacy_topk_ratio_mode:
        residual_salient = select_salient_channels(
            second_moments, d_model, boundary_sizes,
            top_k_ratio=top_k_ratio,
            group_k=int(fixed_group_k),
            group_size=group_size,
            outlier_log_sigma=outlier_log_sigma,
        )
    else:
        residual_salient = select_salient_channels_variable(
            second_moments, d_model, boundary_sizes, segment_group_ks,
            group_size=group_size, outlier_log_sigma=outlier_log_sigma,
        )
    # Reorder WITHIN each segment's salient block before laying it out. Membership is untouched,
    # so group_k / outlier capture / boundary gathers are unaffected — only which channels share a
    # quantization group changes. Only meaningful together with the AWQ fold, which is what makes
    # the salient columns' magnitudes heterogeneous in the first place.
    if reorder_salient:
        residual_salient = reorder_salient_by_post_awq_magnitude(
            model, residual_salient, second_moments, boundary_sizes,
            alpha=awq_alpha, max_s=awq_max,
        )
    segment_perms = {
        k: _build_segment_perm(residual_salient[k], d_model) for k in range(num_segments)
    }
    internal_salient = select_internal_salient_channels(
        second_moments, num_layers,
        group_k=max_group_k,
        down_layer_group_ks=down_layer_group_ks,
    )

    # ---- 2b) AWQ-style per-channel salient scales S (always computed; usage gated by config) ----
    awq_scales = compute_awq_scales(
        second_moments, residual_salient, internal_salient,
        boundary_sizes, num_layers, max_group_k,
        layer_group_ks=layer_group_ks,
        down_layer_group_ks=down_layer_group_ks,
        alpha=awq_alpha, max_s=awq_max,
    )

    # ---- 3) apply the three transforms IN fp16 (dtype-preserving, equivalence-preserving) ----
    boundary_perms = apply_segment_permutation_fp32(model, segment_perms, boundary_sizes)
    block_internal = apply_block_internal_permutations_fp32(model, internal_salient)
    apply_hadamard_rotation_fp32(model, num_layers, num_kv_heads, head_dim)
    # AWQ folding MUST come after both permutations: it addresses the salient slice by position
    # ([0, group_k) for the residual sources, [0, down_group_k) for down_proj), and only the
    # permutations put the salient channels there. It is disjoint from the Hadamard rotation,
    # which touches v_proj OUTPUT rows and o_proj INPUT columns.
    n_awq_folded = apply_awq_folding_fp32(
        model, awq_scales if fold_awq else None, layer_group_ks, down_layer_group_ks,
    ) if fold_awq else 0

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
        "down_layer_group_ks":    list(down_layer_group_ks),
        "fixed_group_k":          fixed_group_k,
        "legacy_topk_ratio_mode": bool(legacy_topk_ratio_mode),
        "top_k_ratio":            float(top_k_ratio),
        "fakequant_param_stats":  fakequant_param_stats,
        "group_size":             group_size,
        "outlier_log_sigma":      float(outlier_log_sigma),
        "down_outlier_log_sigma": down_outlier_log_sigma,
        "boundary_sizes":         list(boundary_sizes),
        "d_model":                d_model,
        "permuted_base_dir":      os.path.abspath(save_dir),
        "auto_segments":          auto_summary,
        # AWQ-style per-channel salient scales (used only when awq_scale is enabled in cfg).
        "awq_scales":             {k: v.cpu() for k, v in awq_scales.items()},
        "awq_alpha":              awq_alpha,
        "awq_max":                awq_max,
        # True => S is ALREADY folded into these saved weights (salient cols amplified, the
        # producing norm / up_proj rows divided). Downstream must then NOT apply S again: the
        # quantizer sees the folded matrix and the export needs no AWQ awareness at all.
        "awq_folded":             bool(fold_awq and n_awq_folded > 0),
        "salient_reordered":      bool(reorder_salient),
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
