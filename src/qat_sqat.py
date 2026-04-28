"""
Selective Salient QAT (SQAT) — asymmetric INT3/INT4 variant.

Compared to the symmetric version this file replaces:

  Symmetric INT-N:
    q ∈ [-q_max, q_max],  q_max = 2^(N-1) − 1
    scale = max(|w|) / q_max
    salient anchor = group abs-max  (one-sided)

  Asymmetric INT-N (this file):
    q ∈ [0, q_lvl],  q_lvl = 2^N − 1
    scale = (w_max − w_min) / q_lvl
    z_int = round(−w_min / scale).clamp(0, q_lvl)
    deq = (q − z_int) * scale
    salient anchor = group signed max OR group signed min  (two-sided)

The "zero error" guarantee for salient channels is preserved by reverse-solving
the scale from whichever end the salient channel anchors.  Because z_int is an
integer, only one side can be made exact analytically; the other side falls
back to STE-aligned grid points.  We choose the larger-magnitude side per
group, so the more impactful end gets the exact treatment.

  If salient is the group max-anchor and the max side is preferred:
      scale = w_max / (q_lvl − z_int)
      → at this channel,  q = q_lvl,  deq = (q_lvl − z_int) * scale = w_max  ✓
  If salient is the group min-anchor and the min side is preferred:
      scale = (−w_min) / z_int
      → at this channel,  q = 0,  deq = (0 − z_int) * scale = w_min  ✓

Saliency-Amplified Deployment Coordinate System (pass3) is unchanged in spirit:
salient channels are scaled by a per-channel gain D before quantization, the
quantizer runs in amplify-space, and the residual is mapped back to original
weight space before injection.  D > 1 boosts both positive and negative
outliers because amplification preserves sign.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .qat_base import QATHandler, round_ste


# ============================================================================
# Activation 2nd-moment outlier analysis (UNCHANGED from symmetric version)
# ============================================================================

def analyze_activation_second_moment_outliers(
    second_moments: Dict[str, torch.Tensor],
    top_k_ratio: float = 0.01,
    log_outlier_sigma: float = 3.0,
    eps: float = 1e-12,
):
    """
    分析 top-k 显著通道是否覆盖了大部分 activation outlier。

    注意:
      不要把 outlier 定义成"更小比例的 top-m"，否则 top-1% 覆盖它们会天然接近 100%，没有信息量。
      这里改用每层 log10(E[x^2]) 上的 robust z-score:
          z = (log_sm - median) / (1.4826 * MAD)
      z >= log_outlier_sigma 视为 outlier。
      如果某层一个 outlier 都没有，就回退到该层 log_sm 的 P99 阈值。
    """
    per_layer = {}

    global_num_channels = 0
    global_num_outliers = 0
    global_num_outlier_hits = 0

    global_total_mass = 0.0
    global_topk_mass = 0.0
    global_outlier_mass = 0.0
    global_hit_outlier_mass = 0.0

    for name, sm in second_moments.items():
        sm = sm.detach().float().flatten().cpu()
        if sm.numel() == 0:
            continue

        n = sm.numel()
        k = max(1, int(n * top_k_ratio))

        _, topk_idx = torch.topk(sm, k, largest=True, sorted=False)
        topk_mask = torch.zeros(n, dtype=torch.bool)
        topk_mask[topk_idx] = True

        log_sm = torch.log10(sm.clamp(min=eps))
        med = log_sm.median()
        mad = (log_sm - med).abs().median().clamp(min=eps)
        robust_std = (1.4826 * mad).clamp(min=eps)

        outlier_mask = ((log_sm - med) / robust_std) >= log_outlier_sigma

        if not outlier_mask.any():
            thr = torch.quantile(log_sm, 0.99)
            outlier_mask = log_sm >= thr

        hit_mask = topk_mask & outlier_mask

        total_mass = sm.sum().clamp(min=eps)
        outlier_mass = sm[outlier_mask].sum().clamp(min=eps)
        hit_outlier_mass = sm[hit_mask].sum()

        sm_median = sm.median().clamp(min=eps)

        per_layer[name] = {
            "num_channels": int(n),
            "topk": int(k),
            "num_outliers": int(outlier_mask.sum().item()),
            "mass_capture": float((sm[topk_mask].sum() / total_mass).item()),
            "outlier_recall": float(
                (hit_mask.sum().float() / outlier_mask.sum().float()).item()
            ),
            "outlier_mass_recall": float((hit_outlier_mass / outlier_mass).item()),
            "max_over_median": float((sm.max() / sm_median).item()),
            "sorted_curve": torch.sort(sm / sm_median, descending=True).values,
        }

        global_num_channels += n
        global_num_outliers += int(outlier_mask.sum().item())
        global_num_outlier_hits += int(hit_mask.sum().item())

        global_total_mass += float(total_mass.item())
        global_topk_mass += float(sm[topk_mask].sum().item())
        global_outlier_mass += float(outlier_mass.item())
        global_hit_outlier_mass += float(hit_outlier_mass.item())

    num_layers = max(len(per_layer), 1)

    global_stats = {
        "num_layers": len(per_layer),
        "num_channels": global_num_channels,
        "num_outliers": global_num_outliers,
        "topk_ratio": float(top_k_ratio),
        "mean_mass_capture": float(
            sum(v["mass_capture"] for v in per_layer.values()) / num_layers
        ),
        "mean_outlier_recall": float(
            sum(v["outlier_recall"] for v in per_layer.values()) / num_layers
        ),
        "mean_outlier_mass_recall": float(
            sum(v["outlier_mass_recall"] for v in per_layer.values()) / num_layers
        ),
        "global_mass_capture": float(global_topk_mass / max(global_total_mass, eps)),
        "global_outlier_recall": float(
            global_num_outlier_hits / max(global_num_outliers, 1)
        ),
        "global_outlier_mass_recall": float(
            global_hit_outlier_mass / max(global_outlier_mass, eps)
        ),
    }

    return {
        "per_layer": per_layer,
        "global": global_stats,
    }


def _resample_curve(curve: torch.Tensor, num_points: int = 256) -> torch.Tensor:
    if curve.numel() == num_points:
        return curve

    idx = torch.linspace(0, curve.numel() - 1, steps=num_points)
    lo = idx.floor().long()
    hi = idx.ceil().long()
    w = idx - lo.float()
    return curve[lo] * (1.0 - w) + curve[hi] * w


def plot_activation_second_moment_statistics(
    analysis: dict,
    save_path: str,
    top_k_ratio: float = 0.01,
    num_points: int = 256,
    max_layers_to_draw: int = 64,
):
    """生成 calibration 统计图（不参与训练图）。"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_layer = analysis["per_layer"]
    global_stats = analysis["global"]

    if len(per_layer) == 0:
        print("[SQAT] No second-moment stats to plot.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    curves = []
    for stats in per_layer.values():
        curves.append(_resample_curve(stats["sorted_curve"], num_points))
    curves = torch.stack(curves, dim=0)

    mean_curve = curves.mean(dim=0)
    q25_curve = torch.quantile(curves, 0.25, dim=0)
    q75_curve = torch.quantile(curves, 0.75, dim=0)

    ranked = sorted(
        per_layer.items(),
        key=lambda kv: kv[1]["outlier_mass_recall"],
        reverse=True,
    )
    top_show = min(20, len(ranked))
    show_items = ranked[:top_show]

    labels = []
    outlier_mass_recalls = []
    mass_captures = []

    for name, stats in show_items:
        short_name = name if len(name) <= 28 else "..." + name[-25:]
        labels.append(short_name)
        outlier_mass_recalls.append(stats["outlier_mass_recall"] * 100.0)
        mass_captures.append(stats["mass_capture"] * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    x = torch.linspace(0, 100, steps=num_points).numpy()

    ax = axes[0]
    for i in range(min(curves.shape[0], max_layers_to_draw)):
        ax.plot(x, curves[i].numpy(), linewidth=0.8, alpha=0.15)
    ax.plot(x, mean_curve.numpy(), linewidth=2.5, label="mean normalized curve")
    ax.fill_between(x, q25_curve.numpy(), q75_curve.numpy(), alpha=0.2, label="p25-p75")
    ax.axvline(top_k_ratio * 100.0, linestyle="--", label=f"top-{top_k_ratio * 100:.1f}%")

    ax.set_yscale("log")
    ax.set_xlabel("Channel rank percentile (descending by E[x^2])")
    ax.set_ylabel("Normalized second moment / layer median")
    ax.set_title("Activation 2nd-moment concentration")

    txt = (
        f"global mass capture = {global_stats['global_mass_capture'] * 100:.1f}%\n"
        f"global outlier recall = {global_stats['global_outlier_recall'] * 100:.1f}%\n"
        f"global outlier mass recall = {global_stats['global_outlier_mass_recall'] * 100:.1f}%"
    )
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend()

    ax = axes[1]
    xpos = list(range(top_show))
    ax.bar(xpos, outlier_mass_recalls, alpha=0.75, label="outlier mass recall (%)")
    ax.plot(xpos, mass_captures, marker="o", linewidth=1.5, label="top-k mass capture (%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Per-layer top-1% coverage")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SQAT] Saved activation 2nd-moment statistics plot to: {save_path}")
    print(
        f"[SQAT] Global top-{top_k_ratio * 100:.1f}% mass capture: "
        f"{global_stats['global_mass_capture'] * 100:.2f}%"
    )
    print(
        f"[SQAT] Global outlier recall: "
        f"{global_stats['global_outlier_recall'] * 100:.2f}%"
    )
    print(
        f"[SQAT] Global outlier mass recall: "
        f"{global_stats['global_outlier_mass_recall'] * 100:.2f}%"
    )


# ============================================================================
# Core Operator: Asymmetric Selective Fakequant (REWRITTEN)
# ============================================================================

def selective_salient_fakequant_asym(
    W_curr: torch.Tensor,            # [N, K]  salient slice in (amp-)space, signed
    group_ids: torch.Tensor,         # [K]
    base_w_max_group: torch.Tensor,  # [N, G]  signed per-group max of NON-salient
    base_w_min_group: torch.Tensor,  # [N, G]  signed per-group min of NON-salient
    q_lvl: int = 15,                 # INT4 → 15, INT3 → 7
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Asymmetric selective fakequant.  Two-sided dynamic anchors with
    integer zero-point, plus reverse-solved scale to keep the salient
    anchor channel exactly representable.

    Math per group:
        w_max = max(base_w_max,  W_curr[salient in this group])
        w_min = min(base_w_min,  W_curr[salient in this group])

        raw_scale = (w_max − w_min) / q_lvl
        z_int     = round(−w_min / raw_scale).clamp(0, q_lvl)

        if salient wins the max side  AND  |w_max| ≥ |w_min|:
            scale = w_max / max(q_lvl − z_int, 1)
            → max-end salient channel reconstructs exactly.
        elif salient wins the min side AND  |w_min| > |w_max|:
            scale = (−w_min) / max(z_int, 1)
            → min-end salient channel reconstructs exactly.
        else:
            scale = raw_scale          (both sides absorbed by integer rounding)

    The exact-end salient channel is then pass-through to remove FP drift,
    matching the symmetric implementation's `is_anchor_mask` semantics.
    """
    N, K = W_curr.shape
    gi = group_ids.unsqueeze(0).expand(N, -1)  # [N, K]

    # --- Step 1: dynamic anchors (signed max & min over all channels) ---
    w_max = base_w_max_group.clone()
    w_max.scatter_reduce_(1, gi, W_curr, reduce="amax", include_self=True)

    w_min = base_w_min_group.clone()
    w_min.scatter_reduce_(1, gi, W_curr, reduce="amin", include_self=True)

    # --- Step 2: which side does salient anchor on? ---
    NEG_INF = torch.finfo(W_curr.dtype).min
    POS_INF = torch.finfo(W_curr.dtype).max

    w_sal_max = torch.full_like(w_max, NEG_INF)
    w_sal_max.scatter_reduce_(1, gi, W_curr, reduce="amax", include_self=True)
    w_sal_min = torch.full_like(w_min, POS_INF)
    w_sal_min.scatter_reduce_(1, gi, W_curr, reduce="amin", include_self=True)

    # salient is max-anchor for this group iff its max ≥ non-salient max,
    # min-anchor iff its min ≤ non-salient min.
    sal_is_max = w_sal_max >= base_w_max_group
    sal_is_min = w_sal_min <= base_w_min_group

    # --- Step 3: choose scale ---
    raw_scale = ((w_max - w_min) / q_lvl).clamp_min(eps)             # [N, G]
    z_int = round_ste((-w_min) / raw_scale).clamp(0, q_lvl)          # [N, G], differentiable

    # Reverse-solved scales.  clamp_min(1) prevents div-by-0 at degenerate
    # corners (z_int == 0 or z_int == q_lvl).  Those corners coincide with
    # use_max=False or use_min=False respectively, so the value isn't used.
    denom_max = (q_lvl - z_int).clamp_min(1.0)
    denom_min = z_int.clamp_min(1.0)
    scale_from_max = w_max / denom_max
    scale_from_min = (-w_min) / denom_min

    # Prefer the side with larger magnitude — it dominates the quant error.
    prefer_max_side = w_max.abs() >= w_min.abs()
    use_max = sal_is_max & prefer_max_side
    use_min = sal_is_min & (~prefer_max_side)

    scale = raw_scale
    scale = torch.where(use_max, scale_from_max, scale)
    scale = torch.where(use_min, scale_from_min, scale)
    scale = scale.clamp_min(eps)

    # --- Step 4: quantize ---
    s_k = scale.gather(1, gi)        # [N, K]
    z_k = z_int.gather(1, gi)        # [N, K]

    q = round_ste(W_curr / s_k + z_k).clamp(0, q_lvl)
    W_quant = (q - z_k) * s_k

    # --- Step 5: anchor pass-through (kill FP drift on the exact end) ---
    w_max_k = w_max.gather(1, gi)
    w_min_k = w_min.gather(1, gi)
    use_max_k = use_max.gather(1, gi)
    use_min_k = use_min.gather(1, gi)

    is_max_anchor = use_max_k & (W_curr >= w_max_k - 1e-5)
    is_min_anchor = use_min_k & (W_curr <= w_min_k + 1e-5)
    is_anchor = is_max_anchor | is_min_anchor

    return torch.where(is_anchor, W_curr, W_quant)


# ============================================================================
# Saliency-Amplified Quantization Residual (UPDATED for asymmetric)
# ============================================================================

def saliency_amplified_quant_residual(
    W_curr_salient: torch.Tensor,        # [out, K] original space
    salient_gain: torch.Tensor,          # [K] (currently unused inside; kept for API parity)
    group_ids: torch.Tensor,             # [K]
    base_w_max_group: torch.Tensor,      # [out, G]
    base_w_min_group: torch.Tensor,      # [out, G]
    q_lvl: int = 15,
) -> torch.Tensor:
    """
    Compute the asymmetric quantization residual in amplify-space and return
    it expressed in the original weight space.

    NOTE on `salient_gain`:
      To keep the implementation simple and faithful to the symmetric version,
      W_curr_salient is expected to ALREADY live in amplify-space when this
      function is called.  In practice the SQAT linear module does the
      multiplication itself (see SelectiveSalientQATLinear.forward), and the
      base_w_max_group / base_w_min_group buffers are amp-space quantities by
      construction (non-salient channels have D = 1, so amp-space and
      original-space coincide on them).

    Returns:
        delta_salient [out, K]: residual (W_quant − W_curr) in amp-space.
                                (When all D = 1 this equals the original-space
                                 residual; otherwise the SQAT module divides
                                 by D before injecting.)
    """
    W_curr_quant = selective_salient_fakequant_asym(
        W_curr=W_curr_salient,
        group_ids=group_ids,
        base_w_max_group=base_w_max_group,
        base_w_min_group=base_w_min_group,
        q_lvl=q_lvl,
    )
    return W_curr_quant - W_curr_salient


# ============================================================================
# Calibration: Activation 2nd Moment Estimation (UNCHANGED)
# ============================================================================

@torch.no_grad()
def estimate_activation_second_moment(
    model: nn.Module,
    dataloader: DataLoader,
    target_modules: list,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """One-pass calibration: collect E[x_j^2] per input channel per target layer."""
    accumulators = {}
    counts = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            x_sq = x.pow(2).mean(dim=tuple(range(x.dim() - 1)))  # [in_features]
            if name not in accumulators:
                accumulators[name] = torch.zeros_like(x_sq)
                counts[name] = 0
            accumulators[name].add_(x_sq)
            counts[name] += 1
        return hook_fn

    for name, module in model.named_modules():
        name_parts = name.split(".")
        if not name_parts:
            continue
        terminal = name_parts[-1]
        if terminal not in target_modules:
            continue
        if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not hooks:
        print("[SQAT] WARNING: No LoRA modules found for calibration. "
              "Check target_modules config matches actual PEFT module names.")

    model.eval()
    for batch in tqdm(dataloader, desc="[SQAT] Calibrating activation 2nd moments"):
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model(**batch)

    for h in hooks:
        h.remove()

    result = {}
    for name, acc in accumulators.items():
        result[name] = acc / counts[name]
    return result


def select_salient_channels(
    second_moments: Dict[str, torch.Tensor],
    top_k_ratio: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Select top-k channels per layer by 2nd moment magnitude."""
    result = {}
    for name, sm in second_moments.items():
        in_features = sm.shape[0]
        k = max(1, int(in_features * top_k_ratio))
        _, indices = sm.topk(k)
        result[name] = indices.sort().values
    return result


# ============================================================================
# Precomputation Utilities (REWRITTEN for asymmetric)
# ============================================================================

def compute_base_minmax_group(
    W_dequant: torch.Tensor,
    salient_indices: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-group SIGNED max/min of NON-salient channels (the anchor floors).

    Padding columns and salient columns are excluded from the reduction:
      - excluded positions are set to −inf for amax / +inf for amin
      - if a whole group is degenerate (impossible in practice), we
        substitute zeros to keep downstream math finite.

    Returns:
        (base_w_max_group, base_w_min_group), each [out_features, num_groups]
    """
    out_f, in_f = W_dequant.shape
    num_groups = math.ceil(in_f / group_size)
    device = W_dequant.device

    # valid: True ⇔ not salient AND not padding
    total = num_groups * group_size
    valid = torch.ones(total, dtype=torch.bool, device=device)
    valid[in_f:] = False
    valid[salient_indices.to(device)] = False

    pad = total - in_f
    if pad > 0:
        W_padded = F.pad(W_dequant, (0, pad), value=0.0)
    else:
        W_padded = W_dequant

    W_grouped = W_padded.view(out_f, num_groups, group_size)
    valid_g = valid.view(num_groups, group_size).unsqueeze(0)  # [1, G, gs]

    NEG_INF = torch.finfo(W_grouped.dtype).min
    POS_INF = torch.finfo(W_grouped.dtype).max

    W_for_max = W_grouped.masked_fill(~valid_g, NEG_INF)
    W_for_min = W_grouped.masked_fill(~valid_g,  POS_INF)

    base_max = W_for_max.amax(dim=2)
    base_min = W_for_min.amin(dim=2)

    # Replace the (impossible-in-practice) "all invalid" sentinel with 0
    # so downstream w_max - w_min math stays finite.
    base_max = torch.where(base_max <= NEG_INF / 2, torch.zeros_like(base_max), base_max)
    base_min = torch.where(base_min >= POS_INF / 2, torch.zeros_like(base_min), base_min)

    return base_max, base_min


def dequantize_layer(module: nn.Module) -> torch.Tensor:
    """Extract dequantized weight from a (possibly PEFT-wrapped) linear layer."""
    if hasattr(module, 'base_layer'):
        base = module.base_layer
    else:
        base = module

    out_features = base.out_features
    in_features = base.in_features

    weight = base.weight
    if hasattr(weight, 'quant_state') and weight.quant_state is not None:
        import bitsandbytes as bnb
        W = bnb.functional.dequantize_4bit(
            weight.data, weight.quant_state
        ).float()
        if W.shape == (out_features, in_features):
            return W.detach()
        elif W.shape == (in_features, out_features):
            return W.t().contiguous().detach()
        elif W.numel() == out_features * in_features:
            return W.reshape(out_features, in_features).detach()
        else:
            raise RuntimeError(
                f"BnB dequantize_4bit returned shape {W.shape}, "
                f"expected ({out_features}, {in_features})"
            )

    if hasattr(base, 'dequantize'):
        W = base.dequantize().detach().float()
        if W.shape == (out_features, in_features):
            return W
        elif W.shape == (in_features, out_features):
            return W.t().contiguous()
        elif W.numel() == out_features * in_features:
            return W.reshape(out_features, in_features)

    if hasattr(weight, 'data') and weight.data.shape == (out_features, in_features):
        return weight.data.detach().float()

    raise RuntimeError(
        f"Cannot dequantize layer: {type(base)}, "
        f"weight shape: {weight.shape}, "
        f"expected: ({out_features}, {in_features})"
    )


# ============================================================================
# SQAT Linear Module (REWRITTEN — asymmetric buffers)
# ============================================================================

class SelectiveSalientQATLinear(nn.Module):
    """
    Drop-in wrapper that injects an asymmetric saliency-amplified deployment
    quantizer residual.

    Forward:
      pass1+2: Y = original_qlora_forward(X)          # NF4 base + global LoRA
      pass3:   Y += X[..., S] @ delta_S.T             # asymmetric quant residual

    pass3 residual:
      W_curr_S  = W_base_S + scaling * B @ A[:, S]    (original space, salient slice)
      W_amp     = W_curr_S * D                          (amplify-space)
      W_amp_Q   = selective_salient_fakequant_asym(W_amp, ...)
      delta_S   = (W_amp_Q − W_amp) / D                 (back to original space)

    Two anchor sides per group: a salient channel can be the group max-anchor
    (zero error on the +q_lvl rail) or min-anchor (zero error on the 0 rail),
    or — if it's not the group extreme — STE-aligned to a grid point.
    """

    def __init__(
        self,
        original_module: nn.Module,
        salient_indices: torch.Tensor,         # [K]
        W_base_salient: torch.Tensor,          # [out, K]
        base_w_max_group: torch.Tensor,        # [out, G]
        base_w_min_group: torch.Tensor,        # [out, G]
        salient_group_ids: torch.Tensor,       # [K]
        q_bits: int = 4,
        lora_scaling: float = 1.0,
        salient_gain: Optional[torch.Tensor] = None,  # [K], default all-ones
    ):
        super().__init__()
        self.original_module = original_module
        self.lora_scaling = lora_scaling

        K = salient_indices.shape[0]

        # Frozen / refreshable buffers
        self.register_buffer("salient_indices", salient_indices)
        self.register_buffer("W_base_salient", W_base_salient)
        self.register_buffer("base_w_max_group", base_w_max_group)
        self.register_buffer("base_w_min_group", base_w_min_group)
        self.register_buffer("salient_group_ids", salient_group_ids)

        if salient_gain is None:
            salient_gain = torch.ones(K)
        self.register_buffer("salient_gain", salient_gain)

        # Asymmetric grid: q ∈ [0, q_lvl], q_lvl = 2^bits − 1
        self.q_bits = q_bits
        self.q_lvl = 2 ** q_bits - 1
        self.has_lora = hasattr(original_module, 'lora_A')

    def set_salient_gain(self, gain: torch.Tensor) -> None:
        """Update the per-channel amplification vector D in-place."""
        assert gain.shape == self.salient_gain.shape, (
            f"salient_gain shape mismatch: "
            f"expected {self.salient_gain.shape}, got {gain.shape}"
        )
        self.salient_gain.copy_(gain.to(self.salient_gain.device))

    def _get_lora_B_weight(self) -> torch.Tensor:
        adapter_name = list(self.original_module.lora_B.keys())[0]
        return self.original_module.lora_B[adapter_name].weight

    def _get_lora_A_salient(self) -> torch.Tensor:
        """A[:, S]: [rank, K]."""
        adapter_name = list(self.original_module.lora_A.keys())[0]
        return self.original_module.lora_A[adapter_name].weight[:, self.salient_indices]

    def _get_BA_salient(self) -> torch.Tensor:
        """B @ A[:, S]: [out, K]."""
        return self._get_lora_B_weight() @ self._get_lora_A_salient()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # --- Path 1+2: Original QLoRA forward (unchanged) ---
        Y = self.original_module(x, *args, **kwargs)

        if not self.has_lora:
            return Y

        # --- Path 3: asymmetric saliency-amplified residual ---
        X_S = x[..., self.salient_indices]  # [..., K], original-space activations

        # Original-space effective weight on salient channels
        W_curr_S = (
            self.W_base_salient + self._get_BA_salient() * self.lora_scaling
        )  # [out, K]

        # Enter amp-space: per-channel multiplication, sign-preserving
        D = self.salient_gain                              # [K]
        W_amp = W_curr_S * D                               # [out, K]

        # Quantize in amp-space; base buffers are amp-space quantities
        # (non-salient channels have D = 1, so amp-space ≡ original-space on them).
        W_amp_quant = selective_salient_fakequant_asym(
            W_curr=W_amp,
            group_ids=self.salient_group_ids,
            base_w_max_group=self.base_w_max_group,
            base_w_min_group=self.base_w_min_group,
            q_lvl=self.q_lvl,
        )  # [out, K], amp-space

        # Map rounding error back to original space:  delta = (W_amp_Q − W_amp) / D
        D_safe = D.clamp_min(1e-7)
        delta_S = (W_amp_quant - W_amp) / D_safe           # [out, K], original space

        Y = Y + F.linear(X_S, delta_S)
        return Y


# ============================================================================
# SQAT Handler (UPDATED: builds two min/max buffers, refreshes both)
# ============================================================================

class SelectiveSalientQAT(QATHandler):
    """
    Selective Salient QAT handler — asymmetric INT3/INT4 variant.
    """

    def __init__(self):
        self.patched_layers: Dict[str, SelectiveSalientQATLinear] = {}

    def prepare_model(
        self,
        model: nn.Module,
        cfg: dict,
        tokenizer=None,
        calibration_dataloader=None,
        salient_gain_map: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> nn.Module:
        sqat_cfg = cfg["qat"]["sqat"]
        target_modules = cfg["lora"]["target_modules"]
        q_bits = cfg["model"]["quant_bits"]
        group_size = cfg["qat"].get("group_size", 128)
        top_k_ratio = sqat_cfg["top_k_ratio"]
        device = next(model.parameters()).device

        lora_cfg = cfg["lora"]
        lora_scaling = lora_cfg["alpha"] / lora_cfg["rank"]

        self._refresh_interval = sqat_cfg.get("base_max_refresh_interval", 0)
        self._group_size = group_size
        if self._refresh_interval > 0:
            print(f"[SQAT] base min/max refresh every {self._refresh_interval} steps")

        # --- Step 1: Calibration ---
        assert calibration_dataloader is not None, (
            "SQAT requires a calibration_dataloader. "
            "Pass it via prepare_model(calibration_dataloader=...)."
        )

        second_moments = estimate_activation_second_moment(
            model, calibration_dataloader, target_modules, device=str(device),
        )
        # --- Step 1b: Analyze + plot activation 2nd-moment statistics ---
        activation_analysis = analyze_activation_second_moment_outliers(
            second_moments=second_moments,
            top_k_ratio=top_k_ratio,
            log_outlier_sigma=sqat_cfg.get("outlier_log_sigma", 3.0),
        )
        self.activation_analysis = activation_analysis

        if sqat_cfg.get("plot_activation_stats", True):
            plot_activation_second_moment_statistics(
                analysis=activation_analysis,
                save_path=sqat_cfg.get(
                    "activation_stats_plot_path",
                    "debug/sqat_activation_second_moment.png",
                ),
                top_k_ratio=top_k_ratio,
                num_points=sqat_cfg.get("activation_plot_points", 256),
                max_layers_to_draw=sqat_cfg.get("activation_plot_max_layers", 64),
            )

        # --- Step 2: Select salient channels ---
        salient_map = select_salient_channels(second_moments, top_k_ratio)

        print(f"[SQAT] Selected salient channels for {len(salient_map)} layers "
              f"(top-{top_k_ratio*100:.1f}%)")
        for lname, sidx in list(salient_map.items())[:3]:
            print(f"  [SQAT]   {lname}: K={sidx.shape[0]}, "
                  f"max_idx={sidx.max().item()}, min_idx={sidx.min().item()}")

        # --- Step 2c: Identify modules to patch ---
        modules_to_patch = {}
        for name, module in model.named_modules():
            if name in salient_map:
                if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
                    modules_to_patch[name] = module

        # --- Step 2b: Auto-build salient_gain_map (AWQ-faithful per-channel search) ---
        gain_alpha = sqat_cfg.get("salient_gain_alpha", 0.0)
        gain_max_cap = sqat_cfg.get("salient_gain_max", 2.0)
        if salient_gain_map is None and gain_alpha > 0.0:
            salient_gain_map = {}
            for layer_name, sidx in salient_map.items():
                if layer_name not in second_moments:
                    continue
                sm = second_moments[layer_name].to(device)

                module_for_layer = modules_to_patch.get(layer_name)
                if module_for_layer is None:
                    continue
                W_full = dequantize_layer(module_for_layer)
                s_w = W_full.abs().mean(dim=0).to(device)

                s_x = sm.sqrt()
                s_x_sal = s_x[sidx].clamp(min=1e-7)
                s_w_sal = s_w[sidx].clamp(min=1e-7)
                raw_D = (s_x_sal / s_w_sal).pow(gain_alpha)

                raw_D = raw_D / raw_D.min().clamp(min=1e-7)
                raw_D = raw_D.clamp(min=1.0, max=gain_max_cap)
                salient_gain_map[layer_name] = raw_D

            if salient_gain_map:
                all_maxes = [v.max().item() for v in salient_gain_map.values()]
                all_means = [v.mean().item() for v in salient_gain_map.values()]
                print(f"[SQAT] Built salient_gain_map (AWQ-style, alpha={gain_alpha}, "
                      f"cap={gain_max_cap}, {len(salient_gain_map)} layers, "
                      f"avg max_D={sum(all_maxes)/len(all_maxes):.3f}, "
                      f"avg mean_D={sum(all_means)/len(all_means):.3f})")

        # --- Step 3+4: Patch layers ---
        for name, module in modules_to_patch.items():
            salient_idx = salient_map[name].to(device)

            W_dequant = dequantize_layer(module)
            out_f, in_f = W_dequant.shape

            max_idx = salient_idx.max().item()
            assert max_idx < in_f, (
                f"[SQAT] Layer {name}: salient index {max_idx} >= in_features {in_f}."
            )

            W_base_salient = W_dequant[:, salient_idx].to(device)

            group_ids = (salient_idx // group_size).to(device)
            base_w_max, base_w_min = compute_base_minmax_group(
                W_dequant, salient_idx, group_size
            )
            base_w_max = base_w_max.to(device)
            base_w_min = base_w_min.to(device)

            gain = None
            if salient_gain_map is not None and name in salient_gain_map:
                gain = salient_gain_map[name].to(device)

            sqat_linear = SelectiveSalientQATLinear(
                original_module=module,
                salient_indices=salient_idx,
                W_base_salient=W_base_salient,
                base_w_max_group=base_w_max,
                base_w_min_group=base_w_min,
                salient_group_ids=group_ids,
                q_bits=q_bits,
                lora_scaling=lora_scaling,
                salient_gain=gain,
            ).to(device)

            self.patched_layers[name] = sqat_linear

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], sqat_linear)
            else:
                setattr(model, name, sqat_linear)

        print(f"[SQAT] Patched {len(self.patched_layers)} layers with SQAT (asym INT{q_bits}) wrappers.")
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        """Periodically refresh base_w_max_group/base_w_min_group to track LoRA drift."""
        if not hasattr(self, '_refresh_interval'):
            return
        if self._refresh_interval <= 0 or step % self._refresh_interval != 0:
            return
        self._refresh_base_minmax(model, step)

    @torch.no_grad()
    def _refresh_base_minmax(self, model, step):
        """Recompute (base_w_max, base_w_min) from current W_base + LoRA delta."""
        refreshed = 0
        for name, sqat_layer in self.patched_layers.items():
            if not sqat_layer.has_lora:
                continue
            W_base_full = dequantize_layer(sqat_layer.original_module)
            adapter_name = list(sqat_layer.original_module.lora_A.keys())[0]
            A_full = sqat_layer.original_module.lora_A[adapter_name].weight.data.float()
            B_full = sqat_layer.original_module.lora_B[adapter_name].weight.data.float()
            lora_delta = (B_full @ A_full) * sqat_layer.lora_scaling
            W_curr = W_base_full + lora_delta

            new_max, new_min = compute_base_minmax_group(
                W_curr,
                sqat_layer.salient_indices,
                self._group_size,
            )
            sqat_layer.base_w_max_group.copy_(new_max.to(sqat_layer.base_w_max_group.device))
            sqat_layer.base_w_min_group.copy_(new_min.to(sqat_layer.base_w_min_group.device))
            refreshed += 1

        if refreshed > 0 and step > 0:
            print(f"[SQAT] Step {step}: refreshed base_w_max/min_group for {refreshed} layers")

    def on_train_end(self, model):
        if hasattr(self, '_refresh_interval') and self._refresh_interval > 0:
            self._refresh_base_minmax(model, step=-1)