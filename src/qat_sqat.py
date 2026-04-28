"""
Selective Salient QAT (SQAT) — integrated into the training framework.

Flow:
  1. Pre-training calibration: estimate per-channel activation 2nd moment
  2. Select top-k salient channels per linear layer
  3. Patch each target layer with SelectiveSalientQATLinear
  4. Train with error injection on salient channels only

Design simplification (vs. earlier A_salient approach):
  Instead of maintaining an independent A_salient parameter, we directly use
  global LoRA A's salient columns. This eliminates:
    - The A_salient parameter and its initialization
    - Gradient zeroing hooks on global A
    - sync_A_salient_back() at training end
    - The double-contribution bug entirely — there is only ONE A matrix

  The correction term becomes a pure quantization residual:
    delta = Q(W_base_s + scaling * B @ A[:, S]) - (W_base_s + scaling * B @ A[:, S])
  which is exactly the rounding error that LoRA learns to compensate via normal
  backprop through STE.

  The column-index operation A[:, S] on a [rank, d_in] tensor to get [rank, K]
  where K~40 is negligible compared to any matmul in the forward pass.

Saliency-Amplified Deployment Coordinate System (pass3):
  For salient channels, pass3 now operates in an auxiliary amplify-space in which
  each salient channel j is scaled by a per-channel gain D[j] before quantization:

      W_amp       = W_curr_S * D               (amplify-space)
      W_amp_quant = selective_salient_fakequant(W_amp, ...)
      delta_S     = (W_amp_quant / D) - W_curr_S   (mapped back to original space)

  This makes salient channels dominate their quantization groups (higher chance of
  being the group-max anchor), while the injected residual is still expressed in
  the original weight space. pass1 and pass2 are untouched.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .qat_base import QATHandler, round_ste, symmetric_fakequant

# statistics for salient channel selection
def analyze_activation_second_moment_outliers(
    second_moments: Dict[str, torch.Tensor],
    top_k_ratio: float = 0.01,
    log_outlier_sigma: float = 3.0,
    eps: float = 1e-12,
):
    """
    分析 top-k 显著通道是否覆盖了大部分 activation outlier。

    注意:
      不要把 outlier 定义成“更小比例的 top-m”，否则 top-1% 覆盖它们会天然接近 100%，没有信息量。
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

        # top-k mask
        _, topk_idx = torch.topk(sm, k, largest=True, sorted=False)
        topk_mask = torch.zeros(n, dtype=torch.bool)
        topk_mask[topk_idx] = True

        # robust outlier detection on log10 second moment
        log_sm = torch.log10(sm.clamp(min=eps))
        med = log_sm.median()
        mad = (log_sm - med).abs().median().clamp(min=eps)
        robust_std = (1.4826 * mad).clamp(min=eps)

        outlier_mask = ((log_sm - med) / robust_std) >= log_outlier_sigma

        # fallback: if no robust outlier is found, use p99
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

        # 平均每层的统计
        "mean_mass_capture": float(
            sum(v["mass_capture"] for v in per_layer.values()) / num_layers
        ),
        "mean_outlier_recall": float(
            sum(v["outlier_recall"] for v in per_layer.values()) / num_layers
        ),
        "mean_outlier_mass_recall": float(
            sum(v["outlier_mass_recall"] for v in per_layer.values()) / num_layers
        ),

        # 全局按质量加权的统计
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
    """
    生成一张 calibration 统计图:
      左图: 各层按二阶矩降序后的归一化曲线（y 轴 log）
      右图: 每层 top-1% 的 outlier_mass_recall / mass_capture

    保存 png，不参与训练图，不引入训练时开销。
    """
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

    # ---- left panel: normalized sorted curves ----
    curves = []
    for stats in per_layer.values():
        curves.append(_resample_curve(stats["sorted_curve"], num_points))
    curves = torch.stack(curves, dim=0)

    mean_curve = curves.mean(dim=0)
    q25_curve = torch.quantile(curves, 0.25, dim=0)
    q75_curve = torch.quantile(curves, 0.75, dim=0)

    # ---- right panel: per-layer coverage ----
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

    # ---- draw ----
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
# Core Operator (vectorized, no Python loops)
# ============================================================================

def selective_salient_fakequant(W_curr, group_ids, base_max_group, q_max=7.0):
    """
    Dynamic-anchor selective fakequant.  Operates entirely in the amplify-space
    coordinate system: W_curr is always W_amp = W_curr_salient * D.

    base_max_group is the per-group scale floor for non-salient channels.
    It is an amplify-space quantity: numerically equal to the original-space
    non-salient per-group max because non-salient channels have D = 1 and their
    amplify-space magnitudes coincide with their original-space magnitudes.

    Args:
        W_curr:         [N, K]   salient-channel weight slice in amplify-space.
        group_ids:      [K]      quantization group index per salient channel.
        base_max_group: [N, G]   per-group scale floor (non-salient amp-space max).
        q_max:          float    symmetric clamp bound (e.g. 7.0 for INT4).
    """
    N, K = W_curr.shape
    abs_W = W_curr.abs()                                     # [N, K]

    group_indices = group_ids.unsqueeze(0).expand(N, -1)     # [N, K], zero-copy view

    # Fused anchor computation:
    # Initialise from base_max_group (the amplify-space floor), then scatter-reduce
    # the salient abs values into each group.  include_self=True takes max with the
    # floor values in a single pass, replacing the earlier zeros_like + scatter +
    # torch.maximum sequence and saving one tensor allocation.
    anchor = base_max_group.clone()                          # [N, G], starts at floor
    anchor.scatter_reduce_(
        dim=1, index=group_indices, src=abs_W,
        reduce="amax",
        include_self=True,
    )                                                        # anchor[n,g] = max(base[n,g], abs_W salient in g)

    anchor_expanded = anchor.gather(dim=1, index=group_indices)   # [N, K]
    scale = (anchor_expanded / q_max).clamp_(min=1e-7)            # [N, K], in-place clamp

    # Anchor channels (group max in amp-space) pass through unchanged.
    # For the exact max, W_curr/scale == ±q_max and round_ste gives the same result,
    # so W_quant == W_curr mathematically; the mask absorbs floating-point drift.
    is_anchor_mask = abs_W >= (anchor_expanded - 1e-5)            # [N, K] bool
    W_quant = round_ste(torch.clamp(W_curr / scale, -q_max, q_max)) * scale

    return torch.where(is_anchor_mask, W_curr, W_quant)


# ============================================================================
# Saliency-Amplified Quantization Residual (Auxiliary Deployment Coordinate System)
# ============================================================================

def saliency_amplified_quant_residual(
    W_curr_salient: torch.Tensor,   # [out, K]  — original coordinate space
    salient_gain: torch.Tensor,     # [K]       — per-channel amplification D
    group_ids: torch.Tensor,        # [K]
    base_max_group: torch.Tensor,   # [out, G]
    q_max: float = 7.0,
) -> torch.Tensor:
    """
    Compute the quantization residual in the amplify-space (auxiliary deployment
    coordinate system), then map back to original weight space.

    Math:
        W_amp       = W_curr_salient * D                   (enter amplify-space)
        W_amp_quant = selective_salient_fakequant(W_amp, ...)
        delta       = (W_amp_quant - W_amp) / D            (amp-space error → original space)
                    = (W_amp_quant / D) - W_curr_salient   (algebraically equivalent)

    When salient_gain is all-ones this reduces to delta = Q(W_curr_salient) - W_curr_salient.

    Args:
        W_curr_salient: salient-channel slice of current effective weight
                        (W_base_S + scaling * B @ A[:,S]).  Shape [out, K].
                        Original coordinate space.
        salient_gain:   per-channel amplification D.  Shape [K].
                        All-ones disables amplification.
        group_ids:      quantization group index per salient channel [K].
        base_max_group: per-group scale floor.  Shape [out, G].
                        Amplify-space quantity: its values equal the original-space
                        non-salient per-group max because non-salient channels have
                        D = 1, so their amp-space and original-space magnitudes
                        are identical by construction.
        q_max:          symmetric clamp bound.

    Returns:
        delta_salient [out, K]: residual in original weight space.
                                Inject as  Y += X_S @ delta_salient.T.
    """
    # Enter amplify-space: salient_gain [K] broadcasts over the out dimension
    # W_amp = W_curr_salient * salient_gain     # [out, K]

    # Quantize in amplify-space; dynamic anchor reflects amp-space magnitudes
    W_curr_quant = selective_salient_fakequant(
        W_curr=W_curr_salient,
        group_ids=group_ids,
        base_max_group=base_max_group,
        q_max=q_max,
    )                                         # [out, K], amp-space

    # Map rounding error back to original space.
    # (W_amp_quant - W_amp) is the quantization error expressed in amp-space;
    # dividing by D converts it to original-space error.
    # Do NOT inject the amp-space error (W_amp_quant - W_amp) directly.
    # D_safe = salient_gain.clamp(min=1e-7)     # [K]; guard for externally-set gains < 1
    # return (W_amp_quant - W_curr_salient) / D_safe     # [out, K], original space
    return W_curr_quant - W_curr_salient

# ============================================================================
# Calibration: Activation 2nd Moment Estimation
# ============================================================================

@torch.no_grad()
def estimate_activation_second_moment(
    model: nn.Module,
    dataloader: DataLoader,
    target_modules: list,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    One-pass calibration: collect E[x_j^2] per input channel per target layer.

    Returns:
        Dict[layer_name -> Tensor[in_features]] of 2nd moments
    """
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
# Precomputation Utilities
# ============================================================================

def compute_base_max_group(
    W_dequant: torch.Tensor,
    salient_indices: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Per-group max of non-salient channels (floor for dynamic anchor)."""
    out_f, in_f = W_dequant.shape
    num_groups = math.ceil(in_f / group_size)

    mask = torch.ones(in_f, dtype=torch.bool, device=W_dequant.device)
    mask[salient_indices] = False

    pad = num_groups * group_size - in_f
    if pad > 0:
        W_padded = F.pad(W_dequant.abs(), (0, pad))
        mask = F.pad(mask, (0, pad), value=True)
    else:
        W_padded = W_dequant.abs()

    W_grouped = W_padded.view(out_f, num_groups, group_size)
    mask_grouped = mask.view(num_groups, group_size).unsqueeze(0).float()
    W_grouped = W_grouped * mask_grouped

    return W_grouped.amax(dim=2)


def dequantize_layer(module: nn.Module) -> torch.Tensor:
    """
    Extract dequantized weight from a (possibly PEFT-wrapped) linear layer.
    Returns: [out_features, in_features] float tensor.
    """
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
# SQAT Linear Module
# ============================================================================

class SelectiveSalientQATLinear(nn.Module):
    """
    Drop-in wrapper that injects a saliency-amplified deployment quantizer residual.

    Forward:
      pass1+2: Y = original_qlora_forward(X)          # NF4 base + global LoRA (all channels)
      pass3:   Y += X[..., S] @ delta_S.T             # amplify-space quant residual, salient only

    pass3 residual (saliency-amplified deployment coordinate system):
      W_curr_S  = W_base_S + scaling * B @ A[:, S]   (original space, salient slice)
      W_amp     = W_curr_S * D                         (enter amplify-space; D = salient_gain [K])
      W_amp_Q   = selective_salient_fakequant(W_amp)   (quantize in amplify-space)
      delta_S   = (W_amp_Q / D) - W_curr_S             (map back to original space)

    Key properties:
      - D boosts salient channels so they dominate group-max anchors in the quantizer.
      - The injected residual is in the ORIGINAL weight space (not amplify-space).
      - pass1/pass2 are completely unchanged.
      - salient_gain defaults to all-ones (no amplification).

    No independent A_salient parameter -- we reuse global A[:, S] directly.
    Gradient flows through:
      - Path 1+2 (standard QLoRA): task loss drives all of A including salient cols
      - Path 3 (residual injection): STE pushes salient cols of A toward grid-friendly values
    Both gradients accumulate naturally on the same A parameter.
    """

    def __init__(
        self,
        original_module: nn.Module,          # the PEFT LoRA linear
        salient_indices: torch.Tensor,       # [K]
        W_base_salient: torch.Tensor,        # [out, K]
        base_max_group: torch.Tensor,        # [out, num_groups]
        salient_group_ids: torch.Tensor,     # [K]
        q_bits: int = 4,
        lora_scaling: float = 1.0,
        salient_gain: Optional[torch.Tensor] = None,  # [K], default all-ones
    ):
        super().__init__()
        self.original_module = original_module
        self.lora_scaling = lora_scaling

        K = salient_indices.shape[0]

        # Frozen buffers
        self.register_buffer("salient_indices", salient_indices)
        self.register_buffer("W_base_salient", W_base_salient)
        self.register_buffer("base_max_group", base_max_group)
        self.register_buffer("salient_group_ids", salient_group_ids)

        # Per-channel amplification D for the auxiliary deployment coordinate system.
        # Shape [K] — one scalar per salient channel.
        # All-ones means no amplification (pass3 == original behaviour).
        # Update via set_salient_gain() or by passing a non-None tensor here.
        if salient_gain is None:
            salient_gain = torch.ones(K)
        self.register_buffer("salient_gain", salient_gain)

        self.q_max = 2 ** (q_bits - 1) - 1
        self.has_lora = hasattr(original_module, 'lora_A')

    def set_salient_gain(self, gain: torch.Tensor) -> None:
        """
        Update the per-channel amplification vector D in-place.

        Args:
            gain: [K] tensor.  Will be moved to the device of existing buffers.
                  All-ones disables amplification; values > 1 boost those channels
                  in the amplify-space quantizer.
        """
        assert gain.shape == self.salient_gain.shape, (
            f"salient_gain shape mismatch: "
            f"expected {self.salient_gain.shape}, got {gain.shape}"
        )
        self.salient_gain.copy_(gain.to(self.salient_gain.device))

    def _get_lora_B_weight(self) -> torch.Tensor:
        adapter_name = list(self.original_module.lora_B.keys())[0]
        return self.original_module.lora_B[adapter_name].weight

    def _get_lora_A_salient(self) -> torch.Tensor:
        """A[:, S]: [rank, K] -- just an index into the global A."""
        adapter_name = list(self.original_module.lora_A.keys())[0]
        return self.original_module.lora_A[adapter_name].weight[:, self.salient_indices]

    def _get_BA_salient(self) -> torch.Tensor:
        """B @ A[:, S]: [out, K]. Cost O(out * rank * K)."""
        return self._get_lora_B_weight() @ self._get_lora_A_salient()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # --- Path 1+2: Original QLoRA forward (NF4 base + LoRA, all channels) ---
        # Completely unchanged.
        Y = self.original_module(x, *args, **kwargs)

        if not self.has_lora:
            return Y

        # --- Path 3: Saliency-amplified deployment quantizer residual injection ---
        #
        # We work in an auxiliary "amplify-space" coordinate system where each
        # salient channel j is scaled by salient_gain[j] = D[j] before quantization.
        # This makes salient channels dominate their quantization group anchors.
        #
        # The residual is computed in amplify-space but mapped BACK to original
        # weight space before injection, so pass1/pass2 see no amplification:
        #
        #   W_curr_S  = W_base_S + scaling * B @ A[:, S]   (original space)
        #   W_amp     = W_curr_S * D                         (amplify-space)
        #   W_amp_Q   = selective_salient_fakequant(W_amp)   (quantize in amplify-space)
        #   delta_S   = (W_amp_Q / D) - W_curr_S             (back to original space)
        #
        # Y += X_S @ delta_S.T  injects the deployment-quantizer error expressed
        # in original weight space.  When D=1 everywhere this is identical to the
        # original formula  delta_S = Q(W_curr_S) - W_curr_S.

        X_S = x[..., self.salient_indices]  # [..., K], original-space activations

        # Original-space effective weight on salient channels
        W_curr_S = (
            self.W_base_salient + self._get_BA_salient() * self.lora_scaling
        )  # [out, K]

        # Compute residual: amplify -> quantize -> map back -> subtract original
        delta_S = saliency_amplified_quant_residual(
            W_curr_salient=W_curr_S,
            salient_gain=self.salient_gain,     # [K]
            group_ids=self.salient_group_ids,
            base_max_group=self.base_max_group,
            q_max=self.q_max,
        )  # [out, K], original coordinate space

        Y = Y + F.linear(X_S, delta_S)  # inject deployment-quantizer residual
        return Y


# ============================================================================
# SQAT Handler (integrates into training loop)
# ============================================================================

class SelectiveSalientQAT(QATHandler):
    """
    Selective Salient QAT handler.

    Lifecycle:
      prepare_model()  -> calibrate, select salient channels, patch layers
      on_train_begin() -> (nothing extra)
      on_step_end()    -> (nothing extra)
      on_train_end()   -> (nothing extra -- no sync needed)
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
        """
        Prepare the model for SQAT training.

        Args:
            salient_gain_map: optional dict mapping layer name -> [K] gain tensor D.
                              When provided, the corresponding layer's salient_gain
                              buffer is initialised with those values.
                              When None (default), all gains are set to 1.0 (no amplification).
                              Gains can also be changed at any time later via
                              layer.set_salient_gain(gain).
        """
        sqat_cfg = cfg["qat"]["sqat"]
        target_modules = cfg["lora"]["target_modules"]
        q_bits = cfg["model"]["quant_bits"]
        group_size = cfg["qat"].get("group_size", 128)
        top_k_ratio = sqat_cfg["top_k_ratio"]
        device = next(model.parameters()).device

        lora_cfg = cfg["lora"]
        lora_scaling = lora_cfg["alpha"] / lora_cfg["rank"]

        # Periodic base_max_group refresh interval (0 = disabled).
        # Tracks LoRA-induced weight drift so the training quantizer
        # stays consistent with what PTQ will see at export.
        self._refresh_interval = sqat_cfg.get("base_max_refresh_interval", 0)
        self._group_size = group_size
        if self._refresh_interval > 0:
            print(f"[SQAT] base_max_group refresh every {self._refresh_interval} steps")

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

        # --- Step 2c: Identify modules to patch (needed for both gain computation and patching) ---
        modules_to_patch = {}
        for name, module in model.named_modules():
            if name in salient_map:
                if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
                    modules_to_patch[name] = module

        # --- Step 2b: Auto-build salient_gain_map (AWQ-faithful per-channel search) ---
        #
        # AWQ paper computes the optimal per-channel scale as:
        #     D[j] = (s_x[j] / s_w[j])^alpha
        # where s_x is the activation magnitude and s_w is the weight magnitude.
        # This balances the trade-off: amplifying a channel protects its
        # activation-side contribution but inflates the quantization scale for
        # all channels in the same group.
        #
        # Key differences from our previous (broken) implementation:
        #   1. Use weight magnitude s_w as the denominator — channels with
        #      already-large weights need less amplification.
        #   2. Normalize per-group so that max(D) within each quant group
        #      doesn't exceed a safe ceiling (default 2.0).  AWQ's empirical
        #      sweet spot is D in [1, ~2]; values above ~5 destroy non-salient
        #      channels in the same group.
        #   3. All gains are >= 1.0 (never shrink).
        #
        # Set salient_gain_alpha=0.0 in config to disable (falls back to all-ones).
        gain_alpha = sqat_cfg.get("salient_gain_alpha", 0.0)
        gain_max_cap = sqat_cfg.get("salient_gain_max", 2.0)   # safety ceiling
        if salient_gain_map is None and gain_alpha > 0.0:
            salient_gain_map = {}
            for layer_name, sidx in salient_map.items():
                if layer_name not in second_moments:
                    continue
                sm = second_moments[layer_name].to(device)   # [in_features]

                # Dequantize full weight to get per-channel weight magnitudes
                module_for_layer = modules_to_patch.get(layer_name)
                if module_for_layer is None:
                    continue
                W_full = dequantize_layer(module_for_layer)  # [out, in]
                s_w = W_full.abs().mean(dim=0).to(device)    # [in_features]

                # AWQ formula: D[j] = (s_x[j] / s_w[j])^alpha
                s_x = sm.sqrt()                               # [in_features], activation RMS
                s_x_sal = s_x[sidx].clamp(min=1e-7)          # [K]
                s_w_sal = s_w[sidx].clamp(min=1e-7)          # [K]
                raw_D = (s_x_sal / s_w_sal).pow(gain_alpha)  # [K]

                # Normalize: min(D)=1 (only amplify), max(D) <= gain_max_cap
                raw_D = raw_D / raw_D.min().clamp(min=1e-7)  # shift so min=1
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
            base_max = compute_base_max_group(
                W_dequant, salient_idx, group_size
            ).to(device)

            # Optional per-layer salient_gain D; None => all-ones (no amplification)
            gain = None
            if salient_gain_map is not None and name in salient_gain_map:
                gain = salient_gain_map[name].to(device)

            sqat_linear = SelectiveSalientQATLinear(
                original_module=module,
                salient_indices=salient_idx,
                W_base_salient=W_base_salient,
                base_max_group=base_max,
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

        print(f"[SQAT] Patched {len(self.patched_layers)} layers with SQAT wrappers.")
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        # Periodically refresh base_max_group to track LoRA-induced weight drift.
        # Cost: one NF4 dequant + group-max per layer every N steps (negligible).
        if not hasattr(self, '_refresh_interval'):
            return
        if self._refresh_interval <= 0 or step % self._refresh_interval != 0:
            return
        self._refresh_base_max(model, step)

    @torch.no_grad()
    def _refresh_base_max(self, model, step):
        """Recompute base_max_group from current W_base + LoRA delta.

        IMPORTANT: Only base_max_group is updated.  W_base_salient must
        remain the frozen NF4 base slice — the forward path computes
        W_curr_S = W_base_salient + scaling * B @ A[:, S], so overwriting
        W_base_salient with the full current weight would double-count
        the LoRA contribution.
        """
        refreshed = 0
        for name, sqat_layer in self.patched_layers.items():
            if not sqat_layer.has_lora:
                continue
            # Current full effective weight = frozen_base + LoRA delta
            W_base_full = dequantize_layer(sqat_layer.original_module)  # [out, in]
            adapter_name = list(sqat_layer.original_module.lora_A.keys())[0]
            A_full = sqat_layer.original_module.lora_A[adapter_name].weight.data.float()
            B_full = sqat_layer.original_module.lora_B[adapter_name].weight.data.float()
            lora_delta = (B_full @ A_full) * sqat_layer.lora_scaling
            W_curr = W_base_full + lora_delta

            new_base_max = compute_base_max_group(
                W_curr,
                sqat_layer.salient_indices,
                self._group_size,
            ).to(sqat_layer.base_max_group.device)

            sqat_layer.base_max_group.copy_(new_base_max)
            refreshed += 1

        if refreshed > 0 and step > 0:
            print(f"[SQAT] Step {step}: refreshed base_max_group for {refreshed} layers")

    def on_train_end(self, model):
        if hasattr(self, '_refresh_interval') and self._refresh_interval > 0:
            self._refresh_base_max(model, step=-1)