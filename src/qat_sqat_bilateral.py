"""
Bilateral Selective Salient QAT with fixed deployment grids.

This implementation intentionally does NOT use SQAT's dynamic-anchor path.
For every patched LoRA linear we precompute the same group-wise quantizer
parameters used by the deployment grid:

  symmetric:   scale[out, group]
  asymmetric:  scale[out, group] and zero_point[out, group]

The third forward path injects fakequant residuals only on a Fisher-selected
cross-shaped mask:

  output branch owns      S_out x [d_in]
  input branch owns       ([d_out] \ S_out) x S_in

That keeps the input/output intersection from being quantized twice while still
storing the two salient weight tensors separately.
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .qat_base import (
    QATHandler,
    asymmetric_fakequant,
    asymmetric_scale_zero_from_pos_neg,
    round_ste,
)
from .qat_sqat import dequantize_layer


# ============================================================================
# Calibration diagnostics
# ============================================================================

def analyze_second_moment_outliers_topk(
    second_moments: Dict[str, torch.Tensor],
    selected_map: Dict[str, torch.Tensor],
    log_outlier_sigma: float = 3.0,
    eps: float = 1e-12,
) -> dict:
    """
    Analyze whether selected top-k channels cover robust second-moment outliers.

    Outliers are detected with the same robust log10 z-score idea used by
    SQAT's activation analysis, but this function accepts an explicit selected
    channel map so bilateral input/output sides can use independent top-k.
    """
    per_layer = {}
    global_num_channels = 0
    global_num_outliers = 0
    global_num_outlier_hits = 0
    global_total_mass = 0.0
    global_selected_mass = 0.0
    global_outlier_mass = 0.0
    global_hit_outlier_mass = 0.0

    for name, sm in second_moments.items():
        if name not in selected_map:
            continue

        sm = sm.detach().float().flatten().cpu()
        if sm.numel() == 0:
            continue

        n = sm.numel()
        selected = selected_map[name].detach().long().flatten().cpu()
        selected = selected[(selected >= 0) & (selected < n)].unique(sorted=True)
        if selected.numel() == 0:
            continue

        selected_mask = torch.zeros(n, dtype=torch.bool)
        selected_mask[selected] = True

        log_sm = torch.log10(sm.clamp(min=eps))
        med = log_sm.median()
        mad = (log_sm - med).abs().median().clamp(min=eps)
        robust_std = (1.4826 * mad).clamp(min=eps)
        outlier_mask = ((log_sm - med) / robust_std) >= log_outlier_sigma
        if not outlier_mask.any():
            thr = torch.quantile(log_sm, 0.99)
            outlier_mask = log_sm >= thr

        hit_mask = selected_mask & outlier_mask
        total_mass = sm.sum().clamp(min=eps)
        outlier_mass = sm[outlier_mask].sum().clamp(min=eps)
        hit_outlier_mass = sm[hit_mask].sum()
        sm_median = sm.median().clamp(min=eps)

        per_layer[name] = {
            "num_channels": int(n),
            "topk": int(selected.numel()),
            "topk_ratio": float(selected.numel() / max(n, 1)),
            "num_outliers": int(outlier_mask.sum().item()),
            "num_outlier_hits": int(hit_mask.sum().item()),
            "mass_capture": float((sm[selected_mask].sum() / total_mass).item()),
            "outlier_recall": float(
                (hit_mask.sum().float() / outlier_mask.sum().float()).item()
            ),
            "outlier_mass_recall": float((hit_outlier_mass / outlier_mass).item()),
            "max_over_median": float((sm.max() / sm_median).item()),
            "selected_indices_sorted": selected.tolist(),
            "outlier_indices": outlier_mask.nonzero(as_tuple=False).flatten().tolist(),
            "hit_outlier_indices": hit_mask.nonzero(as_tuple=False).flatten().tolist(),
            "sorted_curve": torch.sort(sm / sm_median, descending=True).values,
        }

        global_num_channels += n
        global_num_outliers += int(outlier_mask.sum().item())
        global_num_outlier_hits += int(hit_mask.sum().item())
        global_total_mass += float(total_mass.item())
        global_selected_mass += float(sm[selected_mask].sum().item())
        global_outlier_mass += float(outlier_mass.item())
        global_hit_outlier_mass += float(hit_outlier_mass.item())

    num_layers = max(len(per_layer), 1)
    return {
        "per_layer": per_layer,
        "global": {
            "num_layers": len(per_layer),
            "num_channels": global_num_channels,
            "num_outliers": global_num_outliers,
            "mean_mass_capture": float(
                sum(v["mass_capture"] for v in per_layer.values()) / num_layers
            ),
            "mean_outlier_recall": float(
                sum(v["outlier_recall"] for v in per_layer.values()) / num_layers
            ),
            "mean_outlier_mass_recall": float(
                sum(v["outlier_mass_recall"] for v in per_layer.values()) / num_layers
            ),
            "global_mass_capture": float(global_selected_mass / max(global_total_mass, eps)),
            "global_outlier_recall": float(
                global_num_outlier_hits / max(global_num_outliers, 1)
            ),
            "global_outlier_mass_recall": float(
                global_hit_outlier_mass / max(global_outlier_mass, eps)
            ),
        },
    }


def _resample_curve(curve: torch.Tensor, num_points: int = 256) -> torch.Tensor:
    if curve.numel() == num_points:
        return curve
    idx = torch.linspace(0, curve.numel() - 1, steps=num_points)
    lo = idx.floor().long()
    hi = idx.ceil().long()
    w = idx - lo.float()
    return curve[lo] * (1.0 - w) + curve[hi] * w


def plot_bilateral_second_moment_statistics(
    input_analysis: dict,
    output_analysis: dict,
    save_path: str,
    num_points: int = 256,
    max_layers_to_draw: int = 64,
):
    """Plot input activation and output-gradient second-moment concentration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    sides = [
        ("Input E[x^2]", input_analysis),
        ("Output E[g^2]", output_analysis),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = torch.linspace(0, 100, steps=num_points).numpy()

    for row, (title, analysis) in enumerate(sides):
        per_layer = analysis["per_layer"]
        global_stats = analysis["global"]
        ax_curve = axes[row][0]
        ax_bar = axes[row][1]

        if not per_layer:
            ax_curve.set_title(f"{title}: no layers")
            ax_bar.set_title(f"{title}: no layers")
            continue

        curves = torch.stack(
            [
                _resample_curve(stats["sorted_curve"], num_points)
                for stats in per_layer.values()
            ],
            dim=0,
        )
        mean_curve = curves.mean(dim=0)
        q25_curve = torch.quantile(curves, 0.25, dim=0)
        q75_curve = torch.quantile(curves, 0.75, dim=0)

        for i in range(min(curves.shape[0], max_layers_to_draw)):
            ax_curve.plot(x, curves[i].numpy(), linewidth=0.8, alpha=0.15)
        ax_curve.plot(x, mean_curve.numpy(), linewidth=2.5, label="mean")
        ax_curve.fill_between(
            x, q25_curve.numpy(), q75_curve.numpy(), alpha=0.2, label="p25-p75"
        )
        ax_curve.set_yscale("log")
        ax_curve.set_xlabel("Channel rank percentile (descending score)")
        ax_curve.set_ylabel("Score / layer median")
        ax_curve.set_title(f"{title} concentration")
        txt = (
            f"global mass capture = {global_stats['global_mass_capture'] * 100:.1f}%\n"
            f"global outlier recall = {global_stats['global_outlier_recall'] * 100:.1f}%\n"
            f"global outlier mass recall = {global_stats['global_outlier_mass_recall'] * 100:.1f}%"
        )
        ax_curve.text(
            0.98, 0.02, txt,
            transform=ax_curve.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax_curve.legend()

        ranked = sorted(
            per_layer.items(),
            key=lambda kv: kv[1]["outlier_mass_recall"],
            reverse=True,
        )
        show_items = ranked[:min(20, len(ranked))]
        labels = [
            name if len(name) <= 28 else "..." + name[-25:]
            for name, _ in show_items
        ]
        outlier_mass_recalls = [
            stats["outlier_mass_recall"] * 100.0 for _, stats in show_items
        ]
        mass_captures = [stats["mass_capture"] * 100.0 for _, stats in show_items]
        xpos = list(range(len(show_items)))

        ax_bar.bar(xpos, outlier_mass_recalls, alpha=0.75, label="outlier mass recall (%)")
        ax_bar.plot(xpos, mass_captures, marker="o", linewidth=1.5, label="top-k mass capture (%)")
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax_bar.set_ylabel("Coverage (%)")
        ax_bar.set_title(f"{title} top-k coverage")
        ax_bar.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[BiSQAT] Saved Fisher second-moment statistics plot to: {save_path}")


def _selected_report_for_layer(scores: torch.Tensor, selected: torch.Tensor) -> dict:
    scores = scores.detach().float().flatten().cpu()
    selected = selected.detach().long().flatten().cpu()
    selected = selected[(selected >= 0) & (selected < scores.numel())].unique(sorted=True)
    selected_scores = scores[selected] if selected.numel() else torch.empty(0)
    order = torch.argsort(selected_scores, descending=True)
    by_score = selected[order]
    scores_by_score = selected_scores[order]
    return {
        "selected_indices_sorted": selected.tolist(),
        "selected_indices_by_score": by_score.tolist(),
        "selected_scores_by_score": [float(v) for v in scores_by_score.tolist()],
    }


def _compare_output_stability(current: dict, previous: dict) -> dict:
    per_layer = {}
    overlaps = []
    previous_layers = previous.get("layers", {})
    for name, item in current.get("layers", {}).items():
        cur = set(item.get("output", []))
        prev = set(previous_layers.get(name, {}).get("output", []))
        if not cur and not prev:
            continue
        union = cur | prev
        jaccard = float(len(cur & prev) / max(len(union), 1))
        per_layer[name] = {
            "jaccard": jaccard,
            "intersection": sorted(cur & prev),
            "current_only": sorted(cur - prev),
            "previous_only": sorted(prev - cur),
        }
        overlaps.append(jaccard)
    return {
        "previous_run_id": previous.get("run_id"),
        "mean_output_jaccard": float(sum(overlaps) / max(len(overlaps), 1)),
        "per_layer": per_layer,
    }


def save_bilateral_salient_channel_report(
    input_scores: Dict[str, torch.Tensor],
    output_scores: Dict[str, torch.Tensor],
    input_map: Dict[str, torch.Tensor],
    output_map: Dict[str, torch.Tensor],
    input_analysis: dict,
    output_analysis: dict,
    save_path: str,
    history_path: str,
    input_top_k: int,
    output_top_k: int,
) -> None:
    """Save exact selected channels so calibration-set stability can be compared."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if history_path:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    layers = {}
    all_names = sorted(set(input_map.keys()) | set(output_map.keys()))
    for name in all_names:
        item = {}
        if name in input_map and name in input_scores:
            item["input"] = _selected_report_for_layer(input_scores[name], input_map[name])
            item["input"].update(
                {
                    k: v
                    for k, v in input_analysis["per_layer"].get(name, {}).items()
                    if k in (
                        "num_channels",
                        "topk",
                        "topk_ratio",
                        "num_outliers",
                        "num_outlier_hits",
                        "mass_capture",
                        "outlier_recall",
                        "outlier_mass_recall",
                        "outlier_indices",
                        "hit_outlier_indices",
                    )
                }
            )
        if name in output_map and name in output_scores:
            item["output"] = _selected_report_for_layer(output_scores[name], output_map[name])
            item["output"].update(
                {
                    k: v
                    for k, v in output_analysis["per_layer"].get(name, {}).items()
                    if k in (
                        "num_channels",
                        "topk",
                        "topk_ratio",
                        "num_outliers",
                        "num_outlier_hits",
                        "mass_capture",
                        "outlier_recall",
                        "outlier_mass_recall",
                        "outlier_indices",
                        "hit_outlier_indices",
                    )
                }
            )
        layers[name] = item

    payload = {
        "run_id": run_id,
        "input_top_k": int(input_top_k),
        "output_top_k": int(output_top_k),
        "input_global": input_analysis["global"],
        "output_global": output_analysis["global"],
        "layers": layers,
    }

    compact = {
        "run_id": run_id,
        "input_top_k": int(input_top_k),
        "output_top_k": int(output_top_k),
        "layers": {
            name: {
                "input": layers[name].get("input", {}).get("selected_indices_by_score", []),
                "output": layers[name].get("output", {}).get("selected_indices_by_score", []),
            }
            for name in layers
        },
    }
    if history_path and os.path.exists(history_path) and output_top_k > 0:
        with open(history_path) as f:
            previous_lines = [line.strip() for line in f if line.strip()]
        previous = None
        for line in reversed(previous_lines):
            candidate = json.loads(line)
            if int(candidate.get("output_top_k", -1)) == int(output_top_k):
                previous = candidate
                break
        if previous is not None:
            stability = _compare_output_stability(compact, previous)
            payload["previous_run_output_stability"] = stability
            compact["previous_run_output_stability"] = {
                "previous_run_id": stability["previous_run_id"],
                "mean_output_jaccard": stability["mean_output_jaccard"],
            }
            print(
                "[BiSQAT] Output salient overlap vs previous calibration "
                f"({stability['previous_run_id']}): "
                f"mean Jaccard={stability['mean_output_jaccard']:.4f}"
            )

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[BiSQAT] Saved selected channel report to: {save_path}")

    if history_path:
        with open(history_path, "a") as f:
            f.write(json.dumps(compact) + "\n")
        print(f"[BiSQAT] Appended selected channel history to: {history_path}")


# ============================================================================
# Fixed-grid fakequant primitives
# ============================================================================

def fixed_symmetric_fakequant(
    w: torch.Tensor,
    scale: torch.Tensor,
    q_max: float,
) -> torch.Tensor:
    """Symmetric fixed-scale fakequant with STE."""
    w_f = w.float()
    scale_f = scale.float().clamp(min=1e-7)
    q = round_ste(torch.clamp(w_f / scale_f, -q_max, q_max))
    return (q * scale_f).to(dtype=w.dtype)


def fixed_asymmetric_fakequant(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_max: int,
) -> torch.Tensor:
    """Affine fixed-(scale, zero_point) fakequant with STE."""
    return asymmetric_fakequant(
        w=w.float(),
        scale=scale.float().clamp(min=1e-7),
        zero_point=zero_point.float(),
        q_max=int(q_max),
    ).to(dtype=w.dtype)


def compute_fixed_symmetric_qparams(
    W: torch.Tensor,
    group_size: int,
    q_bits: int,
) -> torch.Tensor:
    """Return fixed per-row/per-input-group scales with shape [out, G]."""
    q_max = float(2 ** (q_bits - 1) - 1)
    out_f, in_f = W.shape
    num_groups = math.ceil(in_f / group_size)
    pad = num_groups * group_size - in_f
    Wp = F.pad(W.float(), (0, pad)) if pad > 0 else W.float()
    Wg = Wp.view(out_f, num_groups, group_size)
    return (Wg.abs().amax(dim=2) / q_max).clamp(min=1e-7)


def compute_fixed_asymmetric_qparams(
    W: torch.Tensor,
    group_size: int,
    q_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fixed affine qparams with shapes scale/zp = [out, G]."""
    q_max = int(2 ** q_bits - 1)
    out_f, in_f = W.shape
    num_groups = math.ceil(in_f / group_size)
    pad = num_groups * group_size - in_f
    Wp = F.pad(W.float(), (0, pad)) if pad > 0 else W.float()
    Wg = Wp.view(out_f, num_groups, group_size)
    pos = Wg.clamp(min=0).amax(dim=2)
    neg = (-Wg).clamp(min=0).amax(dim=2)
    return asymmetric_scale_zero_from_pos_neg(pos, neg, q_max)


def _expand_group_params(
    params: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Expand [out, G] qparams to the exact selected dense slice [rows, cols].

    This is the input-side pre-expansion: selected input columns cut across
    quantization groups, so each selected column needs the qparam of its
    original group.
    """
    group_ids = (col_indices // group_size).to(device=params.device, dtype=torch.long)
    rows = row_indices.to(device=params.device, dtype=torch.long)
    return params.index_select(0, rows).index_select(1, group_ids).contiguous()


def _expand_all_input_params(
    params: torch.Tensor,
    row_indices: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Expand [out, G] qparams to [selected_rows, in_features]."""
    col_indices = torch.arange(in_features, device=params.device)
    return _expand_group_params(params, row_indices, col_indices, group_size)


# ============================================================================
# Fisher calibration
# ============================================================================

def _tensor_channel_second_moment(t: torch.Tensor) -> torch.Tensor:
    """Average t^2 over every dimension except the channel-last dimension."""
    t = t.detach().float()
    reduce_dims = tuple(range(t.dim() - 1))
    return t.pow(2).mean(dim=reduce_dims)


def estimate_fisher_second_moments(
    model: nn.Module,
    dataloader: DataLoader,
    target_modules: list,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    One calibration pass for Fisher-factor statistics.

    Returns:
        input_second_moments:  layer -> E[x_j^2]      [in_features]
        output_grad_moments:  layer -> E[g_i^2]      [out_features]
    """
    input_acc: Dict[str, torch.Tensor] = {}
    input_counts: Dict[str, int] = {}
    grad_acc: Dict[str, torch.Tensor] = {}
    grad_counts: Dict[str, int] = {}
    hooks = []

    def make_grad_hook(name: str):
        def grad_hook(grad: torch.Tensor):
            g_sq = _tensor_channel_second_moment(grad)
            if name not in grad_acc:
                grad_acc[name] = torch.zeros_like(g_sq, device="cpu")
                grad_counts[name] = 0
            grad_acc[name].add_(g_sq.cpu())
            grad_counts[name] += 1
        return grad_hook

    def make_forward_hook(name: str):
        def forward_hook(module, inputs, output):
            x = inputs[0]
            x_sq = _tensor_channel_second_moment(x)
            if name not in input_acc:
                input_acc[name] = torch.zeros_like(x_sq, device="cpu")
                input_counts[name] = 0
            input_acc[name].add_(x_sq.cpu())
            input_counts[name] += 1

            if torch.is_tensor(output) and output.requires_grad:
                output.register_hook(make_grad_hook(name))
        return forward_hook

    for name, module in model.named_modules():
        terminal = name.rsplit(".", 1)[-1] if name else ""
        if terminal not in target_modules:
            continue
        if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
            hooks.append(module.register_forward_hook(make_forward_hook(name)))

    if not hooks:
        print("[BiSQAT] WARNING: no LoRA target modules found for calibration.")

    was_training = model.training
    model.eval()
    for batch in tqdm(dataloader, desc="[BiSQAT] Calibrating Fisher moments"):
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError(
                "Bilateral SQAT calibration requires labels so model(**batch) returns loss."
            )
        loss.backward()
        model.zero_grad(set_to_none=True)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    input_second = {
        name: acc / max(input_counts.get(name, 1), 1)
        for name, acc in input_acc.items()
    }
    output_second = {
        name: acc / max(grad_counts.get(name, 1), 1)
        for name, acc in grad_acc.items()
    }
    return input_second, output_second


def select_topk_channels(
    scores: Dict[str, torch.Tensor],
    top_k: int,
) -> Dict[str, torch.Tensor]:
    """Select top-k channels per layer by score."""
    result = {}
    for name, sm in scores.items():
        n = sm.numel()
        if top_k <= 0:
            result[name] = torch.empty(0, dtype=torch.long)
            continue
        k = min(max(int(top_k), 1), n)
        _, idx = sm.float().topk(k)
        result[name] = idx.sort().values
    return result


select_input_channels = select_topk_channels
select_output_channels = select_topk_channels


# ============================================================================
# Bilateral fixed-grid SQAT Linear
# ============================================================================

class BilateralSalientQATLinear(nn.Module):
    """
    QLoRA wrapper with a fixed-grid bilateral salient fakequant residual path.

    Stored salient tensors:
      W_base_out: output-owned rows, shape [K_out, in]
      W_base_in:  input-owned columns, shape [out, K_in]

    The output-owned rows inside W_base_in are zeroed and masked, so the
    input-side branch can add a full-size Y_in directly without sending any
    residual or gradient through the input/output intersection.
    """

    def __init__(
        self,
        original_module: nn.Module,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        input_row_mask: torch.Tensor,
        W_base_in: torch.Tensor,
        W_base_out: torch.Tensor,
        scales_in: torch.Tensor,
        scales_out: torch.Tensor,
        fixed_scales: torch.Tensor,
        q_bits: int,
        lora_scaling: float,
    ):
        super().__init__()
        self.original_module = original_module
        self.lora_scaling = float(lora_scaling)
        self.q_max = float(2 ** (q_bits - 1) - 1)
        self.symmetric = True
        self.has_lora = hasattr(original_module, "lora_A")

        self.register_buffer("input_indices", input_indices)
        self.register_buffer("output_indices", output_indices)
        self.register_buffer("input_row_mask", input_row_mask)
        self.register_buffer("W_base_in", W_base_in)
        self.register_buffer("W_base_out", W_base_out)
        self.register_buffer("scales_in", scales_in)
        self.register_buffer("scales_out", scales_out)
        self.register_buffer("fixed_scales", fixed_scales)

    @property
    def salient_indices(self):
        """Compatibility alias for older SQAT metadata consumers."""
        return self.input_indices

    def _adapter_name(self) -> str:
        return list(self.original_module.lora_A.keys())[0]

    def _lora_A(self) -> torch.Tensor:
        return self.original_module.lora_A[self._adapter_name()].weight

    def _lora_B(self) -> torch.Tensor:
        return self.original_module.lora_B[self._adapter_name()].weight

    def _current_input_weight(self) -> torch.Tensor:
        A_in = self._lora_A()[:, self.input_indices]
        W_in = self.W_base_in + (self._lora_B() @ A_in) * self.lora_scaling
        return W_in * self.input_row_mask.to(dtype=W_in.dtype)

    def _current_output_weight(self) -> torch.Tensor:
        B_out = self._lora_B()[self.output_indices, :]
        return self.W_base_out + (B_out @ self._lora_A()) * self.lora_scaling

    def _fakequant_in(self, W_curr: torch.Tensor) -> torch.Tensor:
        return fixed_symmetric_fakequant(W_curr, self.scales_in, self.q_max)

    def _fakequant_out(self, W_curr: torch.Tensor) -> torch.Tensor:
        return fixed_symmetric_fakequant(W_curr, self.scales_out, self.q_max)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        Y = self.original_module(x, *args, **kwargs)

        # if self.input_indices.numel() > 0:
        X_in = x[..., self.input_indices]
        # W_in = self._current_input_weight()
        # delta_in = (self._fakequant_in(W_in) - W_in).to(dtype=x.dtype)
        Y = Y + F.linear(X_in, self.W_base_in).to(dtype=Y.dtype)

        #if self.output_indices.numel() > 0:
        # W_out = self._current_output_weight()
        # delta_out = (self._fakequant_out(W_out) - W_out).to(dtype=x.dtype)
        # Y_out = F.linear(x, delta_out).to(dtype=Y.dtype)
        # Y = Y.index_add(-1, self.output_indices, Y_out)

        return Y


class BilateralSalientQATLinearAsymmetric(BilateralSalientQATLinear):
    """Affine asymmetric fixed-grid bilateral SQAT wrapper."""

    def __init__(
        self,
        original_module: nn.Module,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        input_row_mask: torch.Tensor,
        W_base_in: torch.Tensor,
        W_base_out: torch.Tensor,
        scales_in: torch.Tensor,
        scales_out: torch.Tensor,
        zero_points_in: torch.Tensor,
        zero_points_out: torch.Tensor,
        fixed_scales: torch.Tensor,
        fixed_zero_points: torch.Tensor,
        q_bits: int,
        lora_scaling: float,
    ):
        super().__init__(
            original_module=original_module,
            input_indices=input_indices,
            output_indices=output_indices,
            input_row_mask=input_row_mask,
            W_base_in=W_base_in,
            W_base_out=W_base_out,
            scales_in=scales_in,
            scales_out=scales_out,
            fixed_scales=fixed_scales,
            q_bits=q_bits,
            lora_scaling=lora_scaling,
        )
        self.q_max = int(2 ** q_bits - 1)
        self.symmetric = False
        self.register_buffer("zero_points_in", zero_points_in)
        self.register_buffer("zero_points_out", zero_points_out)
        self.register_buffer("fixed_zero_points", fixed_zero_points)

    def _fakequant_in(self, W_curr: torch.Tensor) -> torch.Tensor:
        return fixed_asymmetric_fakequant(
            W_curr, self.scales_in, self.zero_points_in, self.q_max
        )

    def _fakequant_out(self, W_curr: torch.Tensor) -> torch.Tensor:
        return fixed_asymmetric_fakequant(
            W_curr, self.scales_out, self.zero_points_out, self.q_max
        )


# ============================================================================
# Handler
# ============================================================================

class BilateralSalientQAT(QATHandler):
    """Fisher-aware bilateral SQAT with fixed quantizer parameters."""

    def __init__(self):
        self.patched_layers: Dict[str, BilateralSalientQATLinear] = {}

    def prepare_model(
        self,
        model: nn.Module,
        cfg: dict,
        tokenizer=None,
        calibration_dataloader=None,
        **kwargs,
    ) -> nn.Module:
        sqat_cfg = cfg["qat"]["sqat"]
        target_modules = cfg["lora"]["target_modules"]
        q_bits = cfg["model"]["quant_bits"]
        group_size = cfg["qat"].get("group_size", 128)
        symmetric = cfg["qat"].get("symmetric", True)
        input_top_k = int(sqat_cfg.get("input_top_k", 64))
        output_top_k = int(sqat_cfg.get("output_top_k", 64))
        device = next(model.parameters()).device

        if input_top_k <= 0 and output_top_k <= 0:
            print("[BiSQAT] input_top_k=0 and output_top_k=0; bilateral QAT disabled.")
            return model

        lora_cfg = cfg["lora"]
        lora_scaling = lora_cfg["alpha"] / lora_cfg["rank"]

        assert calibration_dataloader is not None, (
            "Bilateral SQAT requires a calibration_dataloader. "
            "Pass it via prepare_model(calibration_dataloader=...)."
        )

        input_moments, grad_moments = estimate_fisher_second_moments(
            model=model,
            dataloader=calibration_dataloader,
            target_modules=target_modules,
            device=str(device),
        )
        input_map = select_input_channels(input_moments, input_top_k)
        output_map = select_output_channels(grad_moments, output_top_k)

        print(
            f"[BiSQAT] Selected input top-{input_top_k} channels "
            f"and output top-{output_top_k} channels"
        )
        for lname in list(output_map.keys())[:3]:
            output_report = _selected_report_for_layer(grad_moments[lname], output_map[lname])
            print(
                f"  [BiSQAT] output salient {lname}: "
                f"{output_report['selected_indices_by_score']}"
            )

        input_analysis = analyze_second_moment_outliers_topk(
            input_moments,
            input_map,
            log_outlier_sigma=sqat_cfg.get("outlier_log_sigma", 3.0),
        )
        output_analysis = analyze_second_moment_outliers_topk(
            grad_moments,
            output_map,
            log_outlier_sigma=sqat_cfg.get("outlier_log_sigma", 3.0),
        )
        self.input_analysis = input_analysis
        self.output_analysis = output_analysis

        if sqat_cfg.get("plot_activation_stats", True):
            plot_bilateral_second_moment_statistics(
                input_analysis=input_analysis,
                output_analysis=output_analysis,
                save_path=sqat_cfg.get(
                    "bilateral_stats_plot_path",
                    "debug/bisqat_fisher_second_moment.png",
                ),
                num_points=sqat_cfg.get("activation_plot_points", 256),
                max_layers_to_draw=sqat_cfg.get("activation_plot_max_layers", 64),
            )

        save_bilateral_salient_channel_report(
            input_scores=input_moments,
            output_scores=grad_moments,
            input_map=input_map,
            output_map=output_map,
            input_analysis=input_analysis,
            output_analysis=output_analysis,
            save_path=sqat_cfg.get(
                "bilateral_salient_channels_path",
                "debug/bisqat_salient_channels.json",
            ),
            history_path=sqat_cfg.get(
                "bilateral_salient_channels_history_path",
                "debug/bisqat_salient_channels_history.jsonl",
            ),
            input_top_k=input_top_k,
            output_top_k=output_top_k,
        )

        modules_to_patch = {}
        for name, module in model.named_modules():
            if name in input_map and name in output_map:
                if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
                    modules_to_patch[name] = module

        for name, module in modules_to_patch.items():
            W_dequant = dequantize_layer(module).to(device)
            out_f, in_f = W_dequant.shape

            input_idx = input_map[name].to(device=device, dtype=torch.long)
            output_idx = output_map[name].to(device=device, dtype=torch.long)

            if input_idx.numel() > 0:
                assert int(input_idx.max().item()) < in_f, (
                    f"[BiSQAT] Layer {name}: input salient index out of range."
                )
            if output_idx.numel() > 0:
                assert int(output_idx.max().item()) < out_f, (
                    f"[BiSQAT] Layer {name}: output salient index out of range."
                )

            input_row_mask = torch.ones(out_f, 1, device=device, dtype=W_dequant.dtype)
            if output_idx.numel() > 0:
                input_row_mask.index_fill_(0, output_idx, 0.0)

            W_base_in = W_dequant.index_select(1, input_idx).contiguous()
            W_base_in = W_base_in * input_row_mask
            W_base_out = W_dequant.index_select(0, output_idx).contiguous()
            all_rows = torch.arange(out_f, device=device, dtype=torch.long)

            if symmetric:
                fixed_scales = compute_fixed_symmetric_qparams(
                    W_dequant, group_size=group_size, q_bits=q_bits
                ).to(device)
                scales_in = _expand_group_params(
                    fixed_scales, all_rows, input_idx, group_size
                )
                scales_out = _expand_all_input_params(
                    fixed_scales, output_idx, in_f, group_size
                )
                wrapper = BilateralSalientQATLinear(
                    original_module=module,
                    input_indices=input_idx,
                    output_indices=output_idx,
                    input_row_mask=input_row_mask,
                    W_base_in=W_base_in,
                    W_base_out=W_base_out,
                    scales_in=scales_in,
                    scales_out=scales_out,
                    fixed_scales=fixed_scales,
                    q_bits=q_bits,
                    lora_scaling=lora_scaling,
                ).to(device)
            else:
                fixed_scales, fixed_zps = compute_fixed_asymmetric_qparams(
                    W_dequant, group_size=group_size, q_bits=q_bits
                )
                fixed_scales = fixed_scales.to(device)
                fixed_zps = fixed_zps.to(device)
                scales_in = _expand_group_params(
                    fixed_scales, all_rows, input_idx, group_size
                )
                zps_in = _expand_group_params(
                    fixed_zps, all_rows, input_idx, group_size
                )
                scales_out = _expand_all_input_params(
                    fixed_scales, output_idx, in_f, group_size
                )
                zps_out = _expand_all_input_params(
                    fixed_zps, output_idx, in_f, group_size
                )
                wrapper = BilateralSalientQATLinearAsymmetric(
                    original_module=module,
                    input_indices=input_idx,
                    output_indices=output_idx,
                    input_row_mask=input_row_mask,
                    W_base_in=W_base_in,
                    W_base_out=W_base_out,
                    scales_in=scales_in,
                    scales_out=scales_out,
                    zero_points_in=zps_in,
                    zero_points_out=zps_out,
                    fixed_scales=fixed_scales,
                    fixed_zero_points=fixed_zps,
                    q_bits=q_bits,
                    lora_scaling=lora_scaling,
                ).to(device)

            self.patched_layers[name] = wrapper
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], wrapper)
            else:
                setattr(model, name, wrapper)

        print(
            f"[BiSQAT] Patched {len(self.patched_layers)} layers "
            f"(group_size={group_size}, bits={q_bits}, symmetric={symmetric})."
        )
        for lname, layer in list(self.patched_layers.items())[:3]:
            print(
                f"  [BiSQAT] {lname}: "
                f"K_in={layer.input_indices.numel()}, "
                f"K_out={layer.output_indices.numel()}, "
                f"W_in={tuple(layer.W_base_in.shape)}, "
                f"W_out={tuple(layer.W_base_out.shape)}"
            )
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        pass

    def on_train_end(self, model):
        pass
