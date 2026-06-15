#!/usr/bin/env python3
"""
Analyze residual-stream salient channels for SQAT segment permutation.

This script intentionally does not use a manually chosen per-layer top-k ratio.
It derives salient/outlier channels from each source's measured E[x^2]
distribution, then chooses contiguous segments automatically with a dynamic
program. Each segment receives a group_k snapped to one of {64, 128, 256}
by default.

Outputs:
    salient_analysis_out/
        fig1_layer_outlier_counts.png
        fig2_layer_outlier_jaccard.png
        fig3_source_capture_heatmap.png
        fig4_segment_groupk.png
        fig5_segment_cost_curve.png
        summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SourceKey = Tuple[int, str]
SOURCE_NAMES = ("attn", "mlp")
GROUP_K_DEFAULTS = (64, 128, 256)
TARGET_CAPTURE = 1.0
MIN_SEGMENT_LEN = 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect true E[x^2] outlier channels, automatically choose SQAT "
            "segments, and evaluate per-source capture."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument("--n_samples", type=int, default=512, help="Calibration samples.")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "metamath", "math", "commonsense"],
        help="Calibration dataset. 'math' is an alias for 'metamath'.",
    )
    parser.add_argument(
        "--outlier_log_sigma",
        type=float,
        default=3.0,
        help="Per-source log(E[x^2]) z-score threshold.",
    )
    parser.add_argument(
        "--group_k_candidates",
        type=int,
        nargs="+",
        default=list(GROUP_K_DEFAULTS),
        help="Allowed per-segment group_k values. The required count is ceiled to these buckets.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=8,
        help=(
            "Maximum allowed number of segments. The script chooses the lowest "
            "total segment_layers*group_k cost under this limit while capturing all true outliers."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./salient_analysis_out",
        help="Directory for figures and summary.json.",
    )
    parser.add_argument(
        "--save_second_moments",
        type=str,
        default=None,
        help="Optional path to save collected second moments as a .pt file.",
    )
    parser.add_argument(
        "--load_second_moments",
        type=str,
        default=None,
        help="Optional .pt file with second moments to skip model calibration.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to the tokenizer/model loaders (some Qwen variants).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    candidates = sorted(set(args.group_k_candidates))
    assert candidates, "--group_k_candidates must not be empty"
    assert all(k > 0 for k in candidates), "--group_k_candidates must be positive"
    assert args.max_segments > 0, "--max_segments must be positive"
    args.group_k_candidates = candidates


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


_DATASET_CONFIGS = {
    "wikitext": (
        "wikitext",
        "wikitext-2-raw-v1",
        "train",
        lambda item: item["text"],
    ),
    "metamath": (
        "meta-math/MetaMathQA",
        None,
        "train",
        lambda item: item["query"] + " " + item["response"],
    ),
    "math": (
        "meta-math/MetaMathQA",
        None,
        "train",
        lambda item: item["query"] + " " + item["response"],
    ),
    "commonsense": (
        "Ctau/commonsense_qa",
        None,
        "train",
        lambda item: item["question"] + " " + " ".join(item["choices"]["text"]),
    ),
}


def load_calibration_data(
    tokenizer: AutoTokenizer,
    n_samples: int,
    seq_len: int,
    dataset_name: str,
) -> List[Dict[str, torch.Tensor]]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_path, hf_name, split, text_fn = _DATASET_CONFIGS[dataset_name]
    print(f"Loading calibration dataset: {dataset_name} ({hf_path})")
    dataset = (
        load_dataset(hf_path, hf_name, split=split)
        if hf_name
        else load_dataset(hf_path, split=split)
    )

    samples: List[Dict[str, torch.Tensor]] = []
    for item in dataset:
        text = text_fn(item).strip()
        if not text:
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=False,
        )
        if enc["input_ids"].shape[1] < 2:
            continue
        samples.append(enc)
        if len(samples) >= n_samples:
            break

    print(f"  Loaded {len(samples)} samples (max_seq_len={seq_len}).")
    return samples


# -----------------------------------------------------------------------------
# Model hooks and second moments
# -----------------------------------------------------------------------------


def _resolve_decoder_layers(model) -> Sequence[torch.nn.Module]:
    obj = model
    for _ in range(5):
        if hasattr(obj, "layers"):
            return obj.layers
        if hasattr(obj, "model"):
            obj = obj.model
            continue
        break
    raise AttributeError(f"Could not find decoder .layers on {type(model).__name__}")


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.no_grad()
def estimate_second_moments(
    model,
    calibration_data: List[Dict[str, torch.Tensor]],
    hidden_size: int,
    num_layers: int,
) -> Dict[SourceKey, torch.Tensor]:
    """
    Estimate per-channel E[x^2] for residual-stream sources:
      (l, 'attn') = q_proj input after input_layernorm
      (l, 'mlp')  = gate_proj input after post_attention_layernorm
    """
    sum_sq: Dict[SourceKey, torch.Tensor] = {}
    tok_count: Dict[SourceKey, int] = {}
    for layer_idx in range(num_layers):
        for src in SOURCE_NAMES:
            sum_sq[(layer_idx, src)] = torch.zeros(hidden_size, dtype=torch.float32)
            tok_count[(layer_idx, src)] = 0

    handles = []

    def make_hook(key: SourceKey):
        def hook(_module, inp, _out):
            x = inp[0].detach().reshape(-1, hidden_size).float().cpu()
            sum_sq[key].add_(x.square().sum(dim=0))
            tok_count[key] += x.shape[0]

        return hook

    layers = _resolve_decoder_layers(model)
    for layer_idx in range(num_layers):
        handles.append(
            layers[layer_idx].self_attn.q_proj.register_forward_hook(
                make_hook((layer_idx, "attn"))
            )
        )
        handles.append(
            layers[layer_idx].mlp.gate_proj.register_forward_hook(
                make_hook((layer_idx, "mlp"))
            )
        )

    model.eval()
    try:
        for batch in tqdm(calibration_data, desc="Calibrating E[x^2]"):
            input_ids = batch["input_ids"].to(_input_device(model))
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    return {key: sum_sq[key] / max(tok_count[key], 1) for key in sum_sq}


# -----------------------------------------------------------------------------
# Outliers and segments
# -----------------------------------------------------------------------------


@dataclass
class SegmentCandidate:
    start: int
    end: int
    group_k: int
    bucket: List[int]
    bucket_ranked: List[int]
    outlier_union_count: int
    source_outlier_total: int
    source_outlier_captured: int
    source_capture_ratio: float
    satisfies_target: bool
    energy_coverage: float
    avg_pairwise_jaccard: float
    group_work: float
    selection_cost: float

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def layer_range(self) -> List[int]:
        return [self.start, self.end - 1]


def _safe_jaccard(set_a: Set[int], set_b: Set[int], empty_value: float = 1.0) -> float:
    union = len(set_a | set_b)
    if union == 0:
        return empty_value
    return len(set_a & set_b) / union


def _detect_source_outliers(
    values: torch.Tensor,
    sigma: float,
    eps: float = 1e-30,
) -> Tuple[torch.Tensor, float, float, float]:
    log_v = torch.log(values.float().clamp(min=eps))
    mean = log_v.mean().item()
    std = log_v.std(unbiased=False).item()
    threshold = mean + sigma * std
    idx = torch.where(log_v > threshold)[0].to(torch.long)
    return idx, threshold, mean, std


def analyze_outlier_distribution(
    second_moments: Dict[SourceKey, torch.Tensor],
    num_layers: int,
    sigma: float,
) -> Dict:
    normalized: Dict[SourceKey, torch.Tensor] = {}
    source_outliers: Dict[SourceKey, torch.Tensor] = {}
    source_stats: Dict[SourceKey, Dict] = {}
    layer_sets: List[Set[int]] = []
    layer_rows: List[Dict] = []

    for key, values in second_moments.items():
        mx = values.max().item()
        normalized[key] = values / mx if mx > 0 else values.clone()
        out_idx, threshold, log_mean, log_std = _detect_source_outliers(values, sigma)
        source_outliers[key] = out_idx
        source_stats[key] = {
            "layer": key[0],
            "src": key[1],
            "count": int(out_idx.numel()),
            "log_threshold": float(threshold),
            "log_mean": float(log_mean),
            "log_std": float(log_std),
            "max_ex2": float(values.max().item()),
            "mean_ex2": float(values.mean().item()),
        }

    for layer_idx in range(num_layers):
        attn = set(source_outliers[(layer_idx, "attn")].tolist())
        mlp = set(source_outliers[(layer_idx, "mlp")].tolist())
        union = attn | mlp
        layer_sets.append(union)
        layer_rows.append(
            {
                "layer": layer_idx,
                "attn_outliers": len(attn),
                "mlp_outliers": len(mlp),
                "union_outliers": len(union),
                "attn_mlp_jaccard": _safe_jaccard(attn, mlp),
            }
        )

    layer_jaccard = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in range(num_layers):
        for j in range(num_layers):
            layer_jaccard[i, j] = _safe_jaccard(layer_sets[i], layer_sets[j])

    return {
        "normalized": normalized,
        "source_outliers": source_outliers,
        "source_stats": source_stats,
        "layer_sets": layer_sets,
        "layer_rows": layer_rows,
        "layer_jaccard": layer_jaccard,
    }


def _segment_sources(start: int, end: int) -> List[SourceKey]:
    return [(layer_idx, src) for layer_idx in range(start, end) for src in SOURCE_NAMES]


def _snap_group_k(required_count: int, candidates: Sequence[int]) -> int:
    for group_k in candidates:
        if required_count <= group_k:
            return group_k
    return candidates[-1]


def _aggregate_score(
    sources: Iterable[SourceKey],
    normalized: Dict[SourceKey, torch.Tensor],
    hidden_size: int,
) -> torch.Tensor:
    agg = torch.zeros(hidden_size, dtype=torch.float32)
    for key in sources:
        agg.add_(normalized[key])
    return agg


def _select_bucket(
    agg_score: torch.Tensor,
    outlier_mask: torch.Tensor,
    group_k: int,
) -> Tuple[List[int], List[int]]:
    ranked = torch.argsort(agg_score, descending=True).tolist()
    selected_ranked: List[int] = []
    selected_set: Set[int] = set()

    for idx in ranked:
        if len(selected_ranked) >= group_k:
            break
        if outlier_mask[idx].item():
            selected_ranked.append(idx)
            selected_set.add(idx)

    for idx in ranked:
        if len(selected_ranked) >= group_k:
            break
        if idx not in selected_set:
            selected_ranked.append(idx)
            selected_set.add(idx)

    return sorted(selected_ranked), selected_ranked


def _avg_pairwise_jaccard(layer_jaccard: np.ndarray, start: int, end: int) -> float:
    if end - start <= 1:
        return 1.0
    vals = []
    for i in range(start, end):
        for j in range(i + 1, end):
            vals.append(float(layer_jaccard[i, j]))
    return float(np.mean(vals)) if vals else 1.0


def evaluate_segment_candidate(
    start: int,
    end: int,
    second_moments: Dict[SourceKey, torch.Tensor],
    normalized: Dict[SourceKey, torch.Tensor],
    source_outliers: Dict[SourceKey, torch.Tensor],
    layer_jaccard: np.ndarray,
    hidden_size: int,
    group_k_candidates: Sequence[int],
) -> SegmentCandidate:
    sources = _segment_sources(start, end)

    outlier_mask = torch.zeros(hidden_size, dtype=torch.bool)
    for key in sources:
        outlier_mask[source_outliers[key]] = True
    outlier_union_count = int(outlier_mask.sum().item())
    group_k = _snap_group_k(outlier_union_count, group_k_candidates)

    agg_score = _aggregate_score(sources, normalized, hidden_size)
    bucket, bucket_ranked = _select_bucket(agg_score, outlier_mask, group_k)
    bucket_t = torch.tensor(bucket, dtype=torch.long)
    bucket_set = set(bucket)

    source_total = 0
    source_captured = 0
    for key in sources:
        src_set = set(source_outliers[key].tolist())
        source_total += len(src_set)
        source_captured += len(src_set & bucket_set)
    source_capture_ratio = (
        source_captured / source_total if source_total > 0 else 1.0
    )
    satisfies_target = source_capture_ratio >= TARGET_CAPTURE - 1e-12

    total_energy = sum(second_moments[key].sum().item() for key in sources)
    if bucket_t.numel() == 0:
        selected_energy = 0.0
    else:
        selected_energy = sum(second_moments[key][bucket_t].sum().item() for key in sources)
    energy_coverage = selected_energy / (total_energy + 1e-30)

    avg_j = _avg_pairwise_jaccard(layer_jaccard, start, end)
    length = end - start
    group_work = float(length * group_k)
    selection_cost = group_work

    return SegmentCandidate(
        start=start,
        end=end,
        group_k=group_k,
        bucket=bucket,
        bucket_ranked=bucket_ranked,
        outlier_union_count=outlier_union_count,
        source_outlier_total=source_total,
        source_outlier_captured=source_captured,
        source_capture_ratio=source_capture_ratio,
        satisfies_target=satisfies_target,
        energy_coverage=energy_coverage,
        avg_pairwise_jaccard=avg_j,
        group_work=group_work,
        selection_cost=selection_cost,
    )


def build_segment_cache(
    second_moments: Dict[SourceKey, torch.Tensor],
    outlier_info: Dict,
    hidden_size: int,
    num_layers: int,
    args: argparse.Namespace,
) -> Dict[Tuple[int, int], SegmentCandidate]:
    cache: Dict[Tuple[int, int], SegmentCandidate] = {}
    for start in range(num_layers):
        for end in range(start + MIN_SEGMENT_LEN, num_layers + 1):
            cache[(start, end)] = evaluate_segment_candidate(
                start=start,
                end=end,
                second_moments=second_moments,
                normalized=outlier_info["normalized"],
                source_outliers=outlier_info["source_outliers"],
                layer_jaccard=outlier_info["layer_jaccard"],
                hidden_size=hidden_size,
                group_k_candidates=args.group_k_candidates,
            )
    return cache


def choose_segments_dp(
    cache: Dict[Tuple[int, int], SegmentCandidate],
    num_layers: int,
    max_segments: int,
) -> Tuple[List[SegmentCandidate], List[Dict]]:
    max_segments = min(max_segments, num_layers)
    inf = float("inf")
    dp = np.full((max_segments + 1, num_layers + 1), inf, dtype=np.float64)
    prev = np.full((max_segments + 1, num_layers + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for seg_count in range(1, max_segments + 1):
        for end in range(1, num_layers + 1):
            for start in range(0, end):
                if end - start < MIN_SEGMENT_LEN:
                    continue
                if (start, end) not in cache or not np.isfinite(dp[seg_count - 1, start]):
                    continue
                seg = cache[(start, end)]
                if not seg.satisfies_target:
                    continue
                cand = dp[seg_count - 1, start] + seg.selection_cost
                if cand < dp[seg_count, end]:
                    dp[seg_count, end] = cand
                    prev[seg_count, end] = start

    curve: List[Dict] = []
    best_seg_count = -1
    best_cost = inf
    for seg_count in range(1, max_segments + 1):
        feasible = bool(np.isfinite(dp[seg_count, num_layers]))
        curve.append(
            {
                "num_segments": seg_count,
                "feasible": feasible,
                "selection_cost": (
                    float(dp[seg_count, num_layers]) if feasible else None
                ),
            }
        )
        if feasible and dp[seg_count, num_layers] < best_cost - 1e-9:
            best_cost = float(dp[seg_count, num_layers])
            best_seg_count = seg_count

    if best_seg_count < 0:
        raise RuntimeError(
            "No segmentation with num_segments <= max_segments captures all true outliers. "
            "Increase --max_segments, increase the largest --group_k_candidates value, "
            "or relax the outlier threshold."
        )

    segments: List[SegmentCandidate] = []
    end = num_layers
    for seg_count in range(best_seg_count, 0, -1):
        start = int(prev[seg_count, end])
        if start < 0:
            raise RuntimeError("DP reconstruction failed.")
        segments.append(cache[(start, end)])
        end = start
    segments.reverse()
    return segments, curve


def compute_source_capture_rows(
    segments: Sequence[SegmentCandidate],
    source_outliers: Dict[SourceKey, torch.Tensor],
) -> List[Dict]:
    rows: List[Dict] = []
    for seg_idx, seg in enumerate(segments):
        bucket_set = set(seg.bucket)
        for layer_idx in range(seg.start, seg.end):
            for src in SOURCE_NAMES:
                src_set = set(source_outliers[(layer_idx, src)].tolist())
                captured = len(src_set & bucket_set)
                total = len(src_set)
                rows.append(
                    {
                        "segment": seg_idx,
                        "layer": layer_idx,
                        "src": src,
                        "captured": captured,
                        "total": total,
                        "ratio": captured / total if total > 0 else None,
                    }
                )
    return rows


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


SEGMENT_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
]


def _seg_color(seg_idx: int) -> str:
    return SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]


def _segment_label(seg_idx: int, seg: SegmentCandidate) -> str:
    return f"S{seg_idx} L{seg.start}-{seg.end - 1} k={seg.group_k}"


def _draw_segment_spans(ax, segments: Sequence[SegmentCandidate], axis: str = "x") -> None:
    for seg_idx, seg in enumerate(segments):
        color = _seg_color(seg_idx)
        if axis == "x":
            ax.axvspan(seg.start - 0.5, seg.end - 0.5, color=color, alpha=0.08)
            ax.axvline(seg.end - 0.5, color=color, lw=1.0, alpha=0.8)
        else:
            ax.axhspan(seg.start - 0.5, seg.end - 0.5, color=color, alpha=0.08)
            ax.axhline(seg.end - 0.5, color=color, lw=1.0, alpha=0.8)


def plot_layer_outlier_counts(layer_rows: List[Dict], segments: Sequence[SegmentCandidate], output_dir: Path) -> None:
    layers = np.array([row["layer"] for row in layer_rows])
    attn = np.array([row["attn_outliers"] for row in layer_rows])
    mlp = np.array([row["mlp_outliers"] for row in layer_rows])
    union = np.array([row["union_outliers"] for row in layer_rows])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.34
    ax.bar(layers - width / 2, attn, width=width, label="attn", color="#4c78a8")
    ax.bar(layers + width / 2, mlp, width=width, label="mlp", color="#f58518")
    ax.plot(layers, union, color="#111111", lw=1.8, marker="o", ms=3, label="attn union mlp")
    _draw_segment_spans(ax, segments, axis="x")

    y_top = max(1, int(max(union.max(initial=0), attn.max(initial=0), mlp.max(initial=0))))
    for seg_idx, seg in enumerate(segments):
        mid = (seg.start + seg.end - 1) / 2
        ax.text(
            mid,
            y_top * 1.08,
            _segment_label(seg_idx, seg),
            ha="center",
            va="bottom",
            fontsize=8,
            color=_seg_color(seg_idx),
            fontweight="bold",
        )

    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("True outlier channels")
    ax.set_title("Per-layer outlier counts from log(E[x^2]) distribution")
    ax.set_ylim(0, y_top * 1.22 + 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = output_dir / "fig1_layer_outlier_counts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_layer_jaccard(layer_jaccard: np.ndarray, segments: Sequence[SegmentCandidate], output_dir: Path) -> None:
    num_layers = layer_jaccard.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(layer_jaccard, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, label="Jaccard similarity", fraction=0.046, pad=0.04)

    for seg_idx, seg in enumerate(segments):
        rect = patches.Rectangle(
            (seg.start - 0.5, seg.start - 0.5),
            seg.size,
            seg.size,
            linewidth=2,
            edgecolor=_seg_color(seg_idx),
            facecolor="none",
        )
        ax.add_patch(rect)
        center = seg.start + seg.size / 2 - 0.5
        ax.text(
            center,
            center,
            f"S{seg_idx}\nk={seg.group_k}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(range(num_layers))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(range(num_layers), fontsize=6)
    ax.set_yticklabels(range(num_layers), fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Layer outlier-set Jaccard from true E[x^2] outliers")
    fig.tight_layout()
    path = output_dir / "fig2_layer_outlier_jaccard.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_source_capture_heatmap(
    capture_rows: List[Dict],
    num_layers: int,
    segments: Sequence[SegmentCandidate],
    output_dir: Path,
) -> None:
    ratio_mat = np.full((num_layers, 2), np.nan, dtype=np.float32)
    label_mat = [["" for _ in range(2)] for _ in range(num_layers)]
    for row in capture_rows:
        col = 0 if row["src"] == "attn" else 1
        layer = row["layer"]
        if row["total"] > 0:
            ratio_mat[layer, col] = row["captured"] / row["total"]
        label_mat[layer][col] = f"{row['captured']}/{row['total']}"

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#f2f2f2")

    fig, ax = plt.subplots(figsize=(4.7, max(8, num_layers * 0.34)))
    im = ax.imshow(ratio_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Captured / true outliers", fraction=0.14, pad=0.06)
    _draw_segment_spans(ax, segments, axis="y")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["attn", "mlp"], fontsize=11)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers), fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Per-source true-outlier capture by selected segment bucket")

    for layer in range(num_layers):
        for col in range(2):
            value = ratio_mat[layer, col]
            if np.isnan(value):
                color = "#777777"
            else:
                color = "black" if 0.25 < value < 0.85 else "white"
            ax.text(
                col,
                layer,
                label_mat[layer][col],
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
                fontweight="bold" if label_mat[layer][col] != "0/0" else "normal",
            )

    for seg_idx, seg in enumerate(segments):
        y = (seg.start + seg.end - 1) / 2
        ax.text(
            1.72,
            y,
            f"S{seg_idx}\nL{seg.start}-{seg.end - 1}\nk={seg.group_k}",
            color=_seg_color(seg_idx),
            fontsize=6,
            ha="left",
            va="center",
            fontweight="bold",
            clip_on=False,
        )

    fig.tight_layout()
    path = output_dir / "fig3_source_capture_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_segment_groupk(segments: Sequence[SegmentCandidate], output_dir: Path) -> None:
    labels = [f"S{i}\nL{seg.start}-{seg.end - 1}" for i, seg in enumerate(segments)]
    xs = np.arange(len(segments))
    union_counts = [seg.outlier_union_count for seg in segments]
    group_ks = [seg.group_k for seg in segments]
    capture = [seg.source_capture_ratio for seg in segments]

    fig, ax1 = plt.subplots(figsize=(max(7, len(segments) * 1.5), 4.5))
    bars = ax1.bar(xs - 0.18, union_counts, width=0.36, color="#4c78a8", label="unique outliers")
    ax1.bar(xs + 0.18, group_ks, width=0.36, color="#f58518", label="snapped group_k")
    for i, bar in enumerate(bars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(group_ks) * 0.02,
            f"{union_counts[i]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for i, gk in enumerate(group_ks):
        ax1.text(
            xs[i] + 0.18,
            gk + max(group_ks) * 0.02,
            str(gk),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    ax2.plot(xs, capture, color="#111111", marker="o", lw=1.8, label="source capture")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Source-outlier capture ratio")

    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Channel count")
    ax1.set_title("Segment bucket size snapped to allowed group_k candidates")
    ax1.grid(True, axis="y", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    path = output_dir / "fig4_segment_groupk.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cost_curve(cost_curve: List[Dict], output_dir: Path) -> None:
    if not cost_curve:
        return
    xs = [row["num_segments"] for row in cost_curve]
    feasible = [row["feasible"] for row in cost_curve]
    costs = [
        np.nan if row["selection_cost"] is None else row["selection_cost"]
        for row in cost_curve
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, costs, marker="o", lw=1.8, color="#4c78a8", label="feasible tie-break cost")
    infeasible_x = [x for x, ok in zip(xs, feasible) if not ok]
    if infeasible_x:
        ax.scatter(
            infeasible_x,
            [0.0] * len(infeasible_x),
            marker="x",
            s=70,
            color="#e45756",
            label="not feasible",
        )
    feasible_x = [x for x, ok in zip(xs, feasible) if ok]
    feasible_y = [c for c, ok in zip(costs, feasible) if ok]
    if feasible_x:
        best_idx = int(np.nanargmin(costs))
        ax.scatter([xs[best_idx]], [costs[best_idx]], s=90, color="#54a24b", zorder=3, label="selected")
    ax.set_xticks(xs)
    ax.set_xlabel("Number of segments")
    ax.set_ylabel("Tie-break cost")
    ax.set_title("Lowest group_k work with 100% outlier capture")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig5_segment_cost_curve.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def _segment_to_json(seg_idx: int, seg: SegmentCandidate) -> Dict:
    return {
        "segment": seg_idx,
        "layers": seg.layer_range,
        "size": seg.size,
        "group_k": seg.group_k,
        "outlier_union_count": seg.outlier_union_count,
        "headroom": seg.group_k - seg.outlier_union_count,
        "source_outlier_total": seg.source_outlier_total,
        "source_outlier_captured": seg.source_outlier_captured,
        "source_capture_ratio": seg.source_capture_ratio,
        "satisfies_target": seg.satisfies_target,
        "energy_coverage": seg.energy_coverage,
        "avg_pairwise_jaccard": seg.avg_pairwise_jaccard,
        "cost": {
            "group_work": seg.group_work,
            "selection_cost": seg.selection_cost,
        },
        "bucket": seg.bucket,
        "bucket_ranked": seg.bucket_ranked,
    }


def print_and_save_summary(
    args: argparse.Namespace,
    output_dir: Path,
    hidden_size: int,
    num_layers: int,
    segments: Sequence[SegmentCandidate],
    outlier_info: Dict,
    capture_rows: List[Dict],
    cost_curve: List[Dict],
) -> None:
    segment_sizes = [seg.size for seg in segments]
    segment_group_ks = [seg.group_k for seg in segments]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"hidden_size={hidden_size}  num_layers={num_layers}")
    print(
        "outlier rule: log(E[x^2]) > "
        f"mean + {args.outlier_log_sigma:g} * std, independently per source"
    )
    print(
        f"segment rule: minimize sum(segment_layers * group_k) with "
        f"num_segments <= {args.max_segments} and 100% true-outlier capture"
    )
    print(f"recommended segment_sizes={segment_sizes}")
    print(f"recommended segment_group_ks={segment_group_ks}")
    print()

    for seg_idx, seg in enumerate(segments):
        print(
            f"S{seg_idx}: L{seg.start}-{seg.end - 1} "
            f"(n={seg.size})  group_k={seg.group_k}  "
            f"unique_outliers={seg.outlier_union_count}  "
            f"source_capture={seg.source_outlier_captured}/{seg.source_outlier_total} "
            f"({seg.source_capture_ratio:.3f})  "
            f"energy={seg.energy_coverage:.3f}  "
            f"avg_jaccard={seg.avg_pairwise_jaccard:.3f}"
        )

    summary = {
        "config": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "outlier_log_sigma": args.outlier_log_sigma,
            "outlier_rule": "log(E[x^2]) > mean + sigma * std per (layer, source)",
            "group_k_candidates": args.group_k_candidates,
            "max_segments": args.max_segments,
            "target_capture": TARGET_CAPTURE,
            "segment_rule": (
                "choose the <= max_segments segmentation with the lowest "
                "sum(segment_layers * group_k), subject to 100% source-outlier capture"
            ),
        },
        "recommended": {
            "segment_sizes": segment_sizes,
            "segment_group_ks": segment_group_ks,
        },
        "segments": [_segment_to_json(i, seg) for i, seg in enumerate(segments)],
        "layers": outlier_info["layer_rows"],
        "source_stats": {
            f"{key[0]}:{key[1]}": value
            for key, value in outlier_info["source_stats"].items()
        },
        "source_capture": capture_rows,
        "segment_cost_curve": cost_curve,
        "layer_jaccard_matrix": outlier_info["layer_jaccard"].tolist(),
    }

    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {path}")
    print("=" * 78)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def load_or_collect_second_moments(args: argparse.Namespace) -> Tuple[Dict[SourceKey, torch.Tensor], int, int]:
    if args.load_second_moments:
        print(f"Loading cached second moments: {args.load_second_moments}")
        payload = torch.load(args.load_second_moments, map_location="cpu")
        second_moments = payload["second_moments"]
        hidden_size = int(payload["hidden_size"])
        num_layers = int(payload["num_layers"])
        return second_moments, hidden_size, num_layers

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers)
    print(f"  hidden_size={hidden_size}  num_layers={num_layers}")

    calibration_data = load_calibration_data(
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        dataset_name=args.dataset,
    )
    second_moments = estimate_second_moments(
        model=model,
        calibration_data=calibration_data,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    if args.save_second_moments:
        payload = {
            "second_moments": second_moments,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
        }
        torch.save(payload, args.save_second_moments)
        print(f"Saved second moments: {args.save_second_moments}")

    return second_moments, hidden_size, num_layers


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Group-k candidates: {args.group_k_candidates}")
    second_moments, hidden_size, num_layers = load_or_collect_second_moments(args)
    print(f"Collected {len(second_moments)} source vectors.")

    print("Detecting true per-source outlier channels from E[x^2] distributions ...")
    outlier_info = analyze_outlier_distribution(
        second_moments=second_moments,
        num_layers=num_layers,
        sigma=args.outlier_log_sigma,
    )

    print("Building segment candidates ...")
    cache = build_segment_cache(
        second_moments=second_moments,
        outlier_info=outlier_info,
        hidden_size=hidden_size,
        num_layers=num_layers,
        args=args,
    )

    print("Choosing segments with dynamic programming ...")
    segments, cost_curve = choose_segments_dp(
        cache=cache,
        num_layers=num_layers,
        max_segments=args.max_segments,
    )
    capture_rows = compute_source_capture_rows(
        segments=segments,
        source_outliers=outlier_info["source_outliers"],
    )

    print("Generating figures ...")
    plot_layer_outlier_counts(outlier_info["layer_rows"], segments, output_dir)
    plot_layer_jaccard(outlier_info["layer_jaccard"], segments, output_dir)
    plot_source_capture_heatmap(capture_rows, num_layers, segments, output_dir)
    plot_segment_groupk(segments, output_dir)
    plot_cost_curve(cost_curve, output_dir)

    print_and_save_summary(
        args=args,
        output_dir=output_dir,
        hidden_size=hidden_size,
        num_layers=num_layers,
        segments=segments,
        outlier_info=outlier_info,
        capture_rows=capture_rows,
        cost_curve=cost_curve,
    )
    print(f"\nAll outputs are under: {output_dir}")


if __name__ == "__main__":
    main()
