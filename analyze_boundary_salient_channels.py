#!/usr/bin/env python3
"""
analyze_boundary_salient_channels.py

Analyze salient input channels (d_model dimension) of Llama-2-7B on the residual stream
and evaluate feasibility of grouping top-k channels by (variable-length) boundary aggregation.

Usage — equal split (4 × 8 layers):
    python analyze_boundary_salient_channels.py --num_boundaries 4

Usage — custom sizes (e.g. 1+1+15+15):
    python analyze_boundary_salient_channels.py --boundary_sizes 1 1 15 15

Outputs:
    ./salient_analysis_out/
        fig1_energy_coverage.png
        fig2a_jaccard_heatmap.png
        fig2b_jaccard_within_boundary.png
        fig3_source_capture_heatmap.png
        summary.json
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze salient residual-stream channels in Llama-2-7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model identifier",
    )
    parser.add_argument("--n_samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seq_len",   type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=["wikitext", "metamath", "commonsense"],
        help=(
            "Calibration dataset (default: wikitext).\n"
            "  wikitext    — wikitext/wikitext-2-raw-v1, field: 'text'\n"
            "  metamath    — meta-math/MetaMathQA, fields: 'query' + 'response'\n"
            "  commonsense — Ctau/commonsense_qa, fields: 'question' + choices 'text'"
        ),
    )

    # Boundary specification — exactly one of the two must resolve to a valid partition.
    bnd = parser.add_mutually_exclusive_group()
    bnd.add_argument(
        "--boundary_sizes", type=int, nargs="+", metavar="N",
        help="Layers per boundary as a space-separated list (e.g. --boundary_sizes 1 1 15 15). "
             "Must sum to num_hidden_layers. Overrides --num_boundaries.",
    )
    bnd.add_argument(
        "--num_boundaries", type=int, default=4,
        help="Number of equal-size boundaries (default: 4). "
             "num_hidden_layers must be divisible by this value.",
    )

    parser.add_argument(
        "--group_k", type=int, default=128,
        help="Salient group size per boundary (must be power of 2)",
    )
    parser.add_argument(
        "--outlier_log_sigma", type=float, default=3.0,
        help="Sigma threshold in log-space for outlier detection",
    )
    parser.add_argument(
        "--top_k_ratio", type=float, default=0.015625,  # 64/4096
        help="Fraction of hidden_size for per-source top-k (default 1%%)",
    )
    parser.add_argument(
        "--agg_mode", type=str, default="full",
        choices=["full", "topk_only", "union_first"],
        help=(
            "Global aggregation strategy applied to ALL boundaries (default: full).\n"
            "  full        — agg[c] = Σ_sources normalized[src][c]\n"
            "  topk_only   — agg[c] = Σ_sources normalized[src][c] · 1[c ∈ top-k of src]\n"
            "  union_first — tier-1: ALL channels in any source's top-k (guaranteed 100%% capture\n"
            "                when |union| ≤ group_k); tier-2: remaining slots filled by full agg.\n"
            "                Within each tier, relative rank follows full agg score.\n"
            "Overridden per-boundary by --agg_modes when provided."
        ),
    )
    parser.add_argument(
        "--agg_modes", type=str, nargs="+", metavar="MODE",
        help=(
            "Per-boundary aggregation mode, one value per boundary in order.\n"
            "Length must equal the number of boundaries.\n"
            "Each value must be 'full' or 'topk_only'.\n"
            "Example (4 boundaries, only boundary 1 uses topk_only):\n"
            "  --agg_modes full topk_only full full\n"
            "Positions not specified cannot be omitted — supply all or none.\n"
            "When provided, --agg_mode is ignored."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="./salient_analysis_out",
        help="Directory for figures and summary.json",
    )
    return parser.parse_args()


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def resolve_boundary_sizes(args: argparse.Namespace, num_layers: int) -> List[int]:
    """
    Return a list of per-boundary layer counts that partitions [0, num_layers).

    If --boundary_sizes was given, validate and use it directly.
    Otherwise, divide num_layers evenly by --num_boundaries.
    """
    if args.boundary_sizes is not None:
        sizes = args.boundary_sizes
        assert all(s > 0 for s in sizes), "Every boundary size must be > 0"
        assert sum(sizes) == num_layers, (
            f"sum(boundary_sizes)={sum(sizes)} != num_layers={num_layers}. "
            f"Got: {sizes}"
        )
        return sizes
    else:
        nb = args.num_boundaries
        assert num_layers % nb == 0, (
            f"num_layers ({num_layers}) is not divisible by --num_boundaries ({nb}). "
            f"Use --boundary_sizes for unequal partitions."
        )
        return [num_layers // nb] * nb


def _boundary_offsets(boundary_sizes: List[int]) -> List[int]:
    """Return start indices [0, s0, s0+s1, ...], length = num_boundaries + 1."""
    return [0] + list(itertools.accumulate(boundary_sizes))


_VALID_MODES = {"full", "topk_only", "union_first"}


def resolve_agg_modes(args: argparse.Namespace, num_boundaries: int) -> List[str]:
    """
    Return a per-boundary list of aggregation mode strings.

    If --agg_modes is given it must contain exactly num_boundaries values, each
    in {'full', 'topk_only', 'union_first'}.  Otherwise --agg_mode (global default)
    is broadcast to all boundaries.
    """
    if args.agg_modes is not None:
        modes = args.agg_modes
        assert len(modes) == num_boundaries, (
            f"--agg_modes has {len(modes)} value(s) but there are {num_boundaries} "
            f"boundaries.  Supply exactly one mode per boundary."
        )
        bad = [m for m in modes if m not in _VALID_MODES]
        assert not bad, (
            f"Unknown agg_mode value(s) in --agg_modes: {bad}. "
            f"Allowed: {sorted(_VALID_MODES)}"
        )
        return modes
    else:
        return [args.agg_mode] * num_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

# Per-dataset loading config: (hf_path, hf_name_or_None, split, text_extractor)
_DATASET_CONFIGS: Dict[str, tuple] = {
    "wikitext": (
        "wikitext", "wikitext-2-raw-v1", "train",
        lambda item: item["text"],
    ),
    "metamath": (
        "meta-math/MetaMathQA", None, "train",
        lambda item: item["query"] + " " + item["response"],
    ),
    "commonsense": (
        "Ctau/commonsense_qa", None, "train",
        lambda item: item["question"] + " " + " ".join(item["choices"]["text"]),
    ),
}


def load_calibration_data(
    tokenizer:    AutoTokenizer,
    n_samples:    int,
    seq_len:      int,
    dataset_name: str = "wikitext",
) -> List[Dict[str, torch.Tensor]]:
    """
    Load a calibration dataset, tokenize to seq_len, return up to n_samples dicts.

    Supported dataset_name values: 'wikitext', 'metamath', 'commonsense'.
    Empty or too-short (<2 tokens) samples are skipped.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_path, hf_name, split, text_fn = _DATASET_CONFIGS[dataset_name]
    print(f"Loading calibration dataset: {dataset_name!r}  ({hf_path})")
    if hf_name:
        dataset = load_dataset(hf_path, hf_name, split=split)
    else:
        dataset = load_dataset(hf_path, split=split)

    samples: List[Dict[str, torch.Tensor]] = []
    for item in dataset:
        text: str = text_fn(item).strip()
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

    print(f"  Loaded {len(samples)} calibration samples (max_seq_len={seq_len}).")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Second-moment estimation (streaming, float32 accumulators)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_second_moments(
    model: AutoModelForCausalLM,
    calibration_data: List[Dict[str, torch.Tensor]],
    hidden_size: int,
    num_layers: int,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Estimate per-channel E[x²] for every (layer_idx, 'attn'|'mlp') source.

    Hooks:
      - q_proj.input[0]    → X_attn[l]  (= input_layernorm output)
      - gate_proj.input[0] → X_mlp[l]   (= post_attention_layernorm output)

    Accumulators stay float32 on CPU to avoid fp16 saturation.
    Returns {(l, 'attn'|'mlp'): tensor[hidden_size]}.
    """
    sum_sq:    Dict[Tuple[int, str], torch.Tensor] = {}
    tok_count: Dict[Tuple[int, str], int]          = {}
    for l in range(num_layers):
        for src in ("attn", "mlp"):
            sum_sq[(l, src)]    = torch.zeros(hidden_size, dtype=torch.float32)
            tok_count[(l, src)] = 0

    handles = []

    def _make_hook(layer_idx: int, src: str):
        def _hook(module, inp, out):
            x = inp[0].detach().reshape(-1, hidden_size).float().cpu()
            sum_sq[(layer_idx, src)].add_(x.pow(2).sum(dim=0))
            tok_count[(layer_idx, src)] += x.shape[0]
        return _hook

    layers = model.model.layers
    for l in range(num_layers):
        handles.append(layers[l].self_attn.q_proj.register_forward_hook(_make_hook(l, "attn")))
        handles.append(layers[l].mlp.gate_proj.register_forward_hook(_make_hook(l, "mlp")))

    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(calibration_data, desc="Calibration forward pass"):
                input_ids      = batch["input_ids"].to(next(model.parameters()).device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(input_ids.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    return {key: sum_sq[key] / max(tok_count[key], 1) for key in sum_sq}


# ─────────────────────────────────────────────────────────────────────────────
# Core boundary analysis
# ─────────────────────────────────────────────────────────────────────────────

def _safe_jaccard(set_a: Set[int], set_b: Set[int]) -> float:
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def _build_agg_score(
    b_sources:    List[Tuple[int, str]],
    normalized:   Dict[Tuple[int, str], torch.Tensor],
    topk_indices: Dict[Tuple[int, str], torch.Tensor],
    hidden_size:  int,
    agg_mode:     str,
) -> torch.Tensor:
    """
    Compute the boundary-level aggregate score vector (float32, CPU).

    full:
        agg[c] = Σ_sources  normalized[src][c]
        Every channel contributes its full normalized value from every source.

    topk_only:
        agg[c] = Σ_sources  normalized[src][c] · 1[c ∈ top-k of src]
        Each source only votes for its own top-k channels; others score 0.

    union_first:
        Two-tier score designed to guarantee 100% per-source capture when
        |union of top-k| ≤ group_k.

        full_agg  = Σ_sources  normalized[src][c]          (full, for intra-tier rank)
        norm_agg  = full_agg / max(full_agg)               ∈ [0, 1)
        score[c]  = norm_agg[c] + 1   if c ∈ ⋃ top-k      → tier-1, score ∈ [1, 2)
                  = norm_agg[c]        otherwise            → tier-2, score ∈ [0, 1)

        topk() on this score always selects all tier-1 channels first (in descending
        full-agg order within tier-1), then fills remaining slots from tier-2.
        When |union| > group_k the top group_k channels from the union are taken
        (still ranked by full agg score).
    """
    # Full agg score is needed by all three modes for intra-tier ranking.
    full_agg = torch.zeros(hidden_size, dtype=torch.float32)
    for key in b_sources:
        full_agg.add_(normalized[key])

    if agg_mode == "full":
        return full_agg

    if agg_mode == "topk_only":
        agg = torch.zeros(hidden_size, dtype=torch.float32)
        for key in b_sources:
            idx = topk_indices[key]
            agg[idx] += normalized[key][idx]
        return agg

    if agg_mode == "union_first":
        # Build union mask across all sources in this boundary.
        union_mask = torch.zeros(hidden_size, dtype=torch.bool)
        for key in b_sources:
            union_mask[topk_indices[key]] = True

        # Normalise full_agg to [0, 1) so the +1 tier offset is always dominant.
        fa_max = full_agg.max().item()
        norm_agg = full_agg / (fa_max + 1e-30)   # ∈ [0, 1)

        # Tier-1 channels (union) sit in [1, 2); tier-2 in [0, 1).
        return norm_agg + union_mask.float()

    raise ValueError(f"Unknown agg_mode: {agg_mode!r}")


def run_boundary_analysis(
    second_moments:    Dict[Tuple[int, str], torch.Tensor],
    boundary_sizes:    List[int],
    group_k:           int,
    top_k_ratio:       float,
    outlier_log_sigma: float,
    agg_modes:         List[str] = None,   # per-boundary; defaults to ["full"] * n_b
) -> Dict:
    """
    Run all three analyses (energy coverage, Jaccard overlap, capture rate)
    for every boundary.  boundary_sizes[b] is the number of layers in boundary b.
    agg_modes[b] controls how per-source scores are aggregated for boundary b
    (see _build_agg_score).
    """
    num_boundaries: int = len(boundary_sizes)
    num_layers:     int = sum(boundary_sizes)
    b_offsets:      List[int] = _boundary_offsets(boundary_sizes)   # length n_b+1
    hidden_size:    int = next(iter(second_moments.values())).shape[0]
    top_k:          int = round(top_k_ratio * hidden_size)

    if agg_modes is None:
        agg_modes = ["full"] * num_boundaries
    assert len(agg_modes) == num_boundaries, (
        f"agg_modes length {len(agg_modes)} != num_boundaries {num_boundaries}"
    )

    print(f"hidden_size={hidden_size}  num_layers={num_layers}  "
          f"num_boundaries={num_boundaries}  top_k={top_k}  group_k={group_k}")
    print(f"boundary_sizes={boundary_sizes}")
    print(f"agg_modes={agg_modes}")

    # ── Per-source normalized E[x²] (÷ per-source max) ───────────────────────
    normalized: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, val in second_moments.items():
        mx = val.max().item()
        normalized[key] = val / mx if mx > 0 else val.clone()

    # ── Per-source top-k indices ──────────────────────────────────────────────
    topk_indices: Dict[Tuple[int, str], torch.Tensor] = {
        key: nv.topk(top_k).indices for key, nv in normalized.items()
    }

    # ── Per-source outlier sets (log-space, mean + sigma * threshold) ─────────
    outlier_sets: Dict[Tuple[int, str], Set[int]] = {}
    for key, val in second_moments.items():
        log_v  = torch.log(val.clamp(min=1e-30))
        thresh = log_v.mean().item() + outlier_log_sigma * log_v.std().item()
        outlier_sets[key] = set(torch.where(log_v > thresh)[0].tolist())

    # ── Per-block salient sets (attn ∪ mlp top-k) for Jaccard matrix ─────────
    block_salient: Dict[int, Set[int]] = {
        l: set(topk_indices[(l, "attn")].tolist()) | set(topk_indices[(l, "mlp")].tolist())
        for l in range(num_layers)
    }

    # ── num_layers × num_layers Jaccard matrix ────────────────────────────────
    jac_matrix = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in range(num_layers):
        for j in range(num_layers):
            jac_matrix[i, j] = _safe_jaccard(block_salient[i], block_salient[j])

    k_scan = [2 ** p for p in range(5, 10)]   # 32, 64, 128, 256, 512

    results: Dict = {
        "config": {
            "num_layers":        num_layers,
            "num_boundaries":    num_boundaries,
            "boundary_sizes":    boundary_sizes,
            "b_offsets":         b_offsets,
            "hidden_size":       hidden_size,
            "top_k":             top_k,
            "group_k":           group_k,
            "top_k_ratio":       top_k_ratio,
            "outlier_log_sigma": outlier_log_sigma,
            "agg_modes":         agg_modes,
        },
        "boundaries":     {},
        "jaccard_matrix": jac_matrix.tolist(),
    }

    # ── Per-boundary analysis ─────────────────────────────────────────────────
    for b in range(num_boundaries):
        b_layers  = list(range(b_offsets[b], b_offsets[b + 1]))
        b_sources = [(l, s) for l in b_layers for s in ("attn", "mlp")]

        # Aggregate score: mode-dependent combination of per-source normalized E[x²]
        agg_score = _build_agg_score(
            b_sources, normalized, topk_indices, hidden_size, agg_modes[b]
        )

        # Salient group: top group_k channels by aggregate score
        group_indices: torch.Tensor = agg_score.topk(group_k).indices
        group_set:     Set[int]     = set(group_indices.tolist())

        # Boundary outlier set = union over all sources
        boundary_outlier_set: Set[int] = set()
        for key in b_sources:
            boundary_outlier_set |= outlier_sets[key]

        # ── Analysis 1: energy & outlier coverage ─────────────────────────────
        total_energy = sum(second_moments[key].sum().item() for key in b_sources)

        def _energy_cov(k_idx: torch.Tensor, _src=b_sources, _te=total_energy) -> float:
            return sum(second_moments[k][k_idx].sum().item() for k in _src) / (_te + 1e-30)

        def _outlier_cov(k_set: Set[int], _obs=boundary_outlier_set) -> float:
            return len(k_set & _obs) / len(_obs) if _obs else 1.0

        energy_curve, outlier_curve = [], []
        for k_val in k_scan:
            ki = agg_score.topk(k_val).indices
            energy_curve.append(_energy_cov(ki))
            outlier_curve.append(_outlier_cov(set(ki.tolist())))

        # ── Analysis 2: within-boundary adjacent-block Jaccard ────────────────
        jaccard_within: List[float] = [
            _safe_jaccard(block_salient[b_layers[i]], block_salient[b_layers[i + 1]])
            for i in range(len(b_layers) - 1)
        ]
        avg_jaccard = float(np.mean(jaccard_within)) if jaccard_within else 0.0

        # ── Analysis 3: per-source capture rate ───────────────────────────────
        source_capture: Dict[str, Dict] = {}
        for key in b_sources:
            captured = len(set(topk_indices[key].tolist()) & group_set)
            source_capture[str(key)] = {
                "layer":    key[0],
                "src":      key[1],
                "captured": captured,
                "top_k":    top_k,
                "ratio":    captured / top_k,
            }

        results["boundaries"][b] = {
            "layers":                 b_layers,
            "boundary_size":          len(b_layers),
            "group_indices":          group_indices.tolist(),
            "energy_coverage":        _energy_cov(group_indices),
            "outlier_coverage":       _outlier_cov(group_set),
            "boundary_outlier_count": len(boundary_outlier_set),
            "k_scan":                 k_scan,
            "energy_curve":           energy_curve,
            "outlier_curve":          outlier_curve,
            "jaccard_within":         jaccard_within,
            "avg_jaccard_within":     avg_jaccard,
            "source_capture":         source_capture,
            "agg_mode":               agg_modes[b],
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# Up to 8 distinct colors; cycled if num_boundaries > 8.
_BOUNDARY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _bc(b: int) -> str:
    return _BOUNDARY_COLORS[b % len(_BOUNDARY_COLORS)]


def _b_label(b: int, bd: Dict) -> str:
    first, last = bd["layers"][0], bd["layers"][-1]
    return f"B{b} ({first}–{last}, n={bd['boundary_size']})"


def plot_fig1_energy_coverage(results: Dict, output_dir: Path) -> None:
    """
    Figure 1: Energy (solid) and outlier (dashed) coverage vs group size k.
    One curve pair per boundary; vertical line marks the configured group_k.
    """
    cfg = results["config"]
    n_b = cfg["num_boundaries"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for b in range(n_b):
        bd     = results["boundaries"][b]
        k_vals = bd["k_scan"]
        color  = _bc(b)
        label  = _b_label(b, bd)
        ax.plot(k_vals, bd["energy_curve"],  color=color, marker="o", lw=2,
                label=f"{label} energy")
        ax.plot(k_vals, bd["outlier_curve"], color=color, marker="s", lw=1.5,
                linestyle="--", alpha=0.75, label=f"{label} outlier")

    ax.axvline(cfg["group_k"], color="grey", lw=1, linestyle=":",
               label=f"group_k={cfg['group_k']}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Group size k", fontsize=12)
    ax.set_ylabel("Coverage ratio", fontsize=12)
    modes_str = ", ".join(f"B{i}:{m}" for i, m in enumerate(cfg["agg_modes"]))
    ax.set_title(
        f"Energy & Outlier Coverage vs. Group Size k\n[{modes_str}]",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig1_energy_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_fig2a_jaccard_heatmap(results: Dict, output_dir: Path) -> None:
    """
    Figure 2a: num_layers × num_layers Jaccard heatmap.
    Cyan rectangles mark each boundary block (variable size).
    """
    cfg            = results["config"]
    num_layers     = cfg["num_layers"]
    num_boundaries = cfg["num_boundaries"]
    boundary_sizes = cfg["boundary_sizes"]
    b_offsets      = cfg["b_offsets"]          # list[int], length n_b+1
    jac            = np.array(results["jaccard_matrix"])

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(jac, cmap="hot", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Jaccard similarity", fraction=0.046, pad=0.04)

    for b in range(num_boundaries):
        start = b_offsets[b]
        size  = boundary_sizes[b]
        rect  = patches.Rectangle(
            (start - 0.5, start - 0.5), size, size,
            linewidth=2, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(rect)
        # Label in the centre of the diagonal block
        cx = start + size / 2 - 0.5
        ax.text(cx, cx, f"B{b}\n(n={size})", color="cyan", fontsize=8,
                ha="center", va="center", fontweight="bold")

    ax.set_xticks(b_offsets[:-1])
    ax.set_yticks(b_offsets[:-1])
    ax.set_xticklabels([f"L{o}" for o in b_offsets[:-1]], fontsize=8)
    ax.set_yticklabels([f"L{o}" for o in b_offsets[:-1]], fontsize=8)
    ax.set_xlabel("Block index", fontsize=11)
    ax.set_ylabel("Block index", fontsize=11)
    ax.set_title(
        "All-pair Jaccard similarity of per-block salient channels\n"
        "(attn ∪ mlp top-k;  cyan boxes = boundaries)",
        fontsize=12,
    )

    plt.tight_layout()
    path = output_dir / "fig2a_jaccard_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_fig2b_jaccard_within_boundary(results: Dict, output_dir: Path) -> None:
    """
    Figure 2b: Average within-boundary adjacent-block Jaccard (bar chart).
    Bar width is proportional to boundary size so narrow boundaries are visually narrower.
    """
    cfg            = results["config"]
    n_b            = cfg["num_boundaries"]
    boundary_sizes = cfg["boundary_sizes"]

    avgs   = [results["boundaries"][b]["avg_jaccard_within"] for b in range(n_b)]
    labels = [_b_label(b, results["boundaries"][b]) for b in range(n_b)]

    # Bar widths proportional to boundary sizes (normalised so total ≈ n_b)
    total_layers = sum(boundary_sizes)
    bar_widths   = [s / total_layers * n_b * 0.8 for s in boundary_sizes]

    # X positions: centre each bar so they tile neatly
    xs = []
    x  = 0.0
    for w in bar_widths:
        xs.append(x + w / 2)
        x += w + 0.05

    fig, ax = plt.subplots(figsize=(max(6, n_b * 1.4), 4))
    for b in range(n_b):
        bar = ax.bar(xs[b], avgs[b], width=bar_widths[b],
                     color=_bc(b), alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.text(xs[b], avgs[b] + 0.015, f"{avgs[b]:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Avg Jaccard (adjacent blocks)", fontsize=11)
    ax.set_title("Within-boundary adjacent-block Jaccard similarity\n"
                 "(bar width ∝ boundary size)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig2b_jaccard_within_boundary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_fig3_capture_heatmap(results: Dict, output_dir: Path) -> None:
    """
    Figure 3: Per-source capture-rate heatmap.
    Rows = layers, columns = {attn, mlp}.
    Cyan lines mark boundary divisions (variable positions).
    """
    cfg        = results["config"]
    num_layers = cfg["num_layers"]
    n_b        = cfg["num_boundaries"]
    b_offsets  = cfg["b_offsets"]

    cap_mat = np.full((num_layers, 2), fill_value=np.nan, dtype=np.float32)
    for b in range(n_b):
        for cd in results["boundaries"][b]["source_capture"].values():
            col = 0 if cd["src"] == "attn" else 1
            cap_mat[cd["layer"], col] = cd["ratio"]

    fig, ax = plt.subplots(figsize=(3.5, max(8, num_layers * 0.35)))
    im = ax.imshow(cap_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Capture ratio", fraction=0.15, pad=0.06)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["attn", "mlp"], fontsize=12)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers), fontsize=6)
    ax.set_ylabel("Layer index", fontsize=11)
    ax.set_title(
        f"Per-source capture ratio\n"
        f"(top-{cfg['top_k']} channels in group-{cfg['group_k']})",
        fontsize=11,
    )

    # Boundary dividers at variable positions
    boundary_agg_modes = [results["boundaries"][b]["agg_mode"] for b in range(n_b)]
    for b in range(1, n_b):
        ax.axhline(y=b_offsets[b] - 0.5, color="cyan", linewidth=1.5)
        ax.text(
            1.55, b_offsets[b] - 0.5,
            f"↑B{b-1}[{boundary_agg_modes[b-1]}]\n↓B{b}[{boundary_agg_modes[b]}]",
            color="cyan", fontsize=4.5, va="center",
        )

    for l in range(num_layers):
        for c in range(2):
            v = cap_mat[l, c]
            if not np.isnan(v):
                txt_color = "black" if 0.25 < v < 0.80 else "white"
                ax.text(c, l, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=txt_color)

    plt.tight_layout()
    path = output_dir / "fig3_source_capture_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(results: Dict, output_dir: Path) -> None:
    """Print per-boundary key metrics to stdout and write summary.json."""
    cfg = results["config"]
    sep = "=" * 70

    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  hidden_size={cfg['hidden_size']}  num_layers={cfg['num_layers']}  "
          f"num_boundaries={cfg['num_boundaries']}")
    print(f"  boundary_sizes={cfg['boundary_sizes']}")
    print(f"  top_k={cfg['top_k']}  group_k={cfg['group_k']}  "
          f"outlier_sigma={cfg['outlier_log_sigma']}")
    print(f"  agg_modes={cfg['agg_modes']}")
    print()

    summary: Dict = {"config": cfg, "boundaries": {}}

    for b in range(cfg["num_boundaries"]):
        bd         = results["boundaries"][b]
        cap_ratios = [v["ratio"] for v in bd["source_capture"].values()]
        cap_mean   = float(np.mean(cap_ratios))
        cap_min    = float(np.min(cap_ratios))
        cap_max    = float(np.max(cap_ratios))

        print(f"  Boundary {b}  {_b_label(b, bd)}  [agg_mode={bd['agg_mode']}]:")
        print(f"    Energy coverage  @ group_k={cfg['group_k']}: "
              f"{bd['energy_coverage']:.4f}")
        print(f"    Outlier coverage "
              f"({bd['boundary_outlier_count']} outlier channels): "
              f"{bd['outlier_coverage']:.4f}")
        print(f"    Avg within-boundary Jaccard: {bd['avg_jaccard_within']:.4f}")
        print(f"    Source capture  mean={cap_mean:.3f}  "
              f"min={cap_min:.3f}  max={cap_max:.3f}")
        print()

        summary["boundaries"][str(b)] = {
            "layers":                 [bd["layers"][0], bd["layers"][-1]],
            "boundary_size":          bd["boundary_size"],
            "agg_mode":               bd["agg_mode"],
            "energy_coverage":        bd["energy_coverage"],
            "outlier_coverage":       bd["outlier_coverage"],
            "boundary_outlier_count": bd["boundary_outlier_count"],
            "avg_jaccard_within":     bd["avg_jaccard_within"],
            "source_capture_mean":    cap_mean,
            "source_capture_min":     cap_min,
            "source_capture_max":     cap_max,
            "energy_curve":  {str(k): v for k, v in zip(bd["k_scan"], bd["energy_curve"])},
            "outlier_curve": {str(k): v for k, v in zip(bd["k_scan"], bd["outlier_curve"])},
        }

    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON → {path}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    assert _is_power_of_two(args.group_k), f"--group_k={args.group_k} must be a power of 2"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    hidden_size: int = model.config.hidden_size
    num_layers:  int = model.config.num_hidden_layers
    print(f"  hidden_size={hidden_size}  num_layers={num_layers}")

    # Resolve boundary sizes and per-boundary agg modes (need num_layers first)
    boundary_sizes: List[int] = resolve_boundary_sizes(args, num_layers)
    agg_modes:      List[str] = resolve_agg_modes(args, len(boundary_sizes))
    print(f"  boundary_sizes={boundary_sizes}  (sum={sum(boundary_sizes)})")
    print(f"  agg_modes={agg_modes}")

    cal_data = load_calibration_data(tokenizer, args.n_samples, args.seq_len, args.dataset)

    print("Estimating per-channel E[x²] …")
    second_moments = estimate_second_moments(model, cal_data, hidden_size, num_layers)
    print(f"  {len(second_moments)} source vectors  "
          f"shape={next(iter(second_moments.values())).shape}")

    print("Running boundary analysis …")
    results = run_boundary_analysis(
        second_moments    = second_moments,
        boundary_sizes    = boundary_sizes,
        group_k           = args.group_k,
        top_k_ratio       = args.top_k_ratio,
        outlier_log_sigma = args.outlier_log_sigma,
        agg_modes         = agg_modes,
    )

    print("Generating figures …")
    plot_fig1_energy_coverage(results, output_dir)
    plot_fig2a_jaccard_heatmap(results, output_dir)
    plot_fig2b_jaccard_within_boundary(results, output_dir)
    plot_fig3_capture_heatmap(results, output_dir)

    print_and_save_summary(results, output_dir)
    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
