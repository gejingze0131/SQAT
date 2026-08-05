#!/usr/bin/env python
"""
Collect lm-eval JSON results into a single flat CSV (results_saltq.csv by default).

The eval scripts write one timestamped JSON per run under results/<suite>/. That is fine for a
single run and useless for a sweep, so this folds them into one long-format table — one row per
(run, task, metric) — that pandas/Excel can pivot directly.

Run-level context (bits / group_size / group_k / the three learning rates / segmentation) is
recovered from the artifacts that sit next to the evaluated model rather than being retyped:

    <model_path>/sqat_permute_meta.pt   group_size, group_k, boundary_sizes, per-layer group_k
    <saltq_base>/saltq_meta.pt          q_bits, symmetric, the frozen/trainable parameter split
    --config <yaml>                     salient_lr, scales_lr, batch/lr, dataset

so a row records what was ACTUALLY run, not what the config said at collection time.

Idempotent: re-running only appends rows whose (timestamp, model, task, metric) is not already
in the CSV, so it is safe to call at the end of every pipeline run.

Usage:
    python scripts/collect_saltq_results.py                      # scan results/ -> results_saltq.csv
    python scripts/collect_saltq_results.py --results_dir results/math --config configs/saltq.yaml
    python scripts/collect_saltq_results.py --note "salient_lr sweep, 1e-5"
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMNS = [
    "timestamp",
    "method",
    "model_dir",
    "dataset",
    "task",
    "metric",
    "value",
    "stderr",
    "num_fewshot",
    "bits",
    "symmetric",
    "group_size",
    "group_k",
    "boundary_sizes",
    "salient_lr",
    "scales_lr",
    "base_lr",
    "epochs",
    "eff_batch",
    "trainable_salient_M",
    "trainable_qparams_M",
    "frozen_codes_M",
    "result_json",
    "note",
]

# lm-eval reports every metric twice (value and its ",stderr" twin); pair them up instead of
# emitting stderr as if it were a separate metric.
_STDERR_SUFFIX = "_stderr"


def _load_torch_meta(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a broken meta must not kill the collection
        print(f"[collect] WARNING: could not read {path}: {exc}")
        return None


def _infer_method(model_dir: str) -> str:
    name = os.path.basename(model_dir.rstrip("/")).lower()
    for key, label in (
        ("saltq", "SALT-Q"),
        ("sqat_permute", "Permuted-SQAT"),
        ("qalora", "QA-LoRA"),
        ("-full-", "LR-QAT"),
        ("-none-", "QLoRA"),
    ):
        if key in name:
            return label
    return "unknown"


def _run_context(model_dir: str, cfg: Optional[dict]) -> Dict[str, object]:
    """Recover what was actually run from the artifacts shipped with the evaluated model."""
    ctx: Dict[str, object] = {c: "" for c in COLUMNS}

    perm = _load_torch_meta(os.path.join(model_dir, "sqat_permute_meta.pt"))
    if perm:
        mm = perm.get("model", perm) if isinstance(perm, dict) else {}
        ctx["group_size"] = mm.get("group_size", "")
        ctx["group_k"] = mm.get("group_k", "")
        bs = mm.get("boundary_sizes")
        ctx["boundary_sizes"] = "|".join(str(x) for x in bs) if bs else ""

    # The SALT-Q base carries the authoritative bit width and the freedom split. It is referenced
    # from the training checkpoint, which we do not have here, so try the conventional location.
    for cand in (
        os.path.join(os.path.dirname(model_dir.rstrip("/")), "saltq", "saltq_base"),
        os.path.join(model_dir, "saltq_base"),
    ):
        sq = _load_torch_meta(os.path.join(cand, "saltq_meta.pt"))
        if sq:
            ctx["bits"] = sq.get("q_bits", "")
            ctx["symmetric"] = sq.get("symmetric", "")
            pc = sq.get("param_counts", {})
            ctx["trainable_salient_M"] = round(pc.get("trainable_salient_weights", 0) / 1e6, 1)
            ctx["trainable_qparams_M"] = round(pc.get("trainable_qparams", 0) / 1e6, 1)
            ctx["frozen_codes_M"] = round(pc.get("frozen_codes", 0) / 1e6, 1)
            break

    if cfg:
        sq_cfg = (cfg.get("qat", {}) or {}).get("saltq", {}) or {}
        tr = cfg.get("training", {}) or {}
        ctx["salient_lr"] = sq_cfg.get("salient_lr", "")
        ctx["scales_lr"] = sq_cfg.get("scales_lr", "")
        ctx["base_lr"] = tr.get("learning_rate", "")
        ctx["epochs"] = tr.get("num_epochs", "")
        ctx["dataset"] = (cfg.get("data", {}) or {}).get("train_dataset", "")
        if not ctx["bits"]:
            ctx["bits"] = (cfg.get("model", {}) or {}).get("quant_bits", "")
        if not ctx["symmetric"]:
            ctx["symmetric"] = (cfg.get("qat", {}) or {}).get("symmetric", "")
        try:
            import yaml  # noqa: F401  (already parsed; kept for the accelerate hint below)

            n_gpu = 1
            acc = "accelerate_config.yaml"
            if os.path.exists(acc):
                import yaml as _y

                n_gpu = int(_y.safe_load(open(acc)).get("num_processes", 1))
            ctx["eff_batch"] = (
                int(tr.get("per_device_train_batch_size", 0))
                * int(tr.get("gradient_accumulation_steps", 0))
                * n_gpu
            )
        except Exception:  # noqa: BLE001
            ctx["eff_batch"] = ""
    return ctx


def _rows_from_json(path: str, cfg: Optional[dict], note: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    conf = payload.get("config", {})
    model_dir = conf.get("model_path", "")
    ctx = _run_context(model_dir, cfg)
    ctx.update(
        timestamp=payload.get("timestamp", ""),
        method=_infer_method(model_dir),
        model_dir=model_dir,
        num_fewshot=conf.get("num_fewshot", ""),
        result_json=path,
        note=note,
    )

    rows: List[Dict[str, object]] = []
    for task, metrics in (payload.get("results", {}) or {}).items():
        # split "<metric>,<filter>" / "<metric>_stderr,<filter>" into value + stderr pairs
        values, errors = {}, {}
        for key, val in metrics.items():
            if key == "alias" or not isinstance(val, (int, float)):
                continue
            name, _, flt = key.partition(",")
            target = errors if name.endswith(_STDERR_SUFFIX) else values
            if name.endswith(_STDERR_SUFFIX):
                name = name[: -len(_STDERR_SUFFIX)]
            target[(name, flt)] = val
        for (name, flt), val in values.items():
            row = dict(ctx)
            row["task"] = task
            row["metric"] = f"{name},{flt}" if flt else name
            row["value"] = val
            row["stderr"] = errors.get((name, flt), "")
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Fold lm-eval JSON results into one CSV")
    ap.add_argument("--results_dir", default="results",
                    help="Directory to scan recursively for *.json (default: results)")
    ap.add_argument("--csv", default="results_saltq.csv", help="Output CSV (appended to)")
    ap.add_argument("--config", default="configs/saltq.yaml",
                    help="Config yaml used for the run, for lr/dataset context")
    ap.add_argument("--filter", default="",
                    help="Only ingest results whose model_dir contains this substring "
                         "(e.g. 'saltq'). Empty = everything.")
    ap.add_argument("--note", default="", help="Free-text note stamped on the new rows")
    args = ap.parse_args()

    cfg = None
    if args.config and os.path.exists(args.config):
        import yaml

        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    existing = set()
    if os.path.exists(args.csv):
        with open(args.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add((row.get("timestamp"), row.get("model_dir"),
                              row.get("task"), row.get("metric")))

    new_rows: List[Dict[str, object]] = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True)):
        try:
            rows = _rows_from_json(path, cfg, args.note)
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] skipping {path}: {exc}")
            continue
        for row in rows:
            if args.filter and args.filter not in str(row["model_dir"]):
                continue
            key = (row["timestamp"], row["model_dir"], row["task"], row["metric"])
            if key in existing:
                continue
            existing.add(key)
            new_rows.append(row)

    if not new_rows:
        print(f"[collect] no new results under {args.results_dir} -> {args.csv} unchanged")
        return 0

    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)

    print(f"[collect] appended {len(new_rows)} row(s) to {args.csv}")
    for row in new_rows:
        val = row["value"]
        val = f"{val:.4f}" if isinstance(val, float) else val
        print(f"    {row['method']:14s} {row['task']:20s} {row['metric']:28s} {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
