"""Fold Commonsense-170k eval JSONs into one long-format CSV.

Two evaluators write into results/commonsense_170k/ and they have different shapes:

  eval_benchmarks.py -> <model>_bench_<ts>.json   {"summary": {task: {score, metric}}, "results": ...}
  eval_mmlu.py       -> <model>_<ts>.json         {"results": {"mmlu": ..., "mmlu_<subject>": ...}}

MMLU IS DELIBERATELY REDUCED TO ONE ROW. lm-eval emits the aggregate `mmlu` plus four category
groups plus ~57 individual subjects; the subjects are noisy at this sample count and would bury
the eight commonsense tasks under 60x their number of rows. Only the aggregate is kept. Pass
--mmlu_groups if the four category groups (humanities / social_sciences / stem / other) are wanted
too; individual subjects are never emitted.

An AVERAGE row is written per model over the eight commonsense tasks, which is the number the
commonsense literature reports. MMLU is excluded from that average on purpose — it is a separate
zero-shot knowledge benchmark, not one of the eight, and folding it in would make the average
incomparable to published tables.

Idempotent: rows already present in the CSV (keyed on timestamp + model_dir + task + metric) are
never duplicated, so re-running after each job is safe.

Usage:
    python scripts/collect_commonsense_results.py
    python scripts/collect_commonsense_results.py --note "3 epochs, autoseg k=128"
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional

COLUMNS = [
    "source", "timestamp", "method", "model_dir", "variant", "bits", "group_size",
    "task", "metric", "value", "stderr", "num_fewshot", "epochs", "eff_batch",
    "lr", "result_json", "note",
]

# The eight tasks of the commonsense-170k suite. MMLU is tracked but excluded from AVERAGE.
COMMONSENSE_TASKS = [
    "boolq", "piqa", "social_iqa", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa",
]


def _infer_method(model_dir: str) -> str:
    """Order matters: ablation dirs contain the parent method's name."""
    name = os.path.basename(model_dir.rstrip("/")).lower()
    for key, label in (
        ("saltq", "SALT-Q"),
        ("qalora", "QA-LoRA"),
        ("qlora-none", "QLoRA (merged)"),
        ("-none-", "QLoRA (merged)"),
        ("sqat_permute", "Permuted-SQAT"),
        ("-full-", "LR-QAT"),
    ):
        if key in name:
            return label
    return "unknown"


def _infer_variant(model_dir: str) -> str:
    """Which export was evaluated — they are NOT interchangeable."""
    name = os.path.basename(model_dir.rstrip("/")).lower()
    if "dequant" in name:
        return "dequant (deployed INT-b)"
    if "merged" in name:
        return "fp16 merged (no quant error)"
    if "deploy-eval" in name:
        return "deployed"
    return ""


def _infer_bits_gs(model_dir: str) -> (str, str):
    name = os.path.basename(model_dir.rstrip("/")).lower()
    bits = gs = ""
    m = re.search(r"int(\d)", name) or re.search(r"(\d)bit", name)
    if m:
        bits = m.group(1)
    m = re.search(r"_g(\d+)", name)
    if m:
        gs = m.group(1)
    return bits, gs


def _split_metric(key: str) -> (str, str):
    """'acc,none' -> ('acc', 'none'); 'acc' -> ('acc', '')."""
    if "," in key:
        a, b = key.split(",", 1)
        return a, b
    return key, ""


def _cfg_ctx(cfg: Optional[dict]) -> Dict[str, object]:
    if not cfg:
        return {}
    t = cfg.get("training", {})
    eff = ""
    try:
        eff = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * 4
    except Exception:  # noqa: BLE001
        pass
    return {"epochs": t.get("num_epochs", ""), "eff_batch": eff,
            "lr": t.get("learning_rate", "")}


def _rows_from_json(path: str, cfg: Optional[dict], note: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    conf = blob.get("config", {})
    model_dir = conf.get("model_path", "")
    ts = blob.get("timestamp", "")
    base = {
        "source": "lm-eval",
        "timestamp": ts,
        "method": _infer_method(model_dir),
        "model_dir": model_dir,
        "variant": _infer_variant(model_dir),
        "num_fewshot": conf.get("num_fewshot", ""),
        "result_json": path,
        "note": note,
    }
    bits, gs = _infer_bits_gs(model_dir)
    base["bits"], base["group_size"] = bits, gs
    base.update(_cfg_ctx(cfg))

    rows: List[Dict[str, object]] = []
    raw = blob.get("results", {}) or {}

    # ---- MMLU file: aggregate only ----
    if any(k == "mmlu" or k.startswith("mmlu_") for k in raw):
        wanted = {"mmlu"}
        if _rows_from_json.mmlu_groups:
            wanted |= {"mmlu_humanities", "mmlu_social_sciences", "mmlu_stem", "mmlu_other"}
        for task, res in raw.items():
            if task not in wanted:
                continue
            for key, val in res.items():
                if not isinstance(val, (int, float)) or key.endswith("_stderr") \
                        or "stderr" in key or key == "alias":
                    continue
                metric, filt = _split_metric(key)
                stderr = res.get(f"{metric}_stderr,{filt}" if filt else f"{metric}_stderr", "")
                rows.append({**base, "task": task, "metric": key, "value": val,
                             "stderr": stderr})
        return rows

    # ---- benchmark file: use the precomputed per-task summary ----
    summary = blob.get("summary", {}) or {}
    scores = []
    for task, s in summary.items():
        if task == "mmlu":
            continue  # MMLU has its own file; do not double-count it here
        val = s.get("score")
        if val is None:
            continue
        rows.append({**base, "task": task, "metric": s.get("metric", "acc"),
                     "value": val, "stderr": ""})
        if task in COMMONSENSE_TASKS:
            scores.append(val)

    if scores:
        rows.append({**base, "task": "AVERAGE", "metric": f"mean_of_{len(scores)}",
                     "value": sum(scores) / len(scores), "stderr": ""})
    return rows


_rows_from_json.mmlu_groups = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/commonsense_170k")
    ap.add_argument("--csv", default="results_commonsense_170k.csv")
    ap.add_argument("--config", default=None,
                    help="Optional yaml to recover epochs / eff_batch / lr from.")
    ap.add_argument("--filter", default="",
                    help="Only keep rows whose model_dir contains this substring.")
    ap.add_argument("--mmlu_groups", action="store_true",
                    help="Also emit the four MMLU category groups (never individual subjects).")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    _rows_from_json.mmlu_groups = args.mmlu_groups

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
    for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"),
                                 recursive=True)):
        if path.endswith("_samples.json"):
            continue
        try:
            rows = _rows_from_json(path, cfg, args.note)
        except Exception as exc:  # noqa: BLE001
            print(f"[collect-cs] skipping {path}: {exc}")
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
        print(f"[collect-cs] nothing new to add to {args.csv}")
        return

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"[collect-cs] appended {len(new_rows)} row(s) to {args.csv}")
    for row in new_rows:
        v = row["value"]
        v = f"{v:.4f}" if isinstance(v, (int, float)) else v
        print(f"    {row['method']:16s} {str(row['variant'])[:24]:24s} "
              f"{row['task']:16s} {row['metric']:14s} {v}")


if __name__ == "__main__":
    main()
