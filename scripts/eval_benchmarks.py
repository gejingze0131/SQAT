#!/usr/bin/env python
"""
Multi-benchmark 0-shot evaluation via HuggingFace backend + lm-eval-harness.

Benchmarks: boolq, hellaswag, winogrande, arc_easy, arc_challenge, piqa,
            social_iqa, openbookqa  (+ mmlu if requested)

Usage:
  # Run all benchmarks on a merged model
  python scripts/eval_benchmarks.py --model_path outputs/qlora-4bit-none-merged

  # Run specific benchmarks
  python scripts/eval_benchmarks.py --model_path /path/to/model --tasks boolq hellaswag arc_challenge

  # With adapter
  python scripts/eval_benchmarks.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --adapter_path outputs/qlora-4bit-sqat-adapter

  # Compare two models side by side
  python scripts/eval_benchmarks.py \
    --model_path outputs/qlora-merged-noquant outputs/qlora-4bit-none-dequant \
    --output_dir results/comparison
"""

import os
import json
import argparse
from datetime import datetime

DEFAULT_TASKS = [
    "boolq",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "piqa",
    # "social_iqa",
    "openbookqa",
]

# lm-eval-harness task name mapping (some tasks use different registry names)
TASK_REGISTRY_MAP = {
    "boolq":         "boolq",
    "hellaswag":     "hellaswag",
    "winogrande":    "winogrande",
    "arc_easy":      "arc_easy",
    "arc_challenge": "arc_challenge",
    "piqa":          "piqa",
    # "social_iqa":    "social_iqa",
    "openbookqa":    "openbookqa",
    "mmlu":          "mmlu",
}

# Primary accuracy metric per task (for the summary table)
TASK_PRIMARY_METRIC = {
    "boolq":         "acc",
    "hellaswag":     "acc_norm",
    "winogrande":    "acc",
    "arc_easy":      "acc_norm",
    "arc_challenge": "acc_norm",
    "piqa":          "acc_norm",
    # "social_iqa":    "acc",
    "openbookqa":    "acc_norm",
    "mmlu":          "acc",
}


def resolve_task_names(tasks):
    """Map user-facing names to lm-eval-harness registry names."""
    resolved = []
    for t in tasks:
        t_lower = t.lower().strip()
        resolved.append(TASK_REGISTRY_MAP.get(t_lower, t_lower))
    return resolved


def eval_benchmarks(
    model_path: str,
    tasks: list = None,
    num_fewshot: int = 0,
    batch_size: int = 8,
    output_dir: str = "results",
    adapter_path: str = None,
    dtype: str = "float16",
):
    """
    Run evaluation using lm-eval-harness with HuggingFace backend.

    Returns:
        dict with "results", "summary", and "config" keys.
    """
    import lm_eval

    if tasks is None:
        tasks = list(DEFAULT_TASKS)

    resolved_tasks = resolve_task_names(tasks)

    print(f"[Eval] Model: {model_path}")
    if adapter_path:
        print(f"[Eval] Adapter: {adapter_path}")
    print(f"[Eval] Tasks ({len(resolved_tasks)}): {', '.join(resolved_tasks)}")
    print(f"[Eval] Num fewshot: {num_fewshot}")
    print(f"[Eval] Batch size: {batch_size}")
    print(f"[Eval] Dtype: {dtype}")

    # Build model args
    model_args = f"pretrained={model_path}"
    model_args += f",dtype={dtype}"
    model_args += ",trust_remote_code=True"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=resolved_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    # Build summary table
    summary = {}
    raw_results = results.get("results", {})
    for task_name, task_results in raw_results.items():
        # Strip group prefix if present (e.g. "harness|arc_easy|0" -> "arc_easy")
        clean_name = task_name
        for known in TASK_PRIMARY_METRIC:
            if known in task_name:
                clean_name = known
                break

        primary_metric = TASK_PRIMARY_METRIC.get(clean_name, "acc")

        # lm-eval-harness may suffix with ",none" or similar
        score = None
        for key in [primary_metric, f"{primary_metric},none", f"{primary_metric},0"]:
            if key in task_results:
                score = task_results[key]
                break

        # Fallback: grab first numeric value
        if score is None:
            for v in task_results.values():
                if isinstance(v, (int, float)):
                    score = v
                    break

        if score is not None:
            summary[clean_name] = {
                "score": score,
                "metric": primary_metric,
            }

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Detailed per-task
    for task_name, task_results in raw_results.items():
        print(f"\n  {task_name}:")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")

    # Summary table
    print("\n" + "-" * 70)
    print(f"  {'Task':<20} {'Metric':<12} {'Score':>8}")
    print("  " + "-" * 42)
    scores = []
    for task_name in (DEFAULT_TASKS + ["mmlu"]):
        if task_name not in summary:
            continue
        s = summary[task_name]
        print(f"  {task_name:<20} {s['metric']:<12} {s['score']:>8.4f}")
        scores.append(s["score"])

    if scores:
        avg = sum(scores) / len(scores)
        print("  " + "-" * 42)
        print(f"  {'AVERAGE':<20} {'':12} {avg:>8.4f}")
    print("-" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path.rstrip("/"))
    result_file = os.path.join(output_dir, f"{model_name}_bench_{timestamp}.json")

    output = {
        "config": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "dtype": dtype,
        },
        "summary": summary,
        "results": raw_results,
        "versions": results.get("versions", {}),
        "timestamp": timestamp,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[Eval] Results saved to {result_file}")
    return output


def compare_results(result_files: list):
    """Load and print a side-by-side comparison of multiple result files."""
    all_results = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(path)
            all_results.append(data)

    if not all_results:
        return

    # Collect all tasks
    all_tasks = []
    for r in all_results:
        for t in r.get("summary", {}):
            if t not in all_tasks:
                all_tasks.append(t)

    # Header
    names = [r["config"].get("model_path", r["_file"]) for r in all_results]
    short_names = [os.path.basename(n.rstrip("/"))[:25] for n in names]

    header = f"  {'Task':<20}" + "".join(f" {n:>25}" for n in short_names)
    print("\n" + "=" * len(header))
    print("COMPARISON")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))

    avgs = [[] for _ in all_results]
    for task in all_tasks:
        row = f"  {task:<20}"
        for i, r in enumerate(all_results):
            s = r.get("summary", {}).get(task, {}).get("score")
            if s is not None:
                row += f" {s:>25.4f}"
                avgs[i].append(s)
            else:
                row += f" {'—':>25}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    row = f"  {'AVERAGE':<20}"
    for avg_list in avgs:
        if avg_list:
            row += f" {sum(avg_list)/len(avg_list):>25.4f}"
        else:
            row += f" {'—':>25}"
    print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-benchmark 0-shot Evaluation (HF + lm-eval-harness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- eval subcommand (default) ---
    p_eval = sub.add_parser("eval", help="Run benchmarks")
    p_eval.add_argument("--model_path", type=str, required=True)
    p_eval.add_argument("--adapter_path", type=str, default=None)
    p_eval.add_argument("--tasks", type=str, nargs="+", default=None,
                        help=f"Tasks to evaluate (default: all 8 benchmarks)")
    p_eval.add_argument("--num_fewshot", type=int, default=0)
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    p_eval.add_argument("--output_dir", type=str, default="results")

    # --- compare subcommand ---
    p_cmp = sub.add_parser("compare", help="Compare saved result files")
    p_cmp.add_argument("result_files", nargs="+", help="JSON result files to compare")

    args = parser.parse_args()

    # Default to eval if no subcommand
    if args.command is None:
        parser.print_help()
        print("\nQuick start:")
        print("  python scripts/eval_benchmarks.py eval --model_path /path/to/model")
        return

    if args.command == "eval":
        eval_benchmarks(
            model_path=args.model_path,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            adapter_path=args.adapter_path,
            dtype=args.dtype,
        )
    elif args.command == "compare":
        compare_results(args.result_files)


if __name__ == "__main__":
    main()
