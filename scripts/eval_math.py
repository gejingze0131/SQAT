#!/usr/bin/env python
"""
Math evaluation via HuggingFace backend + lm-eval-harness.

Default tasks:
  - GSM8K        -> gsm8k
  - MATH500      -> hendrycks_math500

Examples:
  # Evaluate merged / quantized model
  python scripts/eval_math.py --model_path outputs/qlora-4bit-none-merged

  # Evaluate base model + adapter
  python scripts/eval_math.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --adapter_path outputs/metamath-adapter

  # Evaluate only GSM8K
  python scripts/eval_math.py --model_path /path/to/model --tasks gsm8k

  # Custom few-shot
  python scripts/eval_math.py \
    --model_path /path/to/model \
    --tasks gsm8k hendrycks_math500 \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_dir results_math
"""

import os
import json
import argparse
from datetime import datetime


DEFAULT_TASKS = ["gsm8k"]


def _format_metric_value(value):
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def eval_with_lm_eval_harness(
    model_path: str,
    tasks: list = None,
    num_fewshot: int = 0,
    batch_size: int = 8,
    output_dir: str = "results_math",
    adapter_path: str = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    log_samples: bool = False,
    limit: int = None,
):
    """
    Run evaluation using lm-eval-harness with HuggingFace backend.

    log_samples/limit exist for DIAGNOSIS, not for reporting. gsm8k is scored two ways and they
    normally agree to within ~0.3 points, because flexible-extract (take the last number) is a
    superset of strict-match (require the literal "#### <n>" format). When flexible comes in
    BELOW strict the model is answering in the right format and then continuing to generate,
    so the last number is no longer the answer — a decoding-style artifact rather than a
    capability difference. Distinguishing the two requires reading generations, which is what
    log_samples dumps. Anything reported from a `limit`ed run is not comparable to a full run.
    """
    import lm_eval

    if tasks is None:
        tasks = DEFAULT_TASKS

    print(f"[Eval] Model: {model_path}")
    if adapter_path:
        print(f"[Eval] Adapter: {adapter_path}")
    print(f"[Eval] Tasks: {tasks}")
    print(f"[Eval] Num fewshot: {num_fewshot}")
    print(f"[Eval] Batch size: {batch_size}")
    print(f"[Eval] apply_chat_template: {apply_chat_template}")
    print(f"[Eval] fewshot_as_multiturn: {fewshot_as_multiturn}")

    # SQAT-Permute exports need the runtime boundary gather at inference; only then pull in
    # sqat_permute (which depends on bitsandbytes). Plain models keep the string form.
    if os.path.exists(os.path.join(model_path, "sqat_permute_meta.pt")):
        import sys as _sys
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from src.qat_permute_sqat import lm_eval_model_kwargs
        _me = lm_eval_model_kwargs(
            model_path, dtype="float16", batch_size=batch_size, adapter_path=adapter_path,
        )
    else:
        _margs = f"pretrained={model_path},dtype=float16,trust_remote_code=True"
        if adapter_path:
            _margs += f",peft={adapter_path}"
        _me = {"model": "hf", "model_args": _margs}

    # Run evaluation
    results = lm_eval.simple_evaluate(
        **_me,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        log_samples=log_samples,
        **({"limit": limit} if limit else {}),
    )

    # Under data-parallel eval (accelerate launch --num_processes N), simple_evaluate
    # aggregates onto the main process and returns None on the others — nothing to save there.
    if results is None:
        return None

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n  {task_name}:")
            for metric, value in task_results.items():
                print(f"    {metric}: {_format_metric_value(value)}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path.rstrip("/"))
    result_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")

    serializable = {
        "config": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "apply_chat_template": apply_chat_template,
            "fewshot_as_multiturn": fewshot_as_multiturn,
        },
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
        "configs": results.get("configs", {}),
        "timestamp": timestamp,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Eval] Results saved to {result_file}")

    # Per-sample generations go to a SEPARATE file on purpose: collect_saltq_results.py parses the
    # results JSON above, and burying megabytes of generations in it would both slow that down and
    # risk changing the shape it expects.
    if log_samples and results.get("samples"):
        samples_file = result_file.replace(".json", "_samples.json")
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(results["samples"], f, indent=1, ensure_ascii=False, default=str)
        n = sum(len(v) for v in results["samples"].values())
        print(f"[Eval] {n} per-sample records saved to {samples_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Math Evaluation (HF backend + lm-eval-harness)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to quantized/merged model or HF model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to PEFT/LoRA adapter (optional)")
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        help="Evaluation tasks (default: gsm8k hendrycks_math500)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_math")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Apply tokenizer chat template during eval. "
                             "Useful for instruct/chat models when the task prompt format matches.")
    parser.add_argument("--log_samples", action="store_true",
                        help="Dump per-sample generations alongside the scores (diagnosis only).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N docs (diagnosis only; NOT comparable to "
                             "a full run).")
    parser.add_argument("--fewshot_as_multiturn", action="store_true",
                        help="Format few-shot examples as multi-turn chat when chat template is enabled.")

    args = parser.parse_args()

    eval_with_lm_eval_harness(
        model_path=args.model_path,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        log_samples=args.log_samples,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
