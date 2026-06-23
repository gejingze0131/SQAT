#!/usr/bin/env python
"""
MMLU 0-shot evaluation via HuggingFace backend + lm-eval-harness.

Usage:
  # Evaluate GPTQ quantized model
  python scripts/eval_mmlu.py --model_path outputs/qlora-4bit-none-merged

  # Evaluate with adapter
  python scripts/eval_mmlu.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --adapter_path outputs/qlora-4bit-sqat-adapter

  # Custom batch size and output
  python scripts/eval_mmlu.py --model_path /path/to/model --batch_size 16 --output_dir results/
"""

import os
import json
import argparse
from datetime import datetime


def eval_with_lm_eval_harness(
    model_path: str,
    tasks: list = None,
    num_fewshot: int = 0,
    batch_size: int = 8,
    output_dir: str = "results",
    adapter_path: str = None,
):
    """
    Run evaluation using lm-eval-harness with HuggingFace backend.
    """
    import lm_eval

    if tasks is None:
        tasks = ["mmlu"]

    print(f"[Eval] Model: {model_path}")
    if adapter_path:
        print(f"[Eval] Adapter: {adapter_path}")
    print(f"[Eval] Tasks: {tasks}")
    print(f"[Eval] Num fewshot: {num_fewshot}")
    print(f"[Eval] Batch size: {batch_size}")

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
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path.rstrip("/"))
    result_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")

    with open(result_file, "w") as f:
        serializable = {
            "config": {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "tasks": tasks,
                "num_fewshot": num_fewshot,
                "batch_size": batch_size,
            },
            "results": results.get("results", {}),
            "versions": results.get("versions", {}),
            "timestamp": timestamp,
        }
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n[Eval] Results saved to {result_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MMLU 0-shot Evaluation (HF backend)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to quantized/merged model or HF model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to PEFT/LoRA adapter (optional)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu"],
                        help="Evaluation tasks (default: mmlu)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    eval_with_lm_eval_harness(
        model_path=args.model_path,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
    )


if __name__ == "__main__":
    main()