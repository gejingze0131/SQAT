#!/usr/bin/env python
"""
Generative evaluation, stage 1 of 2: sample answers with vLLM.

This is the migration/gen_vllm.py reference adapted to the local dataset layout. It runs in
the `vllm` conda env, NOT the training env — see runs/eval_vllm.sh.

The test splits under datasets/<name>/test.json ship their `instruction` field ALREADY
wrapped in src/data.PROMPT, so the prompt is fed to the model verbatim. Do not re-wrap it:
double-wrapping is invisible in the output file and costs several points.

Writes one JSON object per line: {type, query, output, answer}. scripts/test_acc.py scores it.

  python scripts/gen_vllm.py \
      --model outputs/saltq-3bit-saltq-deploy-vllm \
      --data_path datasets/commonsense \
      --output_file results/commonsense/saltq_int3.jsonl
"""

import argparse
import json
import os
import sys

import torch


# The fixed head of src/data.PROMPT. Duplicated as a literal rather than imported because this
# script runs in the vllm env, which does not have the training package on its path. The test
# splits ship their `instruction` already wrapped in it; if that ever stops being true the model
# is being prompted differently than it was trained, which shows up only as a lower score.
PROMPT_PREFIX = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n### Instruction:\n"
)


def load_test_records(data_path: str, split: str, sub_task=None):
    """Load <data_path>/<split>.json (or a direct .json file) and optionally filter by type."""
    path = data_path
    if os.path.isdir(data_path):
        path = os.path.join(data_path, f"{split}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No test file at {path}")

    with open(path) as f:
        records = json.load(f)

    if sub_task:
        wanted = set(sub_task)
        available = sorted({r.get("type") for r in records})
        records = [r for r in records if r.get("type") in wanted]
        if not records:
            raise ValueError(f"sub_task {sorted(wanted)} matched 0 records; available: {available}")

    if not records[0]["instruction"].startswith(PROMPT_PREFIX):
        raise ValueError(
            f"{path} records are not pre-wrapped in the training prompt "
            f"(expected instruction to start with {PROMPT_PREFIX!r}). Feeding a bare question "
            f"to a model fine-tuned on the wrapped one silently costs several points."
        )

    counts = {}
    for r in records:
        counts[r.get("type")] = counts.get(r.get("type"), 0) + 1
    print(f"[gen] {path}: {len(records)} records")
    for task, n in sorted(counts.items()):
        print(f"[gen]   {task}: {n}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF model dir. For SALT-Q / SQAT-Permute this MUST be the "
                             "-deploy-vllm dir produced by scripts/export_vllm_ready.py: the "
                             "-deploy-eval dir still needs runtime boundary gathers that vLLM "
                             "cannot register.")
    parser.add_argument("--data_path", type=str, default="datasets/commonsense")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--sub_task", nargs="+", default=None,
                        help="Restrict to these `type` values (e.g. boolq piqa).")
    parser.add_argument("--output_file", type=str, default="model_response.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Default: every visible GPU.")
    args = parser.parse_args()

    meta_path = os.path.join(args.model, "sqat_permute_meta.pt")
    if os.path.exists(meta_path):
        print(
            f"ERROR: {args.model} contains sqat_permute_meta.pt, which means its residual stream "
            f"still needs a boundary gather at every segment boundary. vLLM cannot register that "
            f"hook and would score a silently wrong model.\n"
            f"Fold it first:\n"
            f"  python scripts/export_vllm_ready.py --model_path {args.model} "
            f"--output_dir {args.model.rstrip('/')}-vllm",
            file=sys.stderr,
        )
        return 2

    records = load_test_records(args.data_path, args.dataset_split, args.sub_task)

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=[],
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )

    prompts = [r["instruction"] for r in records]
    with torch.no_grad():
        completions = llm.generate(prompts, sampling_params)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    # Truncate rather than append: the reference script appends, so re-running an eval silently
    # doubled every task's sample count and averaged the two runs together.
    with open(args.output_file, "w") as f:
        for record, completion in zip(records, completions):
            json.dump(
                {
                    "type": record.get("type"),
                    "query": record["instruction"],
                    "output": completion.outputs[0].text,
                    "answer": record["output"],
                },
                f,
            )
            f.write("\n")

    print(f"[gen] wrote {len(records)} responses -> {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
