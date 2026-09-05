#!/usr/bin/env python
"""
Build datasets/alpaca/train.json — the Alpaca column's fine-tuning set.

datasets/ is gitignored, so this is the tracked record of how that column's data was produced.
Run it ONCE on a LOGIN node: compute nodes have no route out, and a hub snapshot is not enough
for load_dataset under HF_DATASETS_OFFLINE.

    python scripts/build_alpaca_dataset.py

SOURCE: yahma/alpaca-cleaned (51,760 records) — the de-duplicated, error-corrected rewrite of
Stanford Alpaca that QLoRA, QA-LoRA and QEFT all fine-tune on. NOT tatsu-lab/alpaca (52,002
records), whose known-bad outputs make its numbers a different cell.

THE `input` FIELD. 40% of Alpaca records carry a second context field, and src/data.py's PROMPT
has exactly one slot ({instruction}) — by design: a single template is what keeps train and test
prompts byte-identical across every dataset in this repo. Rather than fork the template, the
context is FOLDED into the instruction here, at build time, as

    instruction + "\\n\\n" + input

and `input` is written back empty, so the file satisfies src/data._to_instruction_output's
"no non-empty input" check and every downstream stage (tokenizer, collator, calibration) sees
one flat instruction. The alternative — Alpaca's own two-template prompt_input/prompt_no_input
pair — would put a "### Input:" section in 40% of the prompts and nowhere else in the repo.

Every record also gets `type: "alpaca"`, the column name the pissa-dataset schema uses (and
which src/data._select_sub_tasks / calibration_sampling=balanced read). Alpaca is a single
undifferentiated task, so there is exactly one type.

NO test split. This column is scored on MMLU (lm-eval, 5-shot) — an external benchmark the model
never trains on — so there is nothing to hold out; runs/eval_vllm.sh dispatches --dataset alpaca
to scripts/eval_mmlu.py instead of to a generative test split.
"""

import argparse
import json
import os

HF_NAME = "yahma/alpaca-cleaned"
EXPECTED_RECORDS = 51760


def build_alpaca(out_dir: str, name: str) -> None:
    from datasets import load_dataset

    ds = load_dataset(name, split="train")
    os.makedirs(out_dir, exist_ok=True)

    records, folded = [], 0
    for rec in ds:
        instruction = (rec.get("instruction") or "").strip()
        context = (rec.get("input") or "").strip()
        if context:
            instruction = f"{instruction}\n\n{context}"
            folded += 1
        records.append({
            "instruction": instruction,
            "input": "",
            "output": rec.get("output") or "",
            "type": "alpaca",
        })

    path = os.path.join(out_dir, "train.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"[build] {name}: {len(records)} records ({folded} with a folded `input`, "
          f"{100.0 * folded / max(len(records), 1):.1f}%) -> {path}")
    if len(records) != EXPECTED_RECORDS:
        print(f"[build] WARNING: expected {EXPECTED_RECORDS} records, got {len(records)}. The "
              f"hub dataset changed; every number produced from this file is a different cell.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out_dir", default="datasets/alpaca")
    ap.add_argument("--name", default=HF_NAME)
    args = ap.parse_args()

    # This script is the thing that needs the network; refuse to "succeed" from a stale cache
    # silently if someone exported the offline flags in their shell.
    for var in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.pop(var, None):
            print(f"[build] unset {var} (this script must reach the hub)")

    build_alpaca(args.out_dir, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
