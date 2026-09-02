#!/usr/bin/env python
"""
Build datasets/wikitext2/{train,validation,test}.json and datasets/c4_val_ppl.json.

datasets/ is gitignored, so this is the tracked record of how the WikiText-2 column's data was
produced. Run it ONCE on a LOGIN node: compute nodes have no route out, and a hub snapshot is
not enough for load_dataset under HF_DATASETS_OFFLINE — the dataset has to be built first.

    python scripts/build_wikitext2_dataset.py

WIKITEXT-2-RAW-V1, NOT WIKITEXT-2-V1. The processed variant replaces rare words with <unk>,
which lowers perplexity by a wide margin; numbers across the two are not comparable. Every
WikiText-2 perplexity in the GPTQ / AWQ / OmniQuant / LoftQ / ApiQ line is the raw variant.

Rows are stored as {"text": ...} in file order, unmodified — the joining, tokenizing and
windowing all happen in src/data.py's `task_type: lm` path, so training, calibration and
perplexity evaluation cut the same stream the same way. Expect, with the Llama-2 tokenizer:
    train 2,874,559 tokens -> 2807 blocks of 1024
    test    341,469 tokens ->  166 windows of 2048   (the GPTQ lineage's standard count)

datasets/c4_val_ppl.json is the C4 control for the calibration ablation (T8): 2000 texts from
the C4 en VALIDATION shard, ~1.07M Llama-2 tokens. It is disjoint BY CONSTRUCTION from
datasets/c4_calib_1024.json, which baseline/LoTA-QAF/sqat/quantize_base.py takes from the
en/c4-train.00001-of-01024 shard. Stored as a bare list of strings, the same shape as that file.
"""

import argparse
import json
import os
import sys

WIKITEXT_SPLITS = ("train", "validation", "test")


def build_wikitext2(out_dir: str) -> None:
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    for split in WIKITEXT_SPLITS:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        path = os.path.join(out_dir, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"text": t} for t in ds["text"]], f)
        print(f"[build] wikitext-2-raw-v1 {split}: {len(ds)} rows -> {path}")


def build_c4_val(out_path: str, n_texts: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", data_files="en/c4-validation.00000-of-00008.json.gz",
                      split="train", streaming=True)
    texts = []
    for rec in ds:
        texts.append(rec["text"])
        if len(texts) >= n_texts:
            break
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(texts, f)
    print(f"[build] C4 en validation: {len(texts)} texts, "
          f"{sum(len(t) for t in texts)} chars -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out_dir", default="datasets/wikitext2")
    ap.add_argument("--c4_val_out", default="datasets/c4_val_ppl.json")
    ap.add_argument("--c4_val_texts", type=int, default=2000)
    ap.add_argument("--skip_c4", action="store_true")
    args = ap.parse_args()

    # This script is the thing that needs the network; refuse to "succeed" from a stale cache
    # silently if someone exported the offline flags in their shell.
    for var in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.pop(var, None):
            print(f"[build] unset {var} (this script must reach the hub)")

    build_wikitext2(args.out_dir)
    if not args.skip_c4:
        build_c4_val(args.c4_val_out, args.c4_val_texts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
