"""
Build the plain-GPTQ base QWHA fine-tunes on top of, one (bits, group_size) cell per run.

This is upstream's own GPTQ branch (`utils.get_quantized_peft_model`, `GPTQConfig(sym=False,
dataset="wikitext2")` -> optimum -> gptqmodel, 128 x 2048 calibration sequences, desc_act off),
lifted out of the training path so the quantization happens ONCE instead of per rank, and so the
cache directory carries the group size.

Deliberately NOT MagR. Upstream's README quantizes with MagR + GPTQ; this repo's INT3 g64 / INT2
g32 rows (GPTQ floor, QA-LoRA, SALT-Q) all sit on a plain GPTQ base, and a baseline on a stronger
base measures the base, not the adapter.

NOT USED BY THE REPORTED ROWS. `dataset="wikitext2"` is generic-text calibration, which is what
this repo measured as the thing that breaks low-bit bases here: the same merged checkpoint scored
INT2 36.64 on the old first-N (BoolQ-only, 9.5k token) set and 66.22 on the task-balanced
3500-record in-domain set, and C4 128x2048 loses the instruction template outright (30.67). The
bcal rows are built by make_bcal_base.py instead. This script is kept as the upstream-faithful
path -- for provenance, and for a calibration ablation on an otherwise identical pipeline.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from qwha_common import gptq_base_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_id", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("-b", "--bits", type=int, required=True)
    ap.add_argument("-g", "--group_size", type=int, required=True)
    ap.add_argument("--dataset", default="wikitext2")
    args = ap.parse_args()

    out = gptq_base_dir(args.model_id, args.bits, args.group_size)
    if os.path.isdir(out):
        print(f"[GPTQ] {out} already exists -- nothing to do.")
        return

    print(f"[GPTQ] INT{args.bits} g{args.group_size} asym, calib={args.dataset} -> {out}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=GPTQConfig(
            bits=args.bits,
            sym=False,
            dataset=args.dataset,
            tokenizer=tokenizer,
            group_size=args.group_size,
        ),
        device_map="cuda",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)      # so the base dir is loadable on its own
    print(f"[GPTQ] saved {out}")
    print(f"[GPTQ] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
