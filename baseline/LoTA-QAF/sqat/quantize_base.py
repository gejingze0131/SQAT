"""Build the GPTQ base LoTA-QAF fine-tunes, with the paper's own quantization recipe.

This is upstream `gptq_quantize.py` with the hardcoded paths turned into flags. Nothing about
the recipe is changed: GPTQModel's own GPTQ, asymmetric, desc_act (act-order) on, damp 0.01,
and 1024 C4 sequences for calibration -- section 4.1 of arXiv:2505.18724.

Deliberately NOT matched to this repo's own GPTQ bases (src/gptq.py, calibrated on the
training data, desc_act off). LoTA-QAF's ternary adaptation is defined ON the GPTQ integer
grid it is given: it moves each weight by at most +-1 grid step, so the base is part of the
method, not a shared substrate. Re-quantizing it our way would be reporting a different
method. The price is that its floor is not our GPTQ floor row -- so run the base through the
same evaluation on its own (--eval the quantized dir) and report that as LoTA-QAF's floor.

    python quantize_base.py --bits 3 --group_size 64 --out outputs/lota_bases
      -> outputs/lota_bases/Llama-2-7B_int3_64_asym
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lota_common import resolve_pretrained  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 8])
    ap.add_argument("--group_size", type=int, required=True)
    ap.add_argument("--sym", action="store_true", help="default is asymmetric, as in the paper")
    ap.add_argument("--out", required=True, help="parent dir; the int{b}_{g}_asym dir is created under it")
    ap.add_argument("--nsamples", type=int, default=1024)
    ap.add_argument("--calib_batch_size", type=int, default=1)
    # Where GPTQModel caches the per-layer calibration activations. 1024 C4 sequences of
    # a few hundred tokens is several GB of hidden states on top of the 13 GB bf16 model;
    # keeping them on the host is slower but numerically identical, and a 12-hour job that
    # OOMs in hour three costs a day of queue.
    ap.add_argument("--cpu_cache", action="store_true")
    # "Llama-2-7B" here is not cosmetic: upstream LoTA_QAF_main.py greps (\d+B) out of the
    # quantized dir name to pick the adapter rank.
    ap.add_argument("--tag", default="Llama-2-7B")
    args = ap.parse_args()

    save_dir = os.path.join(
        args.out, f"{args.tag}_int{args.bits}_{args.group_size}_{'sym' if args.sym else 'asym'}"
    )
    if os.path.isfile(os.path.join(save_dir, "quantize_config.json")):
        print(f"[quantize_base] {save_dir} already quantized; nothing to do.")
        return

    quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size, sym=args.sym)
    print(f"[quantize_base] {args.pretrained} -> {save_dir}")
    print(f"[quantize_base] {quant_config}")

    model = GPTQModel.load(
        resolve_pretrained(args.pretrained),
        quantize_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(args.nsamples))["text"]

    model.quantize(calibration_dataset, batch_size=args.calib_batch_size,
                   calibration_enable_gpu_cache=not args.cpu_cache)
    model.save(save_dir)
    print(f"[quantize_base] saved {save_dir}")


if __name__ == "__main__":
    main()
