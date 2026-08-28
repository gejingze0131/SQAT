#!/usr/bin/env python
"""
GPTQ quantize->dequantize any dense fp16 checkpoint, in place of the RTN the merged-model export
does.

WHY THIS EXISTS. runs/qlora/_pipeline.sh exports two models: a merged fp16 upper bound and a
"dequant" arm meant to be the realistic deployment number. That second arm is plain
round-to-nearest -- src/export.py's GPTQ state is sqat_permute-only and silently drops
qat.sqat_permute.gptq.enabled for every other qat_mode -- and RTN at INT3 destroys Llama-2-7b.
The QLoRA arm came back emitting "riiriirii..." to the token cap for all 22,419 prompts and
scored 0.00 on all eight tasks, which is not a deployment bound, it is a broken file.

That made the only usable QLoRA number its FP16 merge (82.72), i.e. an upper bound with no
quantization in it at all, so nothing in the table answered "how do these methods compare when
BOTH are actually deployed at 3 bits".

This script closes that. It reuses gptq_quantize_model_sequential exactly as
qalora.build_qalora_intb_base does -- perm_group_k=0 and perm_meta=None, meaning every column of
every target linear is GPTQ'd with no salient slice and no boundary gather -- so the baseline is
quantized by the same error-compensated code path SALT-Q's frozen base is built with. Comparing
a GPTQ method against an RTN baseline would have flattered SALT-Q for a reason that has nothing
to do with SALT-Q.

The input is a MERGED dense checkpoint (LoRA already folded in). It must not be a SALT-Q or
SQAT-Permute export: those carry sqat_permute_meta.pt and need boundary gathers, and are refused.

  python scripts/export_gptq_dequant.py \
      --model_path outputs/qlora_none_cs170k_int3_ep1-3bit-none-merged-eval \
      --output_dir outputs/qlora_none_cs170k_int3_ep1-3bit-none-gptq-eval \
      --config     configs/qlora_none_cs170k_int3_g64_ep1.yaml
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from transformers import AutoModelForCausalLM

from src.data import build_data_collator, load_calibration_data
from src.model_loader import load_tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="merged dense fp16 checkpoint")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", required=True,
                    help="supplies the calibration data, target modules, bits and group_size")
    ap.add_argument("--bits", type=int, default=None, help="override cfg.model.quant_bits")
    ap.add_argument("--group_size", type=int, default=None, help="override cfg.qat.group_size")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--calibration_samples", type=int, default=None,
                    help="override cfg.qat.sqat.calibration_samples (records loaded); raise it "
                         "together with --nsamples for a larger Hessian budget")
    ap.add_argument("--calibration_sampling", choices=["first", "shuffle", "balanced"], default=None,
                    help="override cfg.qat.sqat.calibration_sampling (see src/data.py)")
    ap.add_argument("--calibration_source", default=None,
                    help="JSON list of raw texts (e.g. datasets/c4_calib_1024.json): generic-text "
                         "calibration in calibration_seq_len windows, the GPTQ-paper recipe")
    ap.add_argument("--calibration_seq_len", type=int, default=None)
    args = ap.parse_args()

    meta_pt = os.path.join(args.model_path, "sqat_permute_meta.pt")
    if os.path.exists(meta_pt):
        print(f"ERROR: {args.model_path} carries sqat_permute_meta.pt, so its residual stream is "
              f"permuted and a correct forward pass needs a boundary gather per segment. GPTQ "
              f"here would quantize a model this script cannot even evaluate. Fold it first with "
              f"scripts/export_vllm_ready.py, or quantize the pre-export checkpoint.",
              file=sys.stderr)
        return 2

    cfg = yaml.safe_load(open(args.config))
    if args.calibration_samples is not None:
        cfg["qat"]["sqat"]["calibration_samples"] = int(args.calibration_samples)
    if args.calibration_sampling is not None:
        cfg["qat"]["sqat"]["calibration_sampling"] = args.calibration_sampling
    if args.calibration_source is not None:
        cfg["qat"]["sqat"]["calibration_source"] = args.calibration_source
    if args.calibration_seq_len is not None:
        cfg["qat"]["sqat"]["calibration_seq_len"] = int(args.calibration_seq_len)
    bits = args.bits or int(cfg["model"]["quant_bits"])
    group_size = args.group_size or int(cfg["qat"].get("group_size", 128))
    symmetric = bool(cfg["qat"].get("symmetric", False))
    targets = list(cfg["lora"]["target_modules"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg["model"]["dtype"])

    print(f"[GPTQ-export] {args.model_path}")
    print(f"[GPTQ-export] INT{bits} g{group_size} {'sym' if symmetric else 'asym'}, "
          f"targets={targets}, nsamples={args.nsamples}")

    tokenizer = load_tokenizer(cfg, name=args.model_path)
    cal = load_calibration_data(cfg, tokenizer)
    cal_loader = DataLoader(cal, batch_size=args.batch_size,
                            collate_fn=build_data_collator(tokenizer), shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model.to(device).eval()

    ref = {n: p.detach().float().cpu().clone()
           for n, p in model.named_parameters()
           if n.endswith(".weight") and n.split(".")[-2] in targets}

    from src.gptq import gptq_quantize_model_sequential
    gptq_quantize_model_sequential(
        model, cal_loader, targets,
        perm_group_k=0,          # no salient slice: every column is GPTQ'd
        group_size=group_size,
        q_bits=bits,
        symmetric=symmetric,
        device=device,
        perm_meta=None,          # no permutation, no boundary gathers
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        nsamples=args.nsamples,
    )

    # The number that says whether this ran at all. RTN and GPTQ both "succeed"; only the error
    # separates them, and a near-zero error would mean the weights were never quantized.
    errs = []
    for n, p in model.named_parameters():
        if n in ref:
            r = ref[n]
            errs.append((r - p.detach().float().cpu()).norm().item() / max(r.norm().item(), 1e-12))
    errs.sort()
    print(f"[GPTQ-export] relative weight error over {len(errs)} projections: "
          f"p10={errs[len(errs)//10]:.4f} p50={errs[len(errs)//2]:.4f} "
          f"p90={errs[9*len(errs)//10]:.4f}")
    if errs[len(errs)//2] < 1e-3:
        raise RuntimeError("median relative error is ~0 — the weights were not quantized")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    torch.save({"q_bits": bits, "group_size": group_size, "symmetric": symmetric,
                "source_model": os.path.abspath(args.model_path), "quantizer": "gptq",
                "perm_group_k": 0},
               os.path.join(args.output_dir, "gptq_dequant_meta.pt"))
    print(f"[GPTQ-export] saved -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
