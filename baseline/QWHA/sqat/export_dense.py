"""
Export a trained QWHA model to a dense fp16 Llama checkpoint that vLLM can serve.

A QWHA layer computes   y = WHT(x) @ (WHT(W_q) + S)^T / in_features,   with S the sparse trained
spectrum. WHT is its own inverse up to the 1/n it already carries, so the whole layer is exactly
one dense linear map,

    W_eff = iWHT( WHT(dequant(W_q)) + S ),

and the export materializes it per layer. That is the same seam every other row in the tables
goes through (runs/eval_vllm.sh reads a dense fp16 checkpoint, exactly as it does for the SALT-Q
and QA-LoRA exports); it moves no accuracy, and the identity is CHECKED here rather than assumed:
each layer's dense weight is compared against the live QWHA module on random inputs, and the run
aborts if the two disagree beyond fp32 noise.

What the dense checkpoint does NOT do is make the model 3-bit. The deployed QWHA model is INT-b
codes plus an unmergeable fp16 spectrum; the dense file is a numerical stand-in for evaluation.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# qwha_common first: importing it is what puts upstream's src/ and src/init/ on sys.path.
from qwha_common import build_qwha_model, gptq_base_dir, load_qwha_adapter
from hadamard import iwht, wht          # upstream src/init/hadamard.py


@torch.no_grad()
def dense_weight(module) -> torch.Tensor:
    """W_eff for one QWHA-adapted layer, in fp32, shape [out, in]."""
    base_w = wht(module.base_layer.dequantize_weight().T.to(torch.float32))
    delta = module.get_delta_weight("default").to(torch.float32).to_dense()
    return iwht(base_w + delta)


@torch.no_grad()
def check_layer(module, w_eff: torch.Tensor, n: int = 4) -> float:
    """max |QWHA forward - dense forward| on random inputs, relative to the output scale."""
    x = torch.randn(n, module.in_features, device=w_eff.device, dtype=torch.float32)
    y_ref = module(x)
    y_dense = torch.nn.functional.linear(x, w_eff)
    return (y_ref - y_dense).abs().max().item() / y_ref.abs().max().clamp(min=1e-6).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="trained QWHA output_dir")
    ap.add_argument("--out", required=True, help="dense fp16 checkpoint directory")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max allowed relative error between the QWHA layer and its dense form")
    args = ap.parse_args()

    with open(os.path.join(args.adapter_dir, "qwha_run_meta.json")) as f:
        meta = json.load(f)
    model_id, bits, gs = meta["model_id"], meta["bits"], meta["group_size"]
    print(f"[export] {args.adapter_dir}: INT{bits} g{gs} rank{meta['rank']} scale{meta['scale']}")

    # fp32 so the identity check measures the export, not bf16 rounding.
    qwha = build_qwha_model(gptq_base_dir(model_id, bits, gs), rank=meta["rank"],
                            scale=meta["scale"], device="cuda", dtype=torch.float32)
    load_qwha_adapter(qwha, args.adapter_dir, scale=meta["scale"])
    qwha.eval()

    dense = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                 device_map="cpu")
    target = dict(dense.named_modules())

    worst, n_layers = 0.0, 0
    layers = [(n, m) for n, m in qwha.named_modules() if hasattr(m, "qwha_spectrum")]
    for name, module in tqdm(layers, "Materializing"):
        w_eff = dense_weight(module)
        err = check_layer(module, w_eff)
        worst = max(worst, err)
        if err > args.tol:
            raise RuntimeError(f"{name}: dense form differs from the QWHA layer by {err:.2e}")
        key = name.replace("base_model.model.", "", 1)
        if key not in target:
            raise KeyError(f"{key} not found in the fp16 model")
        target[key].weight.data.copy_(w_eff.to(torch.float16).cpu())
        n_layers += 1
        del w_eff
        torch.cuda.empty_cache()

    print(f"[export] {n_layers} layers; worst relative error {worst:.2e} (tol {args.tol:.0e})")

    os.makedirs(args.out, exist_ok=True)
    dense.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(args.out)
    with open(os.path.join(args.out, "qwha_export_meta.json"), "w") as f:
        json.dump({**meta, "adapter_dir": os.path.abspath(args.adapter_dir),
                   "worst_relative_error": worst}, f, indent=2)
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
