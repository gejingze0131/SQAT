"""train == deploy for the LoTA-QAF export.

This repo's most expensive class of bug is a training grid that differs from the deployed one
(AGENTS.md, invariant 1). LoTA-QAF has the same exposure from the other side: training runs
upstream's CustomLoraLinear forward, while the reported number comes from the dense checkpoint
export_lota_dense.py writes. If the two disagree, the score measures neither.

So compare them directly. For a real GPTQ layer and a real (or randomly ternarised) adapter:

    live   = CustomLoraLinear.forward(x)                 what training optimises
    dense  = x @ merged_dense_weight(...)                what vLLM is handed

and require them equal to bf16 rounding. Both the boundary masks and the offset factor are
exercised: the masks bite hardest at 2 bits (4 codes), so run this for each cell's base.

    python tests/test_export_consistency.py --quantized_model_dir <gptq base>
    python tests/test_export_consistency.py --quantized_model_dir <gptq base> \
        --adapter_dir outputs/lota_.../final
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # export_lota_dense

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from peft.tuners.lora.layer import CustomLoraLinear  # noqa: E402

from export_lota_dense import adapter_for, load_adapter_tensors, merged_dense_weight  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantized_model_dir", required=True)
    ap.add_argument("--adapter_dir", default=None, help="default: random ternary init")
    ap.add_argument("--omega", type=int, default=48)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4, help="how many quantized linears to check")
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = GPTQModel.load(
        args.quantized_model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, backend=BACKEND.AUTO_TRAINABLE,
    )
    inner = model.model
    adapters = load_adapter_tensors(args.adapter_dir) if args.adapter_dir else {}

    quant_linears = [(n, m) for n, m in inner.named_modules() if isinstance(m, BaseQuantLinear)]
    # Spread the sample over the depth rather than taking the first N of one block.
    step = max(1, len(quant_linears) // args.n_layers)
    picked = quant_linears[::step][: args.n_layers]
    print(f"[test] {len(quant_linears)} quantized linears; checking {len(picked)}")

    worst = 0.0
    for name, base in picked:
        layer = CustomLoraLinear(base, "default", r=args.rank, lora_alpha=2 * args.rank,
                                 lora_dropout=0.0, threshold=args.omega, residual=True)
        layer.to(base.qweight.device)
        pair = None
        if adapters:
            _, pair = adapter_for(adapters, name)
        if pair is not None:
            A, B = pair
            with torch.no_grad():
                layer.lora_A["default"].weight.copy_(A.to(layer.lora_A["default"].weight.device))
                layer.lora_B["default"].weight.copy_(B.to(layer.lora_B["default"].weight.device))
            src = "trained"
        else:
            # lora_B is zero-initialised by construction, which would make dW == 0 and hide any
            # marker/offset bug. Ternarise both so the thresholded path is actually taken.
            with torch.no_grad():
                for p in (layer.lora_A["default"].weight, layer.lora_B["default"].weight):
                    p.copy_(torch.randint(-1, 2, p.shape, device=p.device).to(p.dtype))
            src = "random"

        A = layer.lora_A["default"].weight.detach()
        B = layer.lora_B["default"].weight.detach()
        x = torch.randn(args.tokens, base.in_features, device=base.qweight.device,
                        dtype=torch.bfloat16)

        with torch.no_grad():
            live = layer(x)
            w = merged_dense_weight(base, A, B, omega=args.omega, residual=True)
            dense = torch.nn.functional.linear(x, w.T)

        denom = live.abs().max().item() or 1.0
        rel = (live - dense).abs().max().item() / denom
        worst = max(worst, rel)
        nnz = int((torch.matmul(A.T.float(), B.T.float()).abs() > args.omega).sum())
        print(f"[test] {name:<48s} {src:<7s} markers={nnz:>10,d}  rel.max|d|={rel:.2e}")
        del layer, w
        torch.cuda.empty_cache()

    # bf16 has ~3 decimal digits; the two paths do the same ops in a different order, so exact
    # equality is not required, only that no term is missing or mis-scaled.
    tol = 5e-3
    print(f"[test] worst relative deviation {worst:.2e} (tol {tol:.0e})")
    if worst > tol:
        raise SystemExit("FAIL: dense export disagrees with the trained forward")
    print("[test] OK — train == deploy")


if __name__ == "__main__":
    main()
