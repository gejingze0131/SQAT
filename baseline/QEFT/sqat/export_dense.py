"""
Export a trained QEFT model to a dense checkpoint runs/eval_vllm.sh can serve.

A QEFT layer is already exactly one dense linear map — frozen dequantized bulk with the trained
fp16 weak columns written back into their own positions — so the export is a scatter, not a
merge, and it is CHECKED rather than assumed: for every layer the scattered weight is compared
against the live QEFTLinear's effective weight (max |Δ| must be 0) and against its forward on
random inputs.

What the dense checkpoint does NOT do is make the model b-bit. A deployed QEFT model is INT-b
codes PLUS k fp16 columns per layer (the file's meta carries the exact fp16 share and effective
bit width); the dense file is a numerical stand-in for evaluation, the same seam every other row
in RESULTS_SUMMARY.md is scored through.

    python baseline/QEFT/sqat/export_dense.py --ckpt outputs/qeft_.../final --out <dir>
    python baseline/QEFT/sqat/export_dense.py --ckpt none --base_dir <base> --out <dir>   # floor
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qeft_common import (build_qeft_model, load_qeft_meta, load_qeft_trainable,  # noqa: E402
                         qeft_base_dir_from_checkpoint, qeft_layers)


@torch.no_grad()
def _forward_error(module, n: int = 4) -> float:
    """max |QEFT layer − dense layer| on random inputs, relative to the output scale.

    Run in fp32 (the module is cast for the duration and restored afterwards) so the number
    measures the DECOMPOSITION — two GEMMs plus a gather against one dense GEMM — rather than
    bf16 rounding, which on a 4096-term reduction would swamp it and force a tolerance loose
    enough to hide a real error.
    """
    dtype = module.weight.dtype
    module.float()
    try:
        x = torch.randn(n, module.in_features, device=module.weight.device)
        y_ref = module(x)
        y_dense = torch.nn.functional.linear(x, module.effective_weight(torch.float32))
        return ((y_ref - y_dense).abs().max().item()
                / max(y_ref.abs().max().item(), 1e-6))
    finally:
        module.to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="trained checkpoint dir (…/final), or 'none' to export the bare "
                         "mixed-precision base — this method's own floor — which needs --base_dir")
    ap.add_argument("--base_dir", default=None,
                    help="overrides the base recorded in the checkpoint; required with --ckpt none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max allowed relative error between the QEFT layer and its dense form")
    args = ap.parse_args()

    bare = args.ckpt.lower() == "none"
    base_dir = args.base_dir or (None if bare else qeft_base_dir_from_checkpoint(args.ckpt))
    if not base_dir:
        raise SystemExit("no base directory: pass --base_dir (or train with a checkpoint meta)")
    meta = load_qeft_meta(base_dir)
    dtype = getattr(torch, meta.get("dtype", "bfloat16"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[export] base {base_dir}: INT{meta['q_bits']} g{meta['group_size']} k={meta['k']} "
          f"({meta['fp16_share'] * 100:.2f}% fp16 → {meta['effective_bits']:.2f} effective bits)")

    # The live model, in the SAME dtype it trained and deploys in, so the check measures the
    # export rather than a cast.
    _zp = False
    if not bare:
        from safetensors import safe_open as _so
        with _so(os.path.join(args.ckpt, "qeft_weak_columns.safetensors"), "pt") as _f:
            _zp = any(k.endswith(".zp_shift") for k in _f.keys())
    model, _ = build_qeft_model(base_dir, dtype=dtype, device=device, weak_dtype=dtype,
                                train_zp=_zp)
    if bare:
        print("[export] bare base: weak columns left at their quantization-time values")
    else:
        n = load_qeft_trainable(model, args.ckpt)
        print(f"[export] loaded {n} trained weak-column tensors from {args.ckpt}")
    model.eval()

    dense = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=dtype, device_map="cpu")
    target = dict(dense.named_modules())

    worst_fwd, n_layers = 0.0, 0
    for name, module in tqdm(sorted(qeft_layers(model).items()), "Materializing"):
        if name not in target:
            raise KeyError(f"{name} is not a module of the base checkpoint")
        w_eff = module.effective_weight(dtype=dtype)
        w_cpu = w_eff.cpu()
        base_w = target[name].weight.data
        ids = module.weak_ids.cpu()
        mask = torch.ones(base_w.shape[1], dtype=torch.bool)
        mask[ids] = False
        # 1) EXACT, both halves, against the base checkpoint on disk rather than against the
        #    module we just read: the quantized bulk must be untouched by a run that is only
        #    allowed to move k columns, and those k columns must be exactly what was trained.
        if getattr(module, "train_zp", False):
            # With a trained zp tier the non-weak columns move by a per-group-uniform shift.
            # Recompute it exactly the way effective_weight does and demand bitwise agreement.
            _delta = (module.zp_shift * module.zp_step).detach().float().cpu()
            _exp = base_w.float().clone()
            _exp[:, module.nonweak_ids.cpu()] += _delta.repeat_interleave(module.group_size, dim=1)
            frozen_drift = (_exp.to(dtype)[:, mask] - w_cpu[:, mask]).abs().max().item()
        else:
            frozen_drift = (base_w[:, mask] - w_cpu[:, mask]).abs().max().item()
        weak_drift = (module.weight_weak.detach().cpu().to(dtype)
                      - w_cpu[:, ids]).abs().max().item()
        if frozen_drift != 0.0 or weak_drift != 0.0:
            raise RuntimeError(f"{name}: export is not a pure scatter "
                               f"(frozen {frozen_drift:.3e}, weak {weak_drift:.3e})")
        # 2) BEHAVIOURAL: the two-GEMM training forward and the dense one agree.
        err = _forward_error(module)
        worst_fwd = max(worst_fwd, err)
        if err > args.tol:
            raise RuntimeError(f"{name}: dense form differs from the QEFT layer by {err:.2e}")
        target[name].weight.data.copy_(w_cpu)
        n_layers += 1
        del w_eff, w_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[export] {n_layers} layers; worst relative forward error {worst_fwd:.2e} "
          f"(tol {args.tol:.0e})")

    os.makedirs(args.out, exist_ok=True)
    dense.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(args.out)
    run_meta = {}
    run_path = os.path.join(args.ckpt, "qeft_run_meta.json") if not bare else None
    if run_path and os.path.exists(run_path):
        with open(run_path) as f:
            run_meta = json.load(f)
    with open(os.path.join(args.out, "qeft_export_meta.json"), "w") as f:
        json.dump({"base_dir": os.path.abspath(base_dir),
                   "checkpoint": None if bare else os.path.abspath(args.ckpt),
                   "bits": meta["q_bits"], "group_size": meta["group_size"], "k": meta["k"],
                   "oproj_weak": meta["oproj_weak"],
                   "fp16_share": meta["fp16_share"], "effective_bits": meta["effective_bits"],
                   "worst_relative_forward_error": worst_fwd,
                   "run": run_meta}, f, indent=2)
    print(f"[export] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
