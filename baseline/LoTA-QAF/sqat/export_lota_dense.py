"""Merge a trained LoTA-QAF adapter and export a dense fp16 checkpoint for the shared eval.

WHY A DENSE EXPORT. Every row of RESULTS_SUMMARY.md is scored the same way: greedy vLLM
generation on datasets/commonsense/test.json through runs/eval_vllm.sh, exact match by
scripts/test_acc.py. Upstream LoTA-QAF instead evaluates through GPTQModel's own kernels with
an LTA adapter object (LoTA/adapter.py) under lm-eval. Scoring this baseline that way would
put a different harness, a different prompt path and a different metric next to our numbers.
So the merge is done here and the result is written as an ordinary Llama checkpoint.

WHAT IS MERGED. Exactly what upstream's LTA.apply()/LTA.merge() do, in the same order and the
same dtypes (bf16, where the ternary products are integers in [-r, r] and therefore exact):

    dW      = A^T B                        integer-valued, in [-r, r]
    markers = sign(dW) * 1[|dW| > omega]   masked so q+marker stays inside [0, maxq]
    q'      = q + markers                  the lossless int merge, Eq. (5)
    resid   = dW - omega * markers
    mu_g    = groupmean(resid) / omega     the offset factor, folded into the zero point
    W       = scales[g] * (q' - zeros[g] + mu_g[g])

The last line is the deployed weight: integer codes at the model's bit width plus a per-group
fractional zero-point shift. That is the same deployment contract as the QA-LoRA rows (a
group-constant delta folded into the affine zero-point), so "2 bits" means the same thing in
both -- no fp16 side channel is being smuggled in by the dense format.

Equivalence with the trained model is not assumed: tests/test_export_consistency.py compares
this weight against the live CustomLoraLinear forward.

    python export_lota_dense.py --quantized_model_dir <gptq base> \
        --adapter_dir outputs/lota_.../final --out outputs/lota_...-3bit-lota-dequant-eval
    python export_lota_dense.py --quantized_model_dir <gptq base> --adapter_dir none --out <dir>
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lota_common import resolve_pretrained  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# --- gptqmodel's own bit unpacking, copied verbatim (LoTA/layer.py copies it too) ---------

def decode_qweight(layer):
    if layer.bits in [2, 4, 8]:
        w = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(layer.qweight, 1).expand(-1, layer.pack_factor, -1),
                layer.wf_unsqueeze_neg_one,
            ).to(layer.dequant_dtype),
            layer.maxq,
        )
        return w.reshape(layer.in_features, layer.out_features)
    if layer.bits == 3:
        w = layer.qweight.reshape(layer.qweight.shape[0] // 3, 3, 1, layer.qweight.shape[1]).expand(-1, -1, 12, -1)
        w = (w >> layer.wf_unsqueeze_neg_one) & 0x7
        w[:, 0, 10] = (w[:, 0, 10] & 0x3) | ((w[:, 1, 0] << 2) & 0x4)
        w[:, 1, 11] = (w[:, 1, 11] & 0x1) | ((w[:, 2, 0] << 1) & 0x6)
        w = w & 0x7
        w = torch.cat([w[:, 0, :11], w[:, 1, 1:12], w[:, 2, 1:11]], dim=1)
        return w.reshape(layer.in_features, layer.out_features)
    raise ValueError(f"Unsupported bits: {layer.bits}")


def decode_zeros(layer):
    if layer.bits in [2, 4, 8]:
        z = torch.bitwise_right_shift(
            torch.unsqueeze(layer.qzeros, 2).expand(-1, -1, layer.pack_factor),
            layer.wf_unsqueeze_zero,
        ).to(layer.dequant_dtype)
        return torch.bitwise_and(z, layer.maxq).reshape(layer.scales.shape)
    if layer.bits == 3:
        z = layer.qzeros.reshape(layer.qzeros.shape[0], layer.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        z = z >> layer.wf_unsqueeze_zero
        z[:, :, 0, 10] = (z[:, :, 0, 10] & 0x3) | ((z[:, :, 1, 0] << 2) & 0x4)
        z[:, :, 1, 11] = (z[:, :, 1, 11] & 0x1) | ((z[:, :, 2, 0] << 1) & 0x6)
        z = z & 0x7
        z = torch.cat([z[:, :, 0, :11], z[:, :, 1, 1:12], z[:, :, 2, 1:11]], dim=2).reshape(layer.scales.shape)
        return z
    raise ValueError(f"Unsupported bits: {layer.bits}")


def merged_dense_weight(layer, lora_A=None, lora_B=None, omega=None, residual=True,
                        compute_dtype=torch.bfloat16):
    """[in_features, out_features] deployed weight. lora_A/B None => the bare GPTQ base."""
    q = decode_qweight(layer)
    zeros = decode_zeros(layer)
    g_idx = layer.g_idx.long()

    # The unpack is self-checked against gptqmodel's own dequantize_weight().
    ref = layer.dequantize_weight()
    got = layer.scales[g_idx] * (q - zeros[g_idx])
    if not torch.equal(ref, got):
        raise RuntimeError(f"unpack mismatch on {layer.name}: max|d|={(ref - got).abs().max()}")

    if lora_A is None:
        return got.to(compute_dtype)

    A = lora_A.to(device=q.device, dtype=compute_dtype)   # [r, in]
    B = lora_B.to(device=q.device, dtype=compute_dtype)   # [out, r]
    AB = torch.matmul(A.T, B.T).to(compute_dtype).contiguous()   # [in, out]

    can_up = q != layer.maxq
    can_down = q != 0
    markers = torch.where(
        (AB > omega) & can_up,
        torch.ones_like(AB),
        torch.where((AB < -omega) & can_down, -torch.ones_like(AB), torch.zeros_like(AB)),
    )

    q_new = q + markers.to(q.dtype)
    w = q_new - zeros[g_idx]

    if residual:
        resid = AB - markers * omega
        _, sorted_indices = torch.sort(g_idx)
        sorted_resid = resid[sorted_indices]
        mu = sorted_resid.view(layer.scales.shape[0], layer.group_size, layer.out_features).mean(dim=1) / omega
        w = w + mu[g_idx]

    return (layer.scales[g_idx] * w).to(compute_dtype)


def load_adapter_tensors(adapter_dir):
    """peft's adapter_model.safetensors -> {peft module path: (A, B)}.

    The keys keep peft's full path, which is nested twice deeper than the model's own: peft
    wraps the GPTQModel wrapper (itself an nn.Module holding the LlamaForCausalLM), so a
    projection reached as `model.layers.0.self_attn.q_proj` inside the model is stored as
    `base_model.model.model.model.layers.0.self_attn.q_proj`. Rather than hardcode that
    prefix, `adapter_for()` matches on the suffix.
    """
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sd = load_file(path)
    out = {}
    for k, v in sd.items():
        if ".lora_A" not in k and ".lora_B" not in k:
            continue
        which = "A" if ".lora_A" in k else "B"
        out.setdefault(k.split(".lora_")[0], {})[which] = v
    missing = [m for m, d in out.items() if set(d) != {"A", "B"}]
    if missing:
        raise RuntimeError(f"adapter is missing an A or B for: {missing[:5]}")
    print(f"[export] adapter tensors for {len(out)} modules from {path}")
    return {m: (d["A"], d["B"]) for m, d in out.items()}


def adapter_for(adapters, name):
    """The (A, B) whose peft path ends in this module's path — exactly one must."""
    hits = [k for k in adapters if k == name or k.endswith("." + name)]
    if len(hits) != 1:
        raise RuntimeError(
            f"{name}: matched {len(hits)} adapter keys ({hits[:3]}); the adapter does not "
            f"correspond to this model"
        )
    return hits[0], adapters[hits[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantized_model_dir", required=True)
    ap.add_argument("--adapter_dir", required=True, help='"none" exports the bare GPTQ base')
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf",
                    help="source of the untouched fp16 tensors (embeddings, norms, lm_head)")
    ap.add_argument("--omega", type=int, default=None, help="default: read from the adapter's lota_run.yaml")
    ap.add_argument("--residual", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with_adapter = args.adapter_dir.lower() != "none"
    omega, residual = args.omega, args.residual
    if with_adapter:
        run_yaml = os.path.join(args.adapter_dir, "lota_run.yaml")
        if os.path.isfile(run_yaml) and (omega is None or residual is None):
            import yaml
            meta = yaml.safe_load(open(run_yaml))["lota"]
            omega = omega if omega is not None else int(meta["interval_point"])
            residual = residual if residual is not None else int(bool(meta["residual"]))
        if omega is None:
            raise SystemExit("--omega is required when the adapter has no lota_run.yaml")
        residual = 1 if residual is None else residual
        print(f"[export] omega={omega} residual={bool(residual)}")

    pretrained = resolve_pretrained(args.pretrained)
    print(f"[export] fp16 skeleton from {pretrained}")
    dense = AutoModelForCausalLM.from_pretrained(
        pretrained, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    dense_mods = dict(dense.named_modules())

    print(f"[export] GPTQ base from {args.quantized_model_dir}")
    gptq = GPTQModel.load(
        args.quantized_model_dir, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, backend=BACKEND.TORCH,
    )
    inner = gptq.model

    adapters = load_adapter_tensors(args.adapter_dir) if with_adapter else {}

    n_done, n_adapted = 0, 0
    matched = set()
    for name, module in inner.named_modules():
        if not isinstance(module, BaseQuantLinear):
            continue
        if not name.endswith(TARGETS):
            raise RuntimeError(f"unexpected quantized module {name}")
        A = B = None
        if with_adapter:
            key, (A, B) = adapter_for(adapters, name)
            matched.add(key)
            n_adapted += 1
        with torch.no_grad():
            w = merged_dense_weight(module, A, B, omega=omega, residual=bool(residual))
        target = dense_mods.get(name)
        if target is None:
            raise RuntimeError(f"{name} has no counterpart in the fp16 skeleton")
        if tuple(target.weight.shape) != (w.shape[1], w.shape[0]):
            raise RuntimeError(f"{name}: shape {tuple(target.weight.shape)} vs {tuple(w.shape)}")
        with torch.no_grad():
            target.weight.copy_(w.T.to(torch.float16).cpu())
        n_done += 1
        del w
        torch.cuda.empty_cache()
    print(f"[export] rewrote {n_done} projections ({n_adapted} with a ternary adapter)")

    unused = set(adapters) - matched
    if unused:
        raise RuntimeError(
            f"{len(unused)} adapter tensors were never merged (e.g. {sorted(unused)[:3]}); "
            f"the export would silently drop part of the fine-tuning"
        )

    os.makedirs(args.out, exist_ok=True)
    dense.save_pretrained(args.out, safe_serialization=True)
    tok_src = args.adapter_dir if (with_adapter and os.path.isfile(
        os.path.join(args.adapter_dir, "tokenizer_config.json"))) else pretrained
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.out)
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
