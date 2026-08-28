"""Build a GPTQ base with THIS REPO's settings, in the format LoTA-QAF trains on.

The paper-faithful base (quantize_base.py) uses GPTQModel's own recipe: 1024 C4 sequences,
act-order, true-sequential. Ours (src/gptq.py) uses 128 sequences of the task's own training
data, no act-order. Those are different starting grids, so a LoTA-QAF row built on the first
one cannot be read as "method only" against SALT-Q and QA-LoRA — its floor moved too.

This script closes that gap from the other side: it runs OUR GPTQ, on OUR calibration data, at
OUR budget, and then packs the resulting integer codes into a GPTQModel checkpoint so LoTA-QAF
can train on the SAME grid QA-LoRA's base carries. Packing rather than re-quantizing is what
makes it the same grid: GPTQModel's pack() recovers integer codes as round(W/s + z), so handing
it OUR dequantized weights with OUR (scale, zero) reproduces our codes exactly. The script
asserts that afterwards by loading the checkpoint back through GPTQModel and comparing
dequantize_weight() against the weights our GPTQ left in place -- if anything about the layout
or the zero-point convention were off, the run stops here rather than reporting a number.

g_idx is arange(in_features) // group_size, i.e. desc_act=False, because src/gptq.py does not
reorder columns by activation norm.

    python make_matched_base.py --config ../../../configs/qalora_cs170k_int2_g32_ep1_span.yaml \
        --out outputs/lota_bases --tag Llama-2-7B
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
SQAT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, SQAT_ROOT)

from transformers import AutoModelForCausalLM  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization.config import FORMAT, QUANT_METHOD, QuantizeConfig  # noqa: E402
from gptqmodel.utils.model import pack_model  # noqa: E402

from lota_common import resolve_pretrained  # noqa: E402
from src.data import build_data_collator, load_calibration_data  # noqa: E402
from src.gptq import gptq_quantize_model_sequential  # noqa: E402
from src.model_loader import load_tokenizer  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="a SQAT config; model, data, qat.group_size and qat.sqat_permute.gptq "
                         "are read from it so the base matches the rows it is compared against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="Llama-2-7B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.chdir(SQAT_ROOT)   # data.train_dataset paths are repo-root relative
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bits = int(cfg["model"]["quant_bits"])
    group_size = int(cfg["qat"]["group_size"])
    symmetric = bool(cfg["qat"].get("symmetric", False))
    if symmetric:
        raise SystemExit("LoTA-QAF's ternary adaptation assumes an asymmetric affine grid")
    gcfg = cfg["qat"]["sqat_permute"]["gptq"]

    save_dir = os.path.join(args.out, f"{args.tag}_int{bits}_{group_size}_asym_sqatcal")
    if os.path.isfile(os.path.join(save_dir, "quantize_config.json")):
        print(f"[matched-base] {save_dir} already built; nothing to do.")
        return

    print("=" * 70)
    print(f"  matched GPTQ base — INT{bits} g{group_size}, this repo's settings")
    print(f"  calibration: {cfg['data']['train_dataset']}, {gcfg['nsamples']} sequences, "
          f"batch {gcfg['batch_size']}, percdamp {gcfg['percdamp']}, blocksize {gcfg['blocksize']}")
    print(f"  act-order:   off (src/gptq.py does not reorder columns)")
    print(f"  out:         {save_dir}")
    print("=" * 70)

    pretrained = resolve_pretrained(cfg["model"]["name"])
    tokenizer = load_tokenizer(cfg, name=pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device)
    model.eval()

    cal = load_calibration_data(cfg, tokenizer)
    dl = DataLoader(cal, batch_size=int(gcfg["batch_size"]), shuffle=False,
                    collate_fn=build_data_collator(tokenizer))

    quantized = gptq_quantize_model_sequential(
        model=model,
        calibration_dataloader=dl,
        target_terminals=TARGETS,
        perm_group_k=0,               # no salient slice: every column is GPTQ'd
        group_size=group_size,
        q_bits=bits,
        symmetric=symmetric,
        device=torch.device(args.device),
        perm_meta=None,               # unpermuted base, no boundary gathers
        percdamp=float(gcfg["percdamp"]),
        blocksize=int(gcfg["blocksize"]),
        nsamples=int(gcfg["nsamples"]),
    )
    print(f"[matched-base] GPTQ done for {len(quantized)} projections")

    # The weights left in the model are the dequantized codes; pack() recovers the codes from
    # them as round(W/s + z), so the packed checkpoint carries exactly this grid.
    reference = {n: m.weight.data.clone().float().cpu()
                 for n, m in model.named_modules() if n in quantized}

    quant_result = {}
    for name, (_w_int, scale, zp) in quantized.items():
        in_f = dict(model.named_modules())[name].in_features
        quant_result[name] = {
            "scale": scale.float(),
            "zero": zp.float(),
            "g_idx": torch.arange(in_f, dtype=torch.int32) // group_size,
        }

    pack_model(
        model=model,
        quant_result=quant_result,
        bits=bits,
        group_size=group_size,
        backend=BACKEND.TORCH,
        format=FORMAT.GPTQ,
        quant_method=QUANT_METHOD.GPTQ,
        lm_head_name="lm_head",
        desc_act=False,
        sym=symmetric,
        parallel_packing=True,
        pack_dtype=torch.int32,
    )

    qcfg = QuantizeConfig(
        bits=bits, group_size=group_size, sym=symmetric, desc_act=False,
        damp_percent=float(gcfg["percdamp"]), format=FORMAT.GPTQ,
        quant_method=QUANT_METHOD.GPTQ, pack_dtype=torch.int32,
    )
    os.makedirs(save_dir, exist_ok=True)
    model.config.quantization_config = qcfg.to_dict()
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    qcfg.save_pretrained(save_dir)
    print(f"[matched-base] wrote {save_dir}")

    # --- the grid must survive the round trip, or the row means nothing -------------------
    del model
    torch.cuda.empty_cache()
    check = GPTQModel.load(save_dir, torch_dtype=torch.bfloat16, device_map=args.device,
                           trust_remote_code=True, backend=BACKEND.TORCH)
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    worst, n = 0.0, 0
    for name, module in check.model.named_modules():
        if not isinstance(module, BaseQuantLinear) or name not in reference:
            continue
        with torch.no_grad():
            got = module.dequantize_weight().float().T.cpu()   # [in,out] -> [out,in]
        d = (got - reference[name]).abs().max().item()
        worst = max(worst, d)
        n += 1
    print(f"[matched-base] round-trip checked {n} projections, max|delta| = {worst:.3e}")
    if n != len(quantized):
        raise SystemExit(f"only {n} of {len(quantized)} projections came back quantized")
    if worst > 1e-3:
        raise SystemExit("packed grid differs from the GPTQ output — do not train on this base")
    print("[matched-base] OK — the packed base carries this repo's GPTQ grid")


if __name__ == "__main__":
    main()
