"""Build the QWHA base on THIS repo's GPTQ grid, with the balanced in-domain calibration.

Why not upstream's own path. `quantize_base.py` goes through optimum's GPTQConfig with
`dataset="wikitext2"` -- generic text, no prompt template. That is exactly the calibration this
repo measured as broken at low bit width: the same merged checkpoint quantized on 128 first-N
records (100% BoolQ, 9.5k tokens) scored 36.64 at INT2 and 66.22 on a task-balanced 3500-record
set (471k tokens), while standard C4 128x2048 loses the instruction template entirely (30.67).
A QWHA row on a wikitext2-calibrated base would be reporting the calibration, not the adapter.

So the base is built the way every bcal row in the tables is built -- `src/data.load_calibration_data`
(calibration_samples 3500, calibration_sampling balanced, calibration_seq_len 2048, full records
with the prompt template, padding masked out of the Hessians) feeding `src/gptq.py` -- and then
PACKED into GPTQModel's checkpoint format, which is what upstream's peft fork wraps. Packing
rather than re-quantizing is what makes it the same grid: pack() recovers codes as
round(W/s + z), so handing it our dequantized weights with our (scale, zero) reproduces our
codes exactly. The round trip is asserted at the end -- if the layout or the zero-point
convention were off, this stops here instead of reporting a number.

g_idx = arange(in_features) // group_size, i.e. desc_act=False, because src/gptq.py does not
reorder columns by activation norm.

    python make_bcal_base.py --config baseline/QWHA/sqat/configs/qwha_cs170k_int2_g32_ep1_span_bcal.yaml
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch
import yaml
from torch.utils.data import DataLoader

# qwha_common first: it puts the repo root and upstream's src/ on sys.path.
from qwha_common import SQAT_REPO, TARGET_MODULES, sqat_data_module, sqat_gptq_module

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization.config import (  # noqa: E402
    FORMAT, META_FIELD_QUANTIZER, META_QUANTIZER_GPTQMODEL, QUANT_METHOD, QuantizeConfig,
)
from gptqmodel.utils.model import pack_model  # noqa: E402
from gptqmodel.version import __version__ as gptqmodel_version  # noqa: E402


def base_dir_for(cfg: dict, out_root: str, tag: str = "Llama-2-7B") -> str:
    bits = int(cfg["model"]["quant_bits"])
    gs = int(cfg["qwha"]["group_size"])
    return os.path.join(out_root, f"{tag}_int{bits}_{gs}_asym_bcal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/qwha_bases")
    ap.add_argument("--tag", default="Llama-2-7B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.chdir(SQAT_REPO)          # data.train_dataset paths are repo-root relative
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bits = int(cfg["model"]["quant_bits"])
    group_size = int(cfg["qwha"]["group_size"])
    symmetric = bool(cfg["qat"].get("symmetric", False))
    if symmetric:
        raise SystemExit("the QWHA rows are asymmetric, like every other row in these tables")
    gcfg = cfg["qat"]["gptq"]
    scfg = cfg["qat"]["sqat"]

    save_dir = base_dir_for(cfg, args.out, args.tag)
    if os.path.isfile(os.path.join(save_dir, "quantize_config.json")):
        print(f"[bcal-base] {save_dir} already built; nothing to do.")
        return

    print("=" * 70)
    print(f"  QWHA base — INT{bits} g{group_size} asym, this repo's GPTQ + balanced calibration")
    print(f"  calibration: {cfg['data']['train_dataset']}, {scfg['calibration_samples']} records, "
          f"sampling={scfg['calibration_sampling']}, seq_len={scfg['calibration_seq_len']}")
    print(f"  gptq:        nsamples {gcfg['nsamples']}, batch {gcfg['batch_size']}, "
          f"percdamp {gcfg['percdamp']}, blocksize {gcfg['blocksize']}, act-order off")
    print(f"  out:         {save_dir}")
    print("=" * 70)
    if int(gcfg["nsamples"]) != int(scfg["calibration_samples"]):
        raise SystemExit(
            f"qat.gptq.nsamples ({gcfg['nsamples']}) must equal qat.sqat.calibration_samples "
            f"({scfg['calibration_samples']}): gptq_quantize_model_sequential only consumes the "
            f"first nsamples records of the calibration set.")

    data = sqat_data_module()
    gptq_quantize_model_sequential = sqat_gptq_module().gptq_quantize_model_sequential

    model_id = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, model_max_length=cfg["model"]["max_seq_len"], padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True).to(args.device)
    model.eval()

    cal = data.load_calibration_data(cfg, tokenizer)
    dl = DataLoader(cal, batch_size=int(gcfg["batch_size"]), shuffle=False,
                    collate_fn=data.build_data_collator(tokenizer))

    quantized = gptq_quantize_model_sequential(
        model=model,
        calibration_dataloader=dl,
        target_terminals=tuple(TARGET_MODULES),
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
    print(f"[bcal-base] GPTQ done for {len(quantized)} projections")

    # The weights left in the model are the dequantized codes; pack() recovers the codes from
    # them as round(W/s + z), so the packed checkpoint carries exactly this grid. Both the
    # weights and the scales are kept: the round trip below measures the error in GRID STEPS,
    # which is the only unit in which "the same grid" means anything.
    reference = {n: (m.weight.data.clone().float().cpu(), quantized[n][1].float().cpu())
                 for n, m in model.named_modules() if n in quantized}

    modules = dict(model.named_modules())
    quant_result = {
        name: {
            "scale": scale.float(),
            "zero": zp.float(),
            "g_idx": torch.arange(modules[name].in_features, dtype=torch.int32) // group_size,
        }
        for name, (_w_int, scale, zp) in quantized.items()
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

    # FORMAT.GPTQ_V2, not GPTQ, and the difference is one quantization step on every weight.
    # gptqmodel's loader converts a `gptq` (v1) checkpoint on the way in -- v1 serialised
    # `qzeros = zero - 1`, so from_quantized() adds 1 back unconditionally, and its own writer
    # therefore converts v2 -> v1 before saving as v1. We do not: PackableQuantLinear.pack()
    # stores the TRUE zero, i.e. the v2 convention already, so declaring v1 here would hand the
    # loader zeros it then shifts by one. The round-trip assertion below is what caught this.
    qcfg = QuantizeConfig(
        bits=bits, group_size=group_size, sym=symmetric, desc_act=False,
        damp_percent=float(gcfg["percdamp"]), format=FORMAT.GPTQ_V2,
        quant_method=QUANT_METHOD.GPTQ, pack_dtype=torch.int32,
    )
    # Without a producer stamp, loading sym=False refuses outright ("only supported if produced
    # by gptqmodel version >= 0.9.0"); QuantizeConfig() leaves meta empty, and only gptqmodel's
    # own quantize() path fills it in.
    qcfg.meta_set_versionable(META_FIELD_QUANTIZER,
                              [f"{META_QUANTIZER_GPTQMODEL}:{gptqmodel_version}"])
    os.makedirs(save_dir, exist_ok=True)
    model.config.quantization_config = qcfg.to_dict()
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    qcfg.save_pretrained(save_dir)
    print(f"[bcal-base] wrote {save_dir}")

    # --- the grid must survive the round trip, or the row means nothing --------------------
    del model
    torch.cuda.empty_cache()
    # float16, not bfloat16: pack() stores the scales in fp16, and reloading in bf16 would recast
    # them to 8 mantissa bits — the comparison would then measure the reload dtype, not the grid.
    check = GPTQModel.load(save_dir, torch_dtype=torch.float16, device_map=args.device,
                           trust_remote_code=True, backend=BACKEND.TORCH)
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    worst, worst_name, n = 0.0, "", 0
    for name, module in check.model.named_modules():
        if not isinstance(module, BaseQuantLinear) or name not in reference:
            continue
        w_ref, scale = reference[name]
        with torch.no_grad():
            got = module.dequantize_weight().float().T.cpu()      # [in,out] -> [out,in]
        # In grid steps: a v1/v2 zero-point mix-up is exactly 1.0 here, while the fp16 cast
        # pack() applies to our fp32 scales is ~1e-3 of a step.
        steps = ((got - w_ref).abs()
                 / scale.repeat_interleave(group_size, dim=1)[:, :w_ref.shape[1]]).max().item()
        if steps > worst:
            worst, worst_name = steps, name
        n += 1
    print(f"[bcal-base] round-trip checked {n} projections, "
          f"max error = {worst:.2e} grid steps ({worst_name})")
    if n != len(quantized):
        raise SystemExit(f"only {n} of {len(quantized)} projections came back quantized")
    if worst > 0.01:
        raise SystemExit("packed grid differs from the GPTQ output — do not train on this base")
    print("[bcal-base] OK — the packed base carries this repo's balanced-calibration GPTQ grid")


if __name__ == "__main__":
    main()
