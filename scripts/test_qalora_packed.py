#!/usr/bin/env python
"""
QALoRAPackedLinear must be a STORAGE change: the base weight it rebuilds is bit-identical to the
dense fp16 checkpoint, and the QA-LoRA forward on top of it is unchanged.

WHY THIS EXISTS. QA-LoRA trains on the real GPTQ INT-b grid, and build_qalora_intb_base ships that
grid as a DENSE fp16 checkpoint — 61 GiB for Qwen2.5-32B, replicated per DDP rank, against 44.4
GiB of usable VRAM. The grid is b bits plus a per-group (scale, zero), so the dense form carries
8x the information it needs; packed it is 7.3 GiB at INT2. That only makes the 32B row comparable
to the 7B one if the weight is the SAME weight, which is what this checks.

The dense checkpoint is produced by gptq_quantize_model_sequential as
    mod.weight.data.copy_(group_dequantize(W_int, s, z, ...).to(fp16))
with perm_group_k=0 and no permutation for QA-LoRA, so the rebuild calls the same
group_dequantize on the same tensors and casts once — equality is by construction, and this test
is what keeps it that way.

Checks:
  1. dequantize() == the dense fp16 weight, bitwise, at INT2/3/4 and asym+sym;
  2. forward() == F.linear against that dense weight, bitwise (incl. bias);
  3. the packed buffer really is ~q_bits/16 of the dense one;
  4. the module satisfies the contract the QA-LoRA code needs of a base layer:
     in_features/out_features, callable, a class name without "4bit" (patch_qalora_model rejects
     NF4 bases by that substring), and the `dequantize()` hook src.export._dequant_base_weight
     already looks for.

Run:  python scripts/test_qalora_packed.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.qalora import QALoRAPackedLinear
from src.quant_primitives import group_dequantize, group_quantize

OUT, IN, GS = 96, 256, 32


def main():
    fail = 0
    worst_all = 0.0
    for q_bits in (2, 3, 4):
        for symmetric in (False, True):
            g = torch.Generator().manual_seed(q_bits * 3 + symmetric)
            W = (torch.randn(OUT, IN, generator=g) * 0.02).float()
            # The grid exactly as the quantizer produces it
            W_int, scale, zero = group_quantize(W, GS, q_bits, symmetric)
            dense = group_dequantize(W_int, scale, zero, GS, IN, symmetric).to(torch.float16)

            bias = torch.randn(OUT, generator=g).to(torch.float16)
            mod = QALoRAPackedLinear(IN, OUT, W_int, scale, zero, group_size=GS, q_bits=q_bits,
                                     symmetric=symmetric, bias=bias, dtype=torch.float16)
            reb = mod.dequantize()
            d = (reb.float() - dense.float()).abs().max().item()
            worst_all = max(worst_all, d)
            if d != 0.0:
                fail += 1
                print(f"  [FAIL] INT{q_bits} {'sym ' if symmetric else 'asym'}: "
                      f"rebuilt != dense, max|Δ| = {d:g}")

    ok = worst_all == 0.0
    print(f"  [{'OK  ' if ok else 'FAIL'}] rebuilt base == dense fp16 checkpoint, bitwise, "
          f"INT2/3/4 x asym+sym (worst |Δ| = {worst_all:g})")

    # 2. forward
    g = torch.Generator().manual_seed(7)
    W = (torch.randn(OUT, IN, generator=g) * 0.02).float()
    W_int, scale, zero = group_quantize(W, GS, 2, False)
    dense = group_dequantize(W_int, scale, zero, GS, IN, False).to(torch.float16)
    bias = torch.randn(OUT, generator=g).to(torch.float16)
    mod = QALoRAPackedLinear(IN, OUT, W_int, scale, zero, group_size=GS, q_bits=2,
                             symmetric=False, bias=bias, dtype=torch.float16)
    x = torch.randn(4, 11, IN, generator=g).to(torch.float16)
    ok2 = torch.equal(mod(x), F.linear(x, dense, bias))
    print(f"  [{'OK  ' if ok2 else 'FAIL'}] forward == F.linear on the dense weight, bitwise (with bias)")
    fail += not ok2

    # 3. size
    packed = mod.codes_packed.numel() * mod.codes_packed.element_size()
    ok3 = packed <= dense.numel() * 2 / 6
    print(f"  [{'OK  ' if ok3 else 'FAIL'}] packed codes {packed} B vs dense fp16 "
          f"{dense.numel()*2} B ({dense.numel()*2/packed:.1f}x smaller)")
    fail += not ok3

    # 4. base-layer contract
    checks = {
        "in_features/out_features": (mod.in_features, mod.out_features) == (IN, OUT),
        "callable": callable(mod),
        "class name has no '4bit'": "4bit" not in type(mod).__name__.lower(),
        "no .weight with a quant_state": getattr(getattr(mod, "weight", None), "quant_state", None) is None,
        "dequantize() hook present": hasattr(mod, "dequantize"),
    }
    ok4 = all(checks.values())
    print(f"  [{'OK  ' if ok4 else 'FAIL'}] base-layer contract: "
          + ", ".join(k for k, v in checks.items() if v)
          + ("" if ok4 else " | MISSING: " + ", ".join(k for k, v in checks.items() if not v)))
    fail += not ok4

    # 5. the swap must actually FIND its targets inside a peft-wrapped tree. peft renames
    # everything to base_model.model.<...>.base_layer, and the packed file's keys were recorded on
    # the bare model — a mismatch replaces nothing, silently, and the dense weights stay resident
    # until the first OOM. (It also has to run AFTER get_peft_model: peft refuses to wrap a
    # non-nn.Linear base, "Target module QALoRAPackedLinear is not supported".)
    import tempfile, shutil
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    from src.qalora import swap_in_packed_base, QALORA_PACKED_FILENAME

    work = tempfile.mkdtemp(prefix="qalora_swap_")
    try:
        torch.manual_seed(0)
        TGT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        m = LlamaForCausalLM(cfg).to(torch.float16)
        tensors, ref = {}, {}
        for n, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in TGT:
                W = mod.weight.data.float()
                wi, sc, ze = group_quantize(W, GS, 2, False)
                tensors[f"{n}.codes"] = wi.to(torch.int8)
                tensors[f"{n}.scale"] = sc.float(); tensors[f"{n}.zero"] = ze.float()
                ref[n] = group_dequantize(wi, sc, ze, GS, mod.in_features, False).to(torch.float16)
        save_file(tensors, os.path.join(work, QALORA_PACKED_FILENAME))
        torch.save({"group_size": GS, "q_bits": 2, "symmetric": False},
                   os.path.join(work, "qalora_base_meta.pt"))

        peft_m = get_peft_model(m, LoraConfig(r=8, target_modules=TGT, lora_alpha=16))
        n_rep = swap_in_packed_base(peft_m, work, TGT, dtype=torch.float16)
        ok5 = n_rep == len(ref)
        print(f"  [{'OK  ' if ok5 else 'FAIL'}] swap inside a peft-wrapped tree replaced "
              f"{n_rep} of {len(ref)} target projections")
        fail += not ok5

        # export must be able to recover the base weight through _dequant_base_weight — the hook
        # it uses is dequantize(), but it read base.weight unconditionally first and raised
        # AttributeError on a module that has none. Only the export path hits this, so training
        # succeeding says nothing about it.
        from src.export import _dequant_base_weight

        n_ok = n_bad = 0
        for n, mod in peft_m.named_modules():
            if hasattr(mod, "base_layer") and isinstance(mod.base_layer, QALoRAPackedLinear):
                bare = n.replace("base_model.model.", "", 1)
                W = _dequant_base_weight(mod)
                if torch.equal(W, ref[bare].float()):
                    n_ok += 1
                else:
                    n_bad += 1
        ok7 = n_ok == len(ref) and n_bad == 0
        print(f"  [{'OK  ' if ok7 else 'FAIL'}] export's _dequant_base_weight recovers the packed "
              f"base ({n_ok} exact, {n_bad} wrong, of {len(ref)})")
        fail += not ok7

        got = {n for n, mod in peft_m.named_modules() if isinstance(mod, QALoRAPackedLinear)}
        ok6 = len(got) == len(ref)
        bad = 0
        for n, mod in peft_m.named_modules():
            if isinstance(mod, QALoRAPackedLinear):
                bare = n.replace("base_model.model.", "", 1)
                bare = bare[: -len(".base_layer")] if bare.endswith(".base_layer") else bare
                if not torch.equal(mod.dequantize(), ref[bare]):
                    bad += 1
        ok6 = ok6 and bad == 0
        print(f"  [{'OK  ' if ok6 else 'FAIL'}] every swapped base rebuilds its OWN weight "
              f"({len(got)} modules, {bad} mismatched)")
        fail += not ok6
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nPASS — the packed QA-LoRA base is storage-only." if not fail
          else f"\nFAIL — {fail} check(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
