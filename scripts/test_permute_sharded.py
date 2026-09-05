#!/usr/bin/env python
"""build_permuted_fp16_checkpoint(shard_across_gpus=True) vs the single-GPU path.

STATUS: THIS TEST FAILS ABOUT HALF THE TIME, AND THE FAILURE IS REAL. It is left failing on
purpose — it is a measurement of a known defect, not a flaky assertion to be relaxed.

WHY THE SHARDED PATH EXISTS. The GPTQ sweep could be made to stream one decoder layer at a
time because it only ever touches one layer; the calibration here is a real forward pass
through the whole model, so the only way to fit Qwen2.5-32B (65 GB fp16) on 3x46 GiB is
accelerate's device_map. That was supposed to be a placement change and nothing else.

WHAT WAS MEASURED (4-layer probe, two visible cards, boundary_sizes=[1, 3], group_k=32):

  6 sharded builds in ONE process against a single-GPU reference   6/6 bit-identical
  the standalone two-build test, run 10 times                      ~50% FAIL

  On the SegPerm log lines of one captured failure, the segment that stays on one card agreed
  and the segment device_map splits did not:
      seg 0 (L0 only)          union_topk 4 vs 4, energy_cov 29.6% vs 29.6%, first10 identical
      seg 1 (L1-L3, SPLIT)     union_topk 7 vs 6, energy_cov 34.0% vs 31.7%,
                               first10 [2,3,6,7,11,15,...] vs [0,1,2,6,18,24,...]

  The size of the disagreement is NOT small, and it varies from run to run. Measured on the
  stored boundary permutation, as channels swapped in or out of the 32-wide salient slice:
      run A   3 of 32   single-GPU only [38, 60, 90]   sharded only [18, 29, 39]
      run B  12 of 32   single-GPU only [19, 20, 35, 38, 41, 45, 60, 87, 91, 99, 106, 115]
                        sharded only    [6, 9, 15, 17, 18, 28, 32, 33, 58, 63, 80, 116]

So this is a genuine change of WHICH channels are called salient — up to 37% of the slice, not
a reordering within the same set, and not float noise in the weights (the 0.13 max |delta| is
whole channels landing in different columns). Do not read it as a tie-break detail.

The accumulator is not the cause: _collect_second_moments sums x^2 in float32 ON CPU, which is
order-independent given identical activations. The activations themselves differ with placement.

WHAT THIS DOES AND DOES NOT INVALIDATE. Every 32B row in results_saltq.csv is safe: the
permuted base is built ONCE, written to disk, and the frozen-code build, the training and the
export all read that one file, so each run is internally consistent with the base it actually
used. What is lost is the ability to REBUILD a byte-identical base later, and with it the
premise that a sharded base and a single-GPU base are interchangeable.

THE FIX, when someone takes it on: pin an explicit layer -> device map computed from the layer
count instead of device_map="auto" (which balances against free VRAM and therefore against
whatever else the box was doing), then re-measure. If that is not enough, the calibration
forward can be run with CPU offload, which is slower but placement-free.

Run:  python scripts/test_permute_sharded.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.permute_common import build_permuted_fp16_checkpoint

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_tiny_model(path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
                      max_position_embeddings=64)
    LlamaForCausalLM(cfg).to(torch.float16).save_pretrained(path)
    return path


def make_loader(n=8, seqlen=32, vocab=256, bs=2):
    g = torch.Generator().manual_seed(1)
    rows = [{"input_ids": torch.randint(0, vocab, (seqlen,), generator=g),
             "attention_mask": torch.ones(seqlen, dtype=torch.long)} for _ in range(n)]

    def collate(b):
        return {"input_ids": torch.stack([r["input_ids"] for r in b]),
                "attention_mask": torch.stack([r["attention_mask"] for r in b])}

    return DataLoader(rows, batch_size=bs, collate_fn=collate)


class _Tok:
    def save_pretrained(self, path):
        pass


def _state(d):
    from safetensors.torch import load_file
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update({k: v.cpu() for k, v in load_file(f).items()})
    return out


def main():
    if not torch.cuda.is_available():
        print("  [SKIP] needs CUDA"); return 0
    work = tempfile.mkdtemp(prefix="permsharded_")
    fail = 0
    try:
        src = make_tiny_model(os.path.join(work, "base"))
        kw = dict(tokenizer=_Tok(), boundary_sizes=[1, 3], target_modules=TARGETS,
                  group_k=32, group_size=32, max_segments=4, dtype=torch.float16)

        a = os.path.join(work, "single")
        meta_a = build_permuted_fp16_checkpoint(model_name=src, calibration_dataloader=make_loader(),
                                                save_dir=a, shard_across_gpus=False, **kw)
        # device_map="auto" balances even this tiny model across every visible card — the run
        # prints which ones, and the assertion below is only meaningful when that is >1.
        b = os.path.join(work, "sharded")
        meta_b = build_permuted_fp16_checkpoint(model_name=src, calibration_dataloader=make_loader(),
                                                save_dir=b, shard_across_gpus=True, **kw)

        sa, sb = _state(a), _state(b)
        ok = set(sa) == set(sb)
        print(f"  [{'OK  ' if ok else 'FAIL'}] same tensor set ({len(sa)} vs {len(sb)})")
        fail += not ok
        if ok:
            worst, wname = 0.0, None
            for k in sa:
                d = (sa[k].float() - sb[k].float()).abs().max().item()
                if d > worst:
                    worst, wname = d, k
            ok = worst == 0.0
            print(f"  [{'OK  ' if ok else 'FAIL'}] permuted weights bit-identical "
                  f"(worst |Δ| = {worst:g}{'' if wname is None else ' at ' + wname})")
            fail += not ok

        pa = [p.tolist() for p in meta_a["boundary_perms"]]
        pb = [p.tolist() for p in meta_b["boundary_perms"]]
        ok = pa == pb and meta_a["segment_perms"] == meta_b["segment_perms"]
        print(f"  [{'OK  ' if ok else 'FAIL'}] identical salient selection "
              f"(boundary_perms + segment_perms)")
        fail += not ok
        if not ok:
            # Say WHICH segment moved and by how much. A bare "not identical" cannot distinguish
            # a reordering inside the same salient set (harmless) from a different set (not), and
            # the whole point of the header's diagnosis is that it is the second one, in exactly
            # the segment device_map splits.
            for i, (x, y) in enumerate(zip(pa, pb)):
                sx, sy = set(x[:32]), set(y[:32])
                print(f"         seg {i}: salient set identical={sx == sy}  "
                      f"swapped in/out={len(sx - sy)} of 32  "
                      f"only single-GPU={sorted(sx - sy)}  only sharded={sorted(sy - sx)}")
            if meta_a["segment_perms"] != meta_b["segment_perms"]:
                print("         segment_perms differ too")

        ok = str(meta_a.get("group_k")) == str(meta_b.get("group_k"))
        print(f"  [{'OK  ' if ok else 'FAIL'}] identical group_k ({meta_a.get('group_k')})")
        fail += not ok
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nPASS — sharding is placement-only." if not fail
          else f"\nFAIL — {fail} check(s). EXPECTED ~50% of the time; see this file's header "
               f"for the measurement and what it does and does not invalidate.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
