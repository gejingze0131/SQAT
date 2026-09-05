#!/usr/bin/env python
"""
build_permuted_fp16_checkpoint(shard_across_gpus=True) must produce a BIT-IDENTICAL permuted base.

WHY. The GPTQ sweep could be made to stream one decoder layer at a time because it only ever
touches one layer; the calibration here is a real forward pass through the whole model, so the
only way to fit Qwen2.5-32B (65 GB fp16) on 3x46 GiB is accelerate's device_map. That is a
placement change and nothing else should move with it — every SALT-Q result is downstream of
which channels this step calls salient, so a permuted base that differs by even one index makes
the 32B numbers incomparable to the 7B rows for a reason that has nothing to do with the method.

Two things this guards that reasoning alone does not settle:
  * the E[x^2] hooks accumulate in fp32 ON CPU, so per-layer accumulators must not depend on
    which card the layer landed on;
  * the three transforms are applied to weights that now live on different devices — the
    permutations index with CPU long tensors, the Hadamard and AWQ folds move their operands
    to the weight's device.

Observed on this box: accelerate spreads even the tiny model over cuda:0/1/2, so the
multi-device path is genuinely exercised (the run prints the placement).

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

        ok = str(meta_a.get("group_k")) == str(meta_b.get("group_k"))
        print(f"  [{'OK  ' if ok else 'FAIL'}] identical group_k ({meta_a.get('group_k')})")
        fail += not ok
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nPASS — sharding is placement-only." if not fail else f"\nFAIL — {fail} check(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
