#!/usr/bin/env python
"""
End-to-end SALT-Q pipeline smoke test on a tiny randomly-initialized Llama — CPU-only, no
downloads, runs in well under a minute.

It walks the ENTIRE offline + training + export path exactly as scripts/train.py does:

    build_permuted_fp16_checkpoint   permute salient channels to the front (shared with sqat_permute)
      -> build_saltq_base            GPTQ -> FROZEN codes + initial (s, z) + salient weight init
        -> build_saltq_model         swap linears for SALTQLinear
          -> SALTQ.prepare_model     register the runtime boundary gathers
            -> fwd/bwd/optimizer step
              -> export_saltq        deployed == trained, exactly

What it is actually guarding (things a pure unit test cannot see):
  * the per-layer group_k resolved from perm_meta lines up with the codes/qparams shapes;
  * the boundary gathers are registered and the permuted model still runs;
  * gradients reach the SALT-Q params through a real decoder stack with gradient checkpointing;
  * a checkpoint written by save_saltq_trainable reloads onto a freshly built model, and the
    reloaded model is bit-identical to the one that saved it;
  * export produces a loadable HF model whose logits match the training model's.

Run:  python scripts/test_saltq_e2e.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.permute_common import build_permuted_fp16_checkpoint, load_perm_meta
from src.qat_saltq import (
    SALTQ,
    build_saltq_base,
    build_saltq_model,
    export_saltq,
    load_saltq_trainable,
    saltq_layers,
    save_saltq_trainable,
    verify_saltq_deploy_equivalence,
)

PASSED = 0
FAILED = 0

GROUP_SIZE = 16
GROUP_K = 32
Q_BITS = 2
SYMMETRIC = False
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def check(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {msg}")
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}")


class _StubTokenizer:
    """build_permuted_fp16_checkpoint / build_saltq_base only ever call save_pretrained."""

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_stub.txt"), "w") as f:
            f.write("stub\n")


def make_tiny_model(path):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path)
    return cfg


def make_loader(n=8, seqlen=32, vocab=256, bs=2):
    torch.manual_seed(1)
    data = [
        {"input_ids": torch.randint(0, vocab, (seqlen,)),
         "attention_mask": torch.ones(seqlen, dtype=torch.long)}
        for _ in range(n)
    ]

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    return DataLoader(data, batch_size=bs, collate_fn=collate, shuffle=False)


def fake_cfg(out_dir):
    return {
        "model": {"name": "tiny", "quant_bits": Q_BITS, "dtype": "float32"},
        "qat": {"mode": "saltq", "symmetric": SYMMETRIC, "group_size": GROUP_SIZE,
                "saltq": {"train_layernorms": False, "train_scale": False}},
        "lora": {"target_modules": TARGETS, "rank": 8, "alpha": 16},
        "training": {"output_dir": out_dir},
    }


def main():
    print("=" * 68)
    print("  SALT-Q end-to-end pipeline (tiny Llama, CPU)")
    print("=" * 68)

    work = tempfile.mkdtemp(prefix="saltq_e2e_")
    try:
        base_dir = os.path.join(work, "base")
        perm_dir = os.path.join(work, "permuted_fp16_base")
        saltq_dir = os.path.join(work, "saltq_base")
        make_tiny_model(base_dir)
        tok = _StubTokenizer()
        loader = make_loader()

        # ---- stage 1: permute (shared with sqat_permute) ----
        print("\n[stage 1] build_permuted_fp16_checkpoint")
        build_permuted_fp16_checkpoint(
            model_name=base_dir, tokenizer=tok, calibration_dataloader=loader,
            boundary_sizes=[1, 3], save_dir=perm_dir, target_modules=TARGETS,
            group_k=GROUP_K, group_size=GROUP_SIZE, top_k_ratio=0.1,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        perm_meta = load_perm_meta(perm_dir)
        check(len(perm_meta["boundary_perms"]) == 1, "one runtime boundary gather (2 segments)")

        # ---- stage 2: freeze the codes ----
        print("\n[stage 2] build_saltq_base (GPTQ -> frozen codes)")
        meta = build_saltq_base(
            permuted_base_dir=perm_dir, perm_meta=perm_meta, tokenizer=tok,
            calibration_dataloader=loader, target_terminals=TARGETS,
            group_size=GROUP_SIZE, q_bits=Q_BITS, symmetric=SYMMETRIC,
            save_dir=saltq_dir, device=torch.device("cpu"), nsamples=8, blocksize=32,
            dtype=torch.float32,
        )
        pc = meta["param_counts"]
        check(pc["frozen_codes"] > 0 and pc["trainable_salient_weights"] > 0
              and pc["trainable_qparams"] > 0, "all three freedom tiers populated")
        check(len(meta["layers"]) == 4 * len(TARGETS),
              f"{len(meta['layers'])} SALT-Q layers recorded in meta")
        o_gks = [v["group_k"] for k, v in meta["layers"].items() if k.endswith("o_proj")]
        check(all(g == 0 for g in o_gks), "o_proj has group_k=0 (pure affine-freedom layer)")

        # ---- stage 3: assemble + prepare ----
        print("\n[stage 3] build_saltq_model + prepare_model")
        model, meta2 = build_saltq_model(saltq_dir, dtype=torch.float32,
                                         gradient_checkpointing=True)
        handler = SALTQ()
        model = handler.prepare_model(model, fake_cfg(work), saltq_meta=meta2,
                                      saltq_base_dir=saltq_dir)
        layers = saltq_layers(model)
        check(len(layers) == 4 * len(TARGETS), f"{len(layers)} SALTQLinear installed")
        check(len(handler.boundary_hooks) == 1, "boundary gather registered on the live model")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen_float = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        check(n_train > 0, f"trainable params = {n_train}")
        check(all(m._codes_cpu.device.type == "cpu" and not m._codes_cpu.requires_grad
                  for m in layers.values()),
              f"codes stay on the host with no grad "
              f"({n_frozen_float} frozen float params remain: embed/norm/head)")
        check(all(hasattr(m, "wq") and not m.wq.requires_grad for m in layers.values()),
              "q*s precomputed as a frozen device buffer (z-only forward, no reconstruction)")

        # ---- stage 4: train a step ----
        print("\n[stage 4] forward / backward / optimizer step")
        model.train()
        ids = torch.randint(0, 256, (2, 24))
        out = model(input_ids=ids, labels=ids)
        check(torch.isfinite(out.loss).item(), f"loss finite: {out.loss.item():.4f}")
        out.loss.backward()
        got_grad = [n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
        check(len(got_grad) >= 4 * len(TARGETS),
              f"{len(got_grad)}/{sum(1 for p in model.parameters() if p.requires_grad)} "
              f"trainable tensors received a non-zero gradient")

        codes_before = {n: m._codes_cpu.clone() for n, m in layers.items()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        opt.step()
        check(all(torch.equal(layers[n]._codes_cpu, c) for n, c in codes_before.items()),
              "frozen codes untouched by the optimizer step")

        # ---- stage 5: deploy equivalence AFTER training ----
        print("\n[stage 5] deployed == trained (exact)")
        model.eval()
        verify_saltq_deploy_equivalence(model, max_layers=len(layers))
        check(True, "every layer: max|Δ| == 0 between deployed and training weight")

        # ---- stage 6: checkpoint round-trip ----
        print("\n[stage 6] trainable-only checkpoint round-trip")
        ckpt = os.path.join(work, "ckpt")
        save_saltq_trainable(model, ckpt, saltq_base_dir=saltq_dir)
        files = os.listdir(ckpt)
        ck_bytes = os.path.getsize(os.path.join(ckpt, "saltq_trainable.safetensors"))
        base_bytes = os.path.getsize(os.path.join(saltq_dir, "saltq_base.safetensors"))
        check("saltq_trainable.safetensors" in files, f"checkpoint files: {sorted(files)}")
        check(ck_bytes < base_bytes,
              f"checkpoint ({ck_bytes / 1e3:.0f} kB) < frozen base ({base_bytes / 1e3:.0f} kB) "
              f"— codes are not duplicated per checkpoint")

        with torch.no_grad():
            ref = model(input_ids=ids).logits.clone()
        model2, _ = build_saltq_model(saltq_dir, dtype=torch.float32,
                                      gradient_checkpointing=False)
        SALTQ().prepare_model(model2, fake_cfg(work), saltq_meta=meta2, saltq_base_dir=saltq_dir)
        load_saltq_trainable(model2, ckpt)
        model2.eval()
        with torch.no_grad():
            got = model2(input_ids=ids).logits
        err = (ref - got).abs().max().item()
        check(err == 0.0, f"reloaded checkpoint reproduces logits exactly, max|Δ|={err:.3e}")

        # ---- stage 7: export ----
        print("\n[stage 7] export_saltq (merge-free)")
        cfg = fake_cfg(work)
        exp_dir = os.path.join(work, "exported")
        export_saltq(cfg=cfg, tokenizer=tok, saltq_base_dir=saltq_dir, model=model2,
                     output_dir=exp_dir, dequant_dtype="float32")
        check(os.path.exists(os.path.join(exp_dir, "sqat_permute_meta.pt")),
              "boundary-gather metadata shipped with the export (eval needs it)")

        from transformers import AutoModelForCausalLM
        from src.permute_common import register_boundary_gathers_from_meta

        exported = AutoModelForCausalLM.from_pretrained(exp_dir, dtype=torch.float32)
        register_boundary_gathers_from_meta(
            exported, os.path.join(exp_dir, "sqat_permute_meta.pt"))
        exported.eval()
        with torch.no_grad():
            got = exported(input_ids=ids).logits
        err = (ref - got).abs().max().item()
        # fp32 dense storage keeps the deployed values exactly; the only reason this is not 0.0
        # is non-deterministic GEMM reduction order, hence the tiny tolerance.
        check(err < 1e-4,
              f"exported model reproduces the training model's logits, max|Δ|={err:.3e}")

        print("\n" + "=" * 68)
        print(f"  SALT-Q e2e: {PASSED} passed, {FAILED} failed")
        print("=" * 68)
        return 1 if FAILED else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
