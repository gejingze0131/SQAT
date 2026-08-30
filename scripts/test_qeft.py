#!/usr/bin/env python
"""
QEFT correctness tests (CPU, seconds). Run after touching src/gptq.py, the permutation code or
anything under baseline/QEFT/sqat/.

    python scripts/test_qeft.py

Covers, in order:
  1. src/gptq._front_permutation           — identity on a contiguous prefix, valid otherwise
  2. gptq_quantize_model_sequential        — `salient_ids` leaves an IRREGULAR column set exactly
                                             untouched and quantizes everything else, and does not
                                             change behaviour for callers that do not pass it
  3. select_global_weak_columns            — Alg. 1's per-layer normalization and global top-k
  4. OGR                                   — the single global permutation is output-preserving
  5. QEFTLinear                            — effective weight and forward agree at init and after
                                             an update; only the weak columns are trainable
  6. checkpoint I/O                        — trained columns round-trip, and a mismatched base is
                                             refused rather than silently loaded
  7. build_qeft_base + export              — the whole chain on a 2-layer random Llama
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baseline", "QEFT", "sqat"))

from src.gptq import _front_permutation, gptq_quantize_model_sequential  # noqa: E402
from src.permute_common import (apply_segment_permutation_fp32, _build_segment_perm)  # noqa: E402

from qeft_common import (QEFTLinear, build_qeft_base, build_qeft_model,  # noqa: E402
                         install_qeft_layers, load_qeft_trainable, qeft_layers,
                         save_qeft_trainable, select_global_weak_columns,
                         select_local_weak_columns)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def tiny_llama(hidden=64, inter=128, layers=2, heads=4, vocab=128, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=heads, vocab_size=vocab,
                      max_position_embeddings=64, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).to(torch.float32).eval()


def batches(n=4, bs=2, seq=16, vocab=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, vocab, (bs, seq), generator=g),
             "attention_mask": torch.ones(bs, seq, dtype=torch.long)} for _ in range(n)]


# ---------------------------------------------------------------------------
print("\n1) _front_permutation")
p = _front_permutation([0, 1, 2, 3], 8, torch.device("cpu"))
check("contiguous prefix -> identity", torch.equal(p, torch.arange(8)))
p = _front_permutation([5, 1], 6, torch.device("cpu"))
check("irregular set -> [ids sorted, rest in order]", p.tolist() == [1, 5, 0, 2, 3, 4])
check("permutation is a bijection", sorted(p.tolist()) == list(range(6)))
p2 = _front_permutation([1, 5], 6, torch.device("cpu"))
check("order of the ids does not matter", torch.equal(p, p2))


# ---------------------------------------------------------------------------
print("\n2) gptq_quantize_model_sequential(salient_ids=...)")
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model = tiny_llama()
ref = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(".weight")}
# an irregular set for every o_proj, a contiguous prefix everywhere else (group_k)
sal = {n: sorted(torch.randperm(64)[:16].tolist())
       for n, m in model.named_modules() if n.endswith("o_proj")}
gptq_quantize_model_sequential(
    model, batches(), targets, perm_group_k=16, group_size=16, q_bits=3, symmetric=False,
    device=torch.device("cpu"), perm_meta=None, nsamples=8, keep_salient_fp16=True,
    salient_ids=sal)

drift, quantized = [], []
for n, q in model.named_parameters():
    if n not in ref or n.split(".")[-2] not in targets:
        continue
    mod = n[: -len(".weight")]
    ids = torch.tensor(sal[mod] if mod in sal else list(range(16)))
    mask = torch.ones(ref[n].shape[1], dtype=torch.bool)
    mask[ids] = False
    drift.append((ref[n][:, ids] - q[:, ids]).abs().max().item())
    quantized.append((ref[n][:, mask] - q[:, mask]).abs().max().item())
check("weak columns (irregular AND contiguous) are bit-identical", max(drift) == 0.0,
      f"max drift {max(drift):.2e}")
check("every other column was quantized", min(quantized) > 0.0,
      f"min change {min(quantized):.2e}")

# the same call WITHOUT salient_ids must reproduce the old behaviour exactly
m1, m2 = tiny_llama(), tiny_llama()
for m in (m1, m2):
    gptq_quantize_model_sequential(
        m, batches(), targets, perm_group_k=16, group_size=16, q_bits=3, symmetric=False,
        device=torch.device("cpu"), perm_meta=None, nsamples=8, keep_salient_fp16=True,
        **({"salient_ids": None} if m is m2 else {}))
same = all(torch.equal(a, b) for (_, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()))
check("salient_ids=None is byte-identical to omitting it", same)


# ---------------------------------------------------------------------------
print("\n3) select_global_weak_columns")
d, L = 16, 3
sm = {}
for l in range(L):
    for src in ("attn", "mlp"):
        v = torch.full((d,), 1.0)
        v[3] = 100.0                      # every layer agrees on channel 3
        sm[(l, src)] = v
sm[(0, "attn")] = sm[(0, "attn")].clone()
sm[(0, "attn")][7] = 1e6                  # one layer screams about channel 7...
sm[(1, "attn")] = sm[(1, "attn")].clone()
sm[(1, "attn")][9] = 40.0                 # ...two agree, more weakly, about 9
sm[(2, "attn")] = sm[(2, "attn")].clone()
sm[(2, "attn")][9] = 40.0
weak = select_global_weak_columns(sm, d, L, k=3)
check("the channel every layer agrees on is selected", 3 in weak)
check("a single layer's outlier cannot dominate the vote", 7 in weak and 9 in weak,
      f"selected {weak} (per-layer mean normalization caps 7's contribution)")
check("returns exactly k sorted indices", len(weak) == 3 and weak == sorted(weak))
check("local selection is a plain top-k",
      select_local_weak_columns(torch.tensor([1.0, 9.0, 3.0, 7.0]), 2) == [1, 3])


# ---------------------------------------------------------------------------
print("\n4) OGR is output-preserving")
model = tiny_llama()
x = torch.randint(0, 128, (2, 16))
with torch.no_grad():
    y0 = model(input_ids=x).logits.clone()
perm = _build_segment_perm(sorted(torch.randperm(64)[:16].tolist()), 64)
bp = apply_segment_permutation_fp32(model, {0: perm}, [model.config.num_hidden_layers])
check("a single segment needs no runtime gather", bp == [])
with torch.no_grad():
    y1 = model(input_ids=x).logits
check("logits unchanged by the global reordering", torch.allclose(y0, y1, atol=1e-4),
      f"max |Δ| {(y0 - y1).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
print("\n5) QEFTLinear")
base = nn.Linear(32, 8, bias=False)
lin = QEFTLinear(base, list(range(4)))
check("contiguous weak columns take the slice path", lin.contiguous_prefix)
check("effective weight == base weight at init",
      torch.equal(lin.effective_weight(), base.weight.data))
xin = torch.randn(3, 32)
check("forward == dense forward at init",
      torch.allclose(lin(xin), base(xin), atol=1e-5))
with torch.no_grad():
    lin.weight_weak.add_(0.5)
check("effective weight tracks the trained columns",
      torch.allclose(lin.effective_weight()[:, :4], base.weight.data[:, :4] + 0.5) and
      torch.equal(lin.effective_weight()[:, 4:], base.weight.data[:, 4:]))
check("forward == dense forward after an update",
      torch.allclose(lin(xin), torch.nn.functional.linear(xin, lin.effective_weight()), atol=1e-5))

lin_irr = QEFTLinear(base, [31, 2, 17])
check("irregular weak columns take the gather path", not lin_irr.contiguous_prefix)
check("irregular: forward == dense forward",
      torch.allclose(lin_irr(xin), base(xin), atol=1e-5))
check("only the weak columns are trainable",
      [n for n, p in lin.named_parameters() if p.requires_grad] == ["weight_weak"])
check("the frozen half really is zeroed where the weak columns are",
      float(lin.weight[:, :4].abs().max()) == 0.0)


# ---------------------------------------------------------------------------
print("\n6/7) base build → model → train step → checkpoint → export (2-layer random Llama)")
with tempfile.TemporaryDirectory() as tmp:
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    src_dir = os.path.join(tmp, "src_model")
    m = tiny_llama(hidden=64, inter=128, layers=2, heads=4, vocab=128)
    m.save_pretrained(src_dir)

    class _Tok:                                    # save_pretrained is all the builder needs
        def save_pretrained(self, d):
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "tokenizer_stub.txt"), "w").write("stub")

    base_dir = os.path.join(tmp, "base")
    meta = build_qeft_base(src_dir, _Tok(), batches(n=6), base_dir,
                           k=16, group_size=16, q_bits=3, symmetric=False,
                           targets=targets, oproj_weak=True, nsamples=12,
                           dtype=torch.float32, device=torch.device("cpu"))
    check("meta records one global weak set", len(meta["global_weak_columns"]) == 16)
    check("o_proj got its own per-layer irregular set", len(meta["oproj_weak_ids"]) == 2)
    check("fp16 share and effective bits are recorded",
          0 < meta["fp16_share"] < 1 and meta["q_bits"] < meta["effective_bits"] < 16,
          f"{meta['fp16_share'] * 100:.1f}% fp16 → {meta['effective_bits']:.2f} bits")

    model, meta2 = build_qeft_model(base_dir, dtype=torch.float32, weak_dtype=torch.float32)
    layers = qeft_layers(model)
    check("every target linear became a QEFTLinear", len(layers) == 2 * len(targets))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_weak = sum(m_.weight_weak.numel() for m_ in layers.values())
    check("the weak columns are the only trainable tensors", n_train == n_weak,
          f"{n_train} == {n_weak}")

    # the base on disk must be exactly what the layers reconstruct
    base_model = LlamaForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    base_mods = dict(base_model.named_modules())
    worst = max((layers[n].effective_weight() - base_mods[n].weight.data).abs().max().item()
                for n in layers)
    check("effective weight == the base checkpoint, exactly", worst == 0.0, f"max |Δ| {worst:.2e}")

    # one optimizer step must move ONLY the weak columns
    frozen_before = {n: m_.weight.detach().clone() for n, m_ in layers.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.randint(0, 128, (2, 16))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [m_.weight_weak.grad for m_ in layers.values()]
    check("weak columns receive gradient", all(g is not None and g.abs().sum() > 0 for g in grads))
    opt.step()
    check("the frozen bulk did not move",
          all(torch.equal(frozen_before[n], m_.weight) for n, m_ in layers.items()))

    ckpt = os.path.join(tmp, "ckpt")
    save_qeft_trainable(model, ckpt, base_dir=base_dir)
    trained = {n: m_.weight_weak.detach().clone() for n, m_ in layers.items()}
    model2, _ = build_qeft_model(base_dir, dtype=torch.float32, weak_dtype=torch.float32)
    load_qeft_trainable(model2, ckpt)
    l2 = qeft_layers(model2)
    check("checkpoint round-trips the trained columns",
          all(torch.equal(trained[n], l2[n].weight_weak) for n in trained))
    check("checkpoint holds only the weak columns",
          os.path.getsize(os.path.join(ckpt, "qeft_weak_columns.safetensors"))
          < sum(p.numel() * 4 for p in base_model.parameters()) / 4)

    # a checkpoint from a different k must not load silently
    other, _ = build_qeft_model(base_dir, dtype=torch.float32)
    for m_ in qeft_layers(other).values():
        m_.weight_weak = nn.Parameter(m_.weight_weak[:, :8].clone())
    bad = os.path.join(tmp, "bad")
    save_qeft_trainable(other, bad, base_dir=base_dir)
    try:
        load_qeft_trainable(model2, bad)
        check("a shape-mismatched checkpoint is refused", False)
    except ValueError:
        check("a shape-mismatched checkpoint is refused", True)

    # export: scatter, then compare against the live modules
    exported = {n: m_.effective_weight() for n, m_ in l2.items()}
    for n, w in exported.items():
        base_mods[n].weight.data.copy_(w)
    xb = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        y_train = model2(input_ids=xb).logits
        y_export = base_model(input_ids=xb).logits
    check("the exported dense model reproduces the trained model",
          torch.allclose(y_train, y_export, atol=1e-4),
          f"max |Δ| {(y_train - y_export).abs().max().item():.2e}")

print(f"\n{'=' * 70}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
raise SystemExit(1 if FAIL else 0)
