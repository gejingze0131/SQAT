#!/usr/bin/env python
"""
gptq_quantize_model_sequential(stream_layers=True) must be NUMERICALLY IDENTICAL to the resident
path it replaces.

WHY THIS TEST EXISTS. Streaming was added so the GPTQ floor can be built for Qwen2.5-32B: 65 GB
in fp16 against a 48 GB card, while one decoder layer is ~1 GB. The change is supposed to be pure
placement — the loop already kept every activation on CPU and touched one layer per iteration —
so the ONLY acceptable outcome is bit-identical weights. Anything else means the streaming path
quietly changed the quantizer, and a floor built with it would not be comparable to the 7B rows
that results_saltq.csv already holds.

Checks, on a small random Qwen2 (real decoder layers, real forward, tiny dims):
  1. streamed weights == resident weights, bitwise, for every target projection;
  2. cross-layer propagation still happens (layer L's Hessian sees layer L-1's QUANTIZED output),
     verified by asserting the two paths agree on a model deep enough for it to matter;
  3. rel_err_out reports a real, non-zero relative error per module and agrees with a directly
     computed one — it is what replaced a 124 GB fp32 snapshot of every target weight;
  4. the model comes back wholly on CPU, so the caller can save_pretrained it;
  5. peak GPU memory really is one-layer-sized — the whole point. A streaming path that is
     numerically perfect but still hoists the stack would pass 1-4 and fail at 32B.

Run:  python scripts/test_gptq_stream.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.gptq import gptq_quantize_model_sequential

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _make_model(seed=0):
    """A real Qwen2 stack, small enough to hold two copies at once. GQA and a non-square MLP are
    kept because that is where the per-layer Hessian sizes differ (down_proj is the big one)."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=512, hidden_size=64, intermediate_size=192, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval().to(torch.float32)


def _make_loader(n_batches=4, bsz=2, seqlen=24, vocab=512, seed=1):
    """Padded batches, so the masked-Hessian path is exercised too."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n_batches):
        ids = torch.randint(0, vocab, (bsz, seqlen), generator=g)
        am = torch.ones(bsz, seqlen, dtype=torch.long)
        am[:, -3:] = 0                                   # right padding, excluded from Hessians
        rows.append({"input_ids": ids, "attention_mask": am})
    return DataLoader(rows, batch_size=None, shuffle=False)


def _target_weights(model):
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if n.endswith(".weight") and n.split(".")[-2] in TARGETS}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}")

    base = _make_model()
    W0 = _target_weights(base)

    kw = dict(target_terminals=TARGETS, perm_group_k=0, group_size=32, q_bits=2,
              symmetric=False, perm_meta=None, percdamp=0.01, blocksize=128, nsamples=10**9)

    # --- resident path (what every pre-existing caller does) --------------------------------
    resident = copy.deepcopy(base).to(device)
    gptq_quantize_model_sequential(resident, _make_loader(), device=device,
                                   stream_layers=False, **kw)
    W_res = _target_weights(resident)
    del resident
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- streamed path (model stays on CPU) --------------------------------------------------
    streamed = copy.deepcopy(base)                       # deliberately NOT moved to device
    rel_err = {}
    gptq_quantize_model_sequential(streamed, _make_loader(), device=device,
                                   stream_layers=True, rel_err_out=rel_err, **kw)
    W_str = _target_weights(streamed)

    fail = 0

    # 1 + 2. bitwise agreement, over every projection of every layer
    assert set(W_res) == set(W_str), "the two paths quantized different module sets"
    worst_name, worst = None, 0.0
    for n in sorted(W_res):
        d = (W_res[n] - W_str[n]).abs().max().item()
        if d > worst:
            worst_name, worst = n, d
    ok = worst == 0.0
    print(f"  [{'OK  ' if ok else 'FAIL'}] streamed == resident, bitwise, over {len(W_res)} "
          f"projections (worst |Δ| = {worst:g} at {worst_name})")
    fail += not ok

    # a quantizer that never ran would also be "identical"; make sure both actually moved
    moved = min((W0[n] - W_res[n]).norm().item() / W0[n].norm().item() for n in W0)
    ok = moved > 1e-3
    print(f"  [{'OK  ' if ok else 'FAIL'}] both paths actually quantized "
          f"(smallest relative weight change = {moved:.4f})")
    fail += not ok

    # 3. rel_err_out is complete and matches a directly computed error.
    # rel_err is keyed by MODULE name, W0 by PARAMETER name — the ".weight" suffix is the only
    # difference, and comparing the two sets raw is how this check first went red.
    W0_by_module = {n[: -len(".weight")]: w for n, w in W0.items()}
    Wstr_by_module = {n[: -len(".weight")]: w for n, w in W_str.items()}
    ok = set(rel_err) == set(W0_by_module)
    print(f"  [{'OK  ' if ok else 'FAIL'}] rel_err_out covers every target projection "
          f"({len(rel_err)} of {len(W0_by_module)})")
    fail += not ok
    if ok:
        worst = max(abs(rel_err[n] - (W0_by_module[n] - Wstr_by_module[n]).norm().item()
                        / W0_by_module[n].norm().item()) for n in W0_by_module)
        ok2 = worst < 1e-5 and min(rel_err.values()) > 1e-3
        print(f"  [{'OK  ' if ok2 else 'FAIL'}] rel_err_out matches a direct recomputation "
              f"(worst |Δ| = {worst:.2e}, min error = {min(rel_err.values()):.4f})")
        fail += not ok2

    # 4. nothing left on the GPU
    devs = {p.device.type for p in streamed.parameters()} | {b.device.type for b in streamed.buffers()}
    ok = devs == {"cpu"}
    print(f"  [{'OK  ' if ok else 'FAIL'}] model returned wholly on CPU (devices seen: {devs})")
    fail += not ok

    # 5. peak GPU memory does not grow with DEPTH — the property that makes 32B fit.
    # Not "peak < one layer": the per-layer working set is dominated by the Hessians, which are
    # in_features² and identical at any depth, so on a small model it swamps the stack itself.
    # Doubling the layer count and seeing the same peak is the unambiguous statement.
    if torch.cuda.is_available():
        def _peak_at(n_layers):
            torch.manual_seed(2)
            cfg = Qwen2Config(vocab_size=512, hidden_size=1024, intermediate_size=2816,
                              num_hidden_layers=n_layers, num_attention_heads=16,
                              num_key_value_heads=4, max_position_embeddings=128,
                              tie_word_embeddings=False)
            m = Qwen2ForCausalLM(cfg).eval().to(torch.float32)
            stack = sum(p.numel() * p.element_size() for p in m.model.layers.parameters())
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gptq_quantize_model_sequential(m, _make_loader(n_batches=2, seqlen=32), device=device,
                                           stream_layers=True, **kw)
            peak = torch.cuda.max_memory_allocated()
            del m
            torch.cuda.empty_cache()
            return peak, stack

        p_small, s_small = _peak_at(6)
        p_big,   s_big   = _peak_at(24)
        # 4x the stack. A resident path would grow with it; a streaming one must not.
        ok = p_big < 1.15 * p_small
        print(f"  [{'OK  ' if ok else 'FAIL'}] peak GPU is depth-independent: "
              f"{p_small/2**20:.0f} MiB at 6 layers ({s_small/2**20:.0f} MiB stack) -> "
              f"{p_big/2**20:.0f} MiB at 24 ({s_big/2**20:.0f} MiB stack), "
              f"{p_big/p_small:.2f}x peak for {s_big/s_small:.0f}x depth")
        fail += not ok
    else:
        print("  [SKIP] peak-memory check needs CUDA")

    print("\nPASS — streaming is placement-only." if not fail else f"\nFAIL — {fail} check(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
