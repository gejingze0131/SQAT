#!/usr/bin/env python
"""
Perplexity of a DEPLOYED checkpoint, in the GPTQ lineage's protocol.

This is the WikiText-2 column's scorer. It exists as a separate entry point from
runs/eval_vllm.sh's generative path because perplexity is a teacher-forced, all-token metric on
raw text, and nothing about the commonsense / math scorers applies: there is no prompt, no
answer extraction and no sampling.

THE PROTOCOL (identical to GPTQ / AWQ / OmniQuant / LoftQ / ApiQ, so the numbers can be read
against theirs):

  1. join every row of the split's raw text with "\n\n" into ONE string;
  2. tokenize it ONCE (one BOS at the front of the whole stream);
  3. cut into NON-OVERLAPPING windows of --seq_len, dropping the ragged tail;
  4. per window, shift by one and average cross-entropy over its tokens;
  5. ppl = exp( sum_w loss_w * seq_len / (n_windows * seq_len) ).

Step 5 is the lineage's own arithmetic: it weights each window by `seq_len` although the shift
scores only `seq_len - 1` tokens. It is reproduced EXACTLY rather than "corrected", because
every published WikiText-2 perplexity was produced with it.

MEASURED, not assumed: because every window is the SAME length, that weighting cancels — both
sum(loss_w * L) / (n * L) and sum(loss_w * (L-1)) / (n * (L-1)) reduce to mean(loss_w), so the
off-by-one is a no-op here and there is nothing to correct. `ppl_exact` computes the second form
and is therefore always equal to `ppl`; it is kept as a standing assertion of that, and would
diverge only if a ragged final window were ever admitted (step 3 drops it).

Anchor: this protocol puts unquantized fp16 Llama-2-7B at 5.4721 on wikitext-2-raw-v1 test at
seq_len 2048 (measured 2026-09-03, job 16072379), i.e. the published ~5.47. Any change to steps
1-5 must reproduce that number.

WHAT IS MEASURED: the deployed artifact, the same rule the main tables use. A SALT-Q / SQAT-
permute export is not a self-contained Llama — its residual stream is permuted per segment — so
the boundary gathers are registered here exactly as training and lm-eval register them. A model
carrying that metadata is REFUSED if they cannot be registered rather than scored wrong.

Usage:
  python scripts/eval_ppl.py --model_path outputs/<export> --data datasets/wikitext2 --split test
  python scripts/eval_ppl.py --model_path <dir> --adapter_path <peft_dir>        # adapter-mounted
  python scripts/eval_ppl.py --model_path <dir> --data datasets/c4_val_ppl.json  # the C4 control
  python scripts/eval_ppl.py --model_path <dir> --seq_len 2048 --tag mytag
"""

import argparse
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import lm_token_stream, lm_windows
from src.permute_common import PERM_META_FILENAME, register_boundary_gathers_from_meta


def load_model(model_path: str, dtype: torch.dtype, adapter_path: str | None, device):
    """The deployed artifact, with whatever it needs to run correctly — and nothing else."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)

    if adapter_path:
        # The QLoRA fp16 upper bound is evaluated in its ADAPTER-MOUNTED state: that is what it
        # actually ships, and merging it first is a different artifact.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=dtype)
        print(f"[ppl] adapter mounted from {adapter_path}")

    meta_path = os.path.join(model_path, PERM_META_FILENAME)
    if os.path.exists(meta_path):
        target = getattr(model, "base_model", model)
        hooks = register_boundary_gathers_from_meta(
            getattr(target, "model", target) if adapter_path else model, meta_path)
        if not hooks:
            raise RuntimeError(
                f"{model_path} carries {PERM_META_FILENAME} but no boundary gather was "
                f"registered; refusing to score a permuted model without them.")
        print(f"[ppl] registered {len(hooks)} boundary gather(s)")

    model.to(device).eval()
    return model


@torch.no_grad()
def perplexity(model, windows, device, log_every: int = 20):
    """Steps 4-5. One window at a time: the logits of a 2048-token window are already 250 MB."""
    n, seq_len = windows.shape
    total_nll_lineage = 0.0      # sum of loss_w * seq_len   (the GPTQ arithmetic)
    total_nll_exact = 0.0        # sum of loss_w * (seq_len - 1)
    for i in range(n):
        ids = torch.from_numpy(windows[i]).unsqueeze(0).to(device)
        logits = model(ids).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
        )
        total_nll_lineage += loss.item() * seq_len
        total_nll_exact += loss.item() * (seq_len - 1)
        if log_every and (i + 1) % log_every == 0:
            running = math.exp(total_nll_lineage / ((i + 1) * seq_len))
            print(f"[ppl]   {i + 1}/{n} windows  running ppl {running:.4f}", flush=True)
    return (math.exp(total_nll_lineage / (n * seq_len)),
            math.exp(total_nll_exact / (n * (seq_len - 1))))


def main() -> int:
    ap = argparse.ArgumentParser(description="GPTQ-protocol perplexity of a deployed checkpoint")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default=None,
                    help="PEFT adapter to mount (the QLoRA upper bound's deployed state)")
    ap.add_argument("--data", default="datasets/wikitext2",
                    help="dataset dir (uses <dir>/<split>.json) or a JSON file of raw texts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--seq_len", type=int, default=1024,
                    help="window length. 1024 is LoftQ/ApiQ's block_size (train_clm.sh) and this "
                         "repo's WikiText-2 default; 2048 is the GPTQ-paper length. PPL depends "
                         "on it, so it is recorded in the output JSON.")
    ap.add_argument("--also_seq_len", type=int, nargs="*", default=[],
                    help="extra window lengths to score in the same load (e.g. --also_seq_len 2048)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default=None, help="output file stem (default: basename of model_path)")
    ap.add_argument("--output_dir", default=None,
                    help="default: results/<dataset name>_ppl/")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer to use (default: the model dir; falls back to the base model)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    model_path = args.model_path.rstrip("/")
    if not os.path.isdir(model_path):
        print(f"ERROR: no such model dir: {model_path}", file=sys.stderr)
        return 2

    dataset_name = (os.path.splitext(os.path.basename(args.data.rstrip("/")))[0]
                    if os.path.isfile(args.data) else os.path.basename(args.data.rstrip("/")))
    tag = args.tag or os.path.basename(model_path)
    out_dir = args.output_dir or os.path.join("results", f"{dataset_name}_ppl")
    os.makedirs(out_dir, exist_ok=True)

    tok_src = args.tokenizer or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[ppl] model      {model_path}")
    print(f"[ppl] data       {args.data} split={args.split} (wikitext-2-raw-v1 lineage protocol)")
    print(f"[ppl] seq_len    {[args.seq_len] + list(args.also_seq_len)}")

    stream = lm_token_stream(args.data, args.split, tokenizer, text_field=args.text_field)
    print(f"[ppl] {stream.shape[0]} tokens in the concatenated {args.split} stream")

    model = load_model(model_path, dtype, args.adapter_path, device)

    results = {}
    for seq_len in [args.seq_len] + list(args.also_seq_len):
        windows = lm_windows(stream, seq_len)
        print(f"[ppl] seq_len={seq_len}: {windows.shape[0]} non-overlapping windows "
              f"({windows.size} tokens scored)")
        ppl, ppl_exact = perplexity(model, windows, device)
        results[f"seq_len_{seq_len}"] = {
            "ppl": ppl,
            "ppl_exact": ppl_exact,
            "n_windows": int(windows.shape[0]),
            "n_tokens": int(windows.size),
        }
        print(f"[ppl] seq_len={seq_len}  PPL {ppl:.4f}  (exact-denominator {ppl_exact:.4f})")

    payload = {
        "source": "ppl",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "model_path": model_path,
            "adapter_path": args.adapter_path or "",
            "dataset": dataset_name,
            "data_path": args.data,
            "split": args.split,
            "dtype": args.dtype,
            "protocol": "gptq-lineage: join('\\n\\n') -> tokenize once -> non-overlapping windows"
                        " -> all tokens, exp(sum(loss*L)/(n*L))",
            "variant": "wikitext-2-raw-v1" if dataset_name.startswith("wikitext") else dataset_name,
            "note": args.note,
        },
        "results": results,
    }
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[ppl] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
