#!/usr/bin/env python
"""
Turn an exported eval model into one an EXTERNAL runtime can load.

SALT-Q and SQAT-Permute exports are dense HF checkpoints, but they are not self-contained:
their residual stream is permuted per segment, so a correct forward pass needs a
BoundaryGatherHook after the last layer of every non-final segment. That hook lives in this
repo's eval glue (permute_common.maybe_build_gather_aware_hflm). vLLM has no equivalent — it
loads the checkpoint as a plain Llama and produces fluent, badly wrong output, with nothing
in the logs to say so.

This script folds the permutation back into the weights (exact reindex, verified to
max|Δlogits| == 0 by scripts/test_unpermute_fold.py) and writes a checkpoint with NO
sqat_permute_meta.pt — so nothing downstream re-registers a gather that is now the identity.

Checkpoints without perm metadata (QLoRA / QA-LoRA merged or dequant exports) need none of
this; the script copies them through so the eval pipeline can call it unconditionally.

  python scripts/export_vllm_ready.py \
      --model_path outputs/saltq-3bit-saltq-deploy-eval \
      --output_dir outputs/saltq-3bit-saltq-deploy-vllm
"""

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.permute_common import (
    PERM_META_FILENAME,
    fold_boundary_gathers_into_weights,
    load_perm_meta,
)



def _sanitize_tokenizer_config(out_dir):
    """Make the exported tokenizer config loadable by the EVAL env's transformers.

    The two envs are pinned apart on purpose — vLLM pins its own torch — and they exchange a
    plain HF checkpoint on disk. That works right up until save_pretrained starts writing a key
    the other side reads differently. The train env (transformers 5.14) writes
    `extra_special_tokens` as a LIST of token strings; the eval env (4.57) does
    `special_tokens.keys()` on it and dies with AttributeError before vLLM loads a single weight:

        tokenization_utils_base.py: SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
        AttributeError: 'list' object has no attribute 'keys'

    Qwen2.5's own upstream tokenizer_config.json does not carry the key at all — those tokens
    live in added_tokens_decoder and tokenizer.json regardless — so dropping it restores
    upstream's shape rather than inventing one. Only a LIST is dropped; a dict is what 4.57
    expects and is left alone.

    Writes a NEW file rather than editing in place: the copy-through above hardlinks, so an
    in-place edit would reach back into the source checkpoint through the shared inode.
    """
    import json

    path = os.path.join(out_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return
    dropped = cfg.pop("extra_special_tokens")
    os.remove(path)                       # break the hardlink before writing
    with open(path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[vLLM-export] dropped list-valued extra_special_tokens ({len(dropped)} entries) from "
          f"tokenizer_config.json — the eval env's transformers reads that key as a dict")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if --output_dir already holds a folded model.")
    args = parser.parse_args()

    meta_path = os.path.join(args.model_path, PERM_META_FILENAME)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(meta_path):
        # Nothing to fold. Mirror the checkpoint so the caller gets one predictable path.
        # HARDLINK, not copy. There is nothing to fold here, so every byte written would be a
        # second identical copy of the checkpoint — 65 GB per export at Qwen2.5-32B, and a cell
        # exports three (merged fp16 / RTN dequant / GPTQ floor). Hardlinks are free, vLLM only
        # ever reads these files, and deleting either directory just drops a link. Falls back to
        # a real copy when the two paths are not on the same filesystem.
        print(f"[vLLM-export] {args.model_path} has no {PERM_META_FILENAME}; linking as-is.")
        linked = copied = 0
        for name in os.listdir(args.model_path):
            src = os.path.join(args.model_path, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(args.output_dir, name)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
        _sanitize_tokenizer_config(args.output_dir)
        print(f"[vLLM-export] -> {args.output_dir} ({linked} hardlinked, {copied} copied)")
        return 0

    done_marker = os.path.join(args.output_dir, "config.json")
    if os.path.exists(done_marker) and not args.force:
        print(f"[vLLM-export] {args.output_dir} already exists; pass --force to rebuild.")
        return 0

    dtype = getattr(torch, args.dtype)
    print(f"[vLLM-export] Loading {args.model_path} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    fold_boundary_gathers_into_weights(model, load_perm_meta(meta_path))

    print(f"[vLLM-export] Saving -> {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Deliberately NOT copied: sqat_permute_meta.pt. Its presence is the signal every reader in
    # this repo uses to decide "this model needs boundary gathers", and after the fold it does
    # not — registering them here would re-break the model.
    stale = os.path.join(args.output_dir, PERM_META_FILENAME)
    if os.path.exists(stale):
        os.remove(stale)

    # Same cross-env tokenizer fix as the copy-through path above: this save_pretrained runs in
    # the TRAIN env too, so it writes the same key the eval env cannot read.
    _sanitize_tokenizer_config(args.output_dir)

    print("[vLLM-export] Done. This checkpoint is a plain Llama and needs no runtime hook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
