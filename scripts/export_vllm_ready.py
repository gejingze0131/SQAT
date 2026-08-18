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
        print(f"[vLLM-export] {args.model_path} has no {PERM_META_FILENAME}; copying as-is.")
        for name in os.listdir(args.model_path):
            src = os.path.join(args.model_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
        print(f"[vLLM-export] -> {args.output_dir}")
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

    print("[vLLM-export] Done. This checkpoint is a plain Llama and needs no runtime hook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
