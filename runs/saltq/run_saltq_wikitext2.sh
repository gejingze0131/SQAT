#!/bin/bash
# =============================================================================
# run_saltq_wikitext2.sh — SALT-Q on WikiText-2 (T2's Wiki2 column)
#
# The third fine-tuning task: causal language modelling on wikitext-2-raw-v1's train split,
# scored by perplexity on its test split. Unlike commonsense and math this is NOT a generative
# task — Stage 3 dispatches to scripts/eval_ppl.py (see runs/eval_vllm.sh's perplexity branch),
# so there is no vLLM env and no fold: the exported artifact is scored with its boundary gathers
# registered, exactly as it ships.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the model
# trains on and --dataset decides what it is scored on.
#
#   bash runs/saltq/run_saltq_wikitext2.sh --bits 2
#   bash runs/saltq/run_saltq_wikitext2.sh --skip_train --checkpoint_dir <ckpt>
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset wikitext2 \
    --config  configs/saltq_wiki2_int2_g32_ep3_bcal_sgptql_zp2x.yaml \
    "$@"
