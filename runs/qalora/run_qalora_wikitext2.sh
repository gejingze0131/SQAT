#!/bin/bash
# =============================================================================
# run_qalora_wikitext2.sh — QA-LoRA baseline on WikiText-2 (T2's Wiki2 column)
#
# Causal LM on wikitext-2-raw-v1 train, perplexity on test. Stage 3 dispatches to
# scripts/eval_ppl.py rather than vLLM (see runs/eval_vllm.sh's perplexity branch); the scored
# artifact is the MERGED INT2 checkpoint, which is what QA-LoRA deploys.
#
#   bash runs/qalora/run_qalora_wikitext2.sh --bits 2
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset wikitext2 \
    --config  configs/qalora_wiki2_int2_g32_ep3.yaml \
    "$@"
