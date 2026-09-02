#!/bin/bash
# =============================================================================
# run_qeft_wikitext2.sh — QEFT baseline on WikiText-2 (T2's Wiki2 column)
#
# Causal LM on wikitext-2-raw-v1 train, perplexity on test. Stage 3 dispatches to
# scripts/eval_ppl.py rather than vLLM (see runs/eval_vllm.sh's perplexity branch).
#
# The scored artifact is QEFT's MIXED-PRECISION product: INT2 g32 codes plus k=256 fp16 weak
# columns per linear (2.75 effective bits), so this row is not a pure 2-bit row — see
# baseline/QEFT/sqat/PROVENANCE.md.
#
#   bash runs/qeft/run_qeft_wikitext2.sh --bits 2 --group_size 32 --with_base
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset wikitext2 \
    --config  baseline/QEFT/sqat/configs/qeft_wiki2_int2_g32_ep3_lr9e5.yaml \
    "$@"
