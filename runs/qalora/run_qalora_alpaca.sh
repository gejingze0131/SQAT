#!/bin/bash
# =============================================================================
# run_qalora_alpaca.sh — QA-LoRA on Alpaca (the MMLU column)
#
# GPTQ INT2 g32 frozen base + group-pooled adapter; the scored artifact is the MERGED INT2
# checkpoint, which is what QA-LoRA deploys. Stage 3 dispatches to scripts/eval_mmlu.py
# (runs/eval_vllm.sh's MMLU branch), not to vLLM.
#
#   bash runs/qalora/run_qalora_alpaca.sh --bits 2
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset alpaca \
    --config  configs/qalora_alpaca_int2_g32_ep3.yaml \
    "$@"
