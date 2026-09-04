#!/bin/bash
# =============================================================================
# run_qeft_alpaca.sh — QEFT baseline on Alpaca (the MMLU column)
#
# The scored artifact is QEFT's MIXED-PRECISION product: INT2 g32 codes plus k=256 fp16 weak
# columns per linear (2.75 effective bits), so this row is not a pure 2-bit row — see
# baseline/QEFT/sqat/PROVENANCE.md. Stage 3 dispatches to scripts/eval_mmlu.py
# (runs/eval_vllm.sh's MMLU branch), not to vLLM.
#
#   bash runs/qeft/run_qeft_alpaca.sh --bits 2 --group_size 32 --with_base
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset alpaca \
    --config  baseline/QEFT/sqat/configs/qeft_alpaca_int2_g32_ep3_lr5e5.yaml \
    "$@"
