#!/bin/bash
# =============================================================================
# run_qlora_alpaca.sh — QLoRA on Alpaca (the MMLU column)
#
# Produces this column's fp16 UPPER bound: the adapter-mounted deployed state, exported as
# <output_dir>-2bit-none-merged-eval (NF4 base dequantized + adapter merged, no re-quantization).
# The matching FLOOR — that same checkpoint put through GPTQ INT2 g32 on the column's shared C4
# 128x2048 calibration — is a separate step (jobs/alpaca_floor_gptq_int2.pbs), because it is a
# different artifact and must be labelled as one.
#
# Stage 3 dispatches to scripts/eval_mmlu.py rather than vLLM (runs/eval_vllm.sh's MMLU branch).
#
#   bash runs/qlora/run_qlora_alpaca.sh --bits 2
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset alpaca \
    --config  configs/qlora_none_alpaca_int2_g32_ep3.yaml \
    "$@"
