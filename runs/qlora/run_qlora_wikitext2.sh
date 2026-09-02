#!/bin/bash
# =============================================================================
# run_qlora_wikitext2.sh — QLoRA on WikiText-2 (T2's Wiki2 column)
#
# Produces this column's fp16 UPPER bound: the adapter-mounted deployed state, exported as
# <output_dir>-2bit-none-merged-eval (NF4 base dequantized + adapter merged, no re-quantization).
# The matching FLOOR — that same checkpoint put through GPTQ INT2 g32 on the task's own
# calibration — is a separate step, jobs/wiki2_qlora_int2_floor.pbs, because it is a different
# artifact and must be labelled as one.
#
# Stage 3 dispatches to scripts/eval_ppl.py rather than vLLM (runs/eval_vllm.sh).
#
#   bash runs/qlora/run_qlora_wikitext2.sh --bits 2
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset wikitext2 \
    --config  configs/qlora_none_wiki2_int2_g32_ep3.yaml \
    "$@"
