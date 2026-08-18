#!/bin/bash
# =============================================================================
# run_qlora_math.sh — plain QLoRA baseline (qat_mode=none) on MetaMathQA (GSM8K + MATH)
#
# The floor every QAT method has to beat: LoRA on an NF4 base, no quantization-aware
# training at all. Same config family as Permuted-SQAT, so only the method differs.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/qlora/run_qlora_math.sh --bits 3
#   bash runs/qlora/run_qlora_math.sh --bits 2 --config configs/sqat_permute_math.yaml
#   bash runs/qlora/run_qlora_math.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset math \
    --config  configs/sqat_permute_math.yaml \
    "$@"
