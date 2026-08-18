#!/bin/bash
# =============================================================================
# run_qlora_commonsense.sh — plain QLoRA baseline (qat_mode=none) on Commonsense-170k (8-task suite)
#
# The floor every QAT method has to beat: LoRA on an NF4 base, no quantization-aware
# training at all. Same config family as Permuted-SQAT, so only the method differs.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/qlora/run_qlora_commonsense.sh --bits 3
#   bash runs/qlora/run_qlora_commonsense.sh --bits 2 --config configs/qlora_none_cs170k_int2_g32.yaml
#   bash runs/qlora/run_qlora_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  configs/sqat_permute_commonsense.yaml \
    "$@"
