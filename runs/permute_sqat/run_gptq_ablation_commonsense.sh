#!/bin/bash
# =============================================================================
# run_gptq_ablation_commonsense.sh — Permuted-SQAT ablation: full GPTQ (lower bound) on Commonsense-170k (8-task suite)
#
# Takes a TRAINED sqat_permute checkpoint and GPTQs the whole merged weight, salient slice
# included — the lower bound that says what the salient carve-out is worth.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/permute_sqat/run_gptq_ablation_commonsense.sh --bits 3
#   bash runs/permute_sqat/run_gptq_ablation_commonsense.sh --bits 2 --config configs/sqat_permute_commonsense.yaml
#   bash runs/permute_sqat/run_gptq_ablation_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_gptq_ablation.sh" \
    --dataset commonsense \
    --config  configs/sqat_permute_commonsense.yaml \
    "$@"
