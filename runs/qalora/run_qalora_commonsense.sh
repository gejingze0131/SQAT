#!/bin/bash
# =============================================================================
# run_qalora_commonsense.sh — QA-LoRA baseline on Commonsense-170k (8-task suite)
#
# QA-LoRA (Xu et al., 2023). Reads the same config family as Permuted-SQAT so that every
# parameter unrelated to the QAT method stays identical and only the method differs.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/qalora/run_qalora_commonsense.sh --bits 3
#   bash runs/qalora/run_qalora_commonsense.sh --bits 2 --config configs/qalora_cs170k_int2_g32.yaml
#   bash runs/qalora/run_qalora_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  configs/sqat_permute_commonsense.yaml \
    "$@"
