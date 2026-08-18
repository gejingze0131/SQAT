#!/bin/bash
# =============================================================================
# run_permute_fp16_ablation_commonsense.sh — Permuted-SQAT ablation: fp16 salient (upper bound) on Commonsense-170k (8-task suite)
#
# Identical permutation, but the salient slice is left in fp16 — the upper bound the
# quantized salient slice is measured against.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/permute_sqat/run_permute_fp16_ablation_commonsense.sh --bits 3
#   bash runs/permute_sqat/run_permute_fp16_ablation_commonsense.sh --bits 2 --config configs/sqat_permute_commonsense.yaml
#   bash runs/permute_sqat/run_permute_fp16_ablation_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_fp16_ablation.sh" \
    --dataset commonsense \
    --config  configs/sqat_permute_commonsense.yaml \
    "$@"
