#!/bin/bash
# =============================================================================
# run_full_qat_commonsense.sh — Full QAT / LR-QAT baseline on Commonsense-170k (8-task suite)
#
# Fake-quantizes EVERY target linear's base weight on each forward pass — the opposite
# end of the freedom axis from SALT-Q's salient-only allocation.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/full_qat/run_full_qat_commonsense.sh --bits 3
#   bash runs/full_qat/run_full_qat_commonsense.sh --bits 2 --config configs/sqat_permute_commonsense.yaml
#   bash runs/full_qat/run_full_qat_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  configs/sqat_permute_commonsense.yaml \
    "$@"
