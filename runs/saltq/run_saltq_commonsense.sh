#!/bin/bash
# =============================================================================
# run_saltq_commonsense.sh — SALT-Q on Commonsense-170k (8-task suite)
#
# SALT-Q = Saliency-Allocated Low-bit Trainability. Full pipeline: offline permute + GPTQ
# code freezing, training, merge-free export, generative eval.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/saltq/run_saltq_commonsense.sh --bits 3
#   bash runs/saltq/run_saltq_commonsense.sh --bits 2 --config configs/saltq_cs170k_int2_g32.yaml
#   bash runs/saltq/run_saltq_commonsense.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  configs/saltq_cs170k_int3_g64.yaml \
    "$@"
