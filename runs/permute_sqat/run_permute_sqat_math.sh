#!/bin/bash
# =============================================================================
# run_permute_sqat_math.sh — Permuted Selective-QAT on MetaMathQA (GSM8K + MATH)
#
# The predecessor of SALT-Q: segment-permuted Selective QAT (validate -> train -> export
# -> eval).
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the
# model trains on and --dataset decides what it is scored on: two knobs that must agree and
# used to be set independently. Everything else still comes from the command line and is
# forwarded through, so the bit sweep works as before:
#
#   bash runs/permute_sqat/run_permute_sqat_math.sh --bits 3
#   bash runs/permute_sqat/run_permute_sqat_math.sh --bits 2 --config configs/sqat_permute_math.yaml
#   bash runs/permute_sqat/run_permute_sqat_math.sh --skip_eval
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset math \
    --config  configs/sqat_permute_math.yaml \
    "$@"
