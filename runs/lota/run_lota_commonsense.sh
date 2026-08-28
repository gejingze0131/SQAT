#!/bin/bash
# =============================================================================
# run_lota_commonsense.sh — LoTA-QAF baseline on Commonsense-170k (8-task suite)
#
# LoTA-QAF (Chen et al., NeurIPS'25), run from the authors' code in baseline/LoTA-QAF/ against
# this repo's training recipe. The task is fixed HERE rather than passed as a flag, because the
# config decides what the model trains on and --dataset decides what it is scored on: two knobs
# that must agree and used to be set independently. Everything else forwards through, so the bit
# sweep works as before:
#
#   bash runs/lota/run_lota_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/lota/run_lota_commonsense.sh --bits 2 --group_size 32 --with_base \
#       --config baseline/LoTA-QAF/sqat/configs/lota_cs170k_int2_g32_ep1_span.yaml
#   bash runs/lota/run_lota_commonsense.sh --bits 3 --group_size 64 --skip_train
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  baseline/LoTA-QAF/sqat/configs/lota_cs170k_int3_g64_ep1_span.yaml \
    "$@"
