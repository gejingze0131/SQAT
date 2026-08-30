#!/bin/bash
# =============================================================================
# run_qwha_commonsense.sh — QWHA baseline on Commonsense-170k (8-task suite)
#
# QWHA (Jeon et al., arXiv:2509.17428), run from the authors' code in baseline/QWHA/ against this
# repo's training recipe and its balanced in-domain calibration. The task is fixed HERE rather
# than passed as a flag, because the config decides what the model trains on and --dataset
# decides what it is scored on: two knobs that must agree.
#
#   bash runs/qwha/run_qwha_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/qwha/run_qwha_commonsense.sh --bits 2 --group_size 32 --with_base \
#       --config baseline/QWHA/sqat/configs/qwha_cs170k_int2_g32_ep1_span_bcal.yaml
#   bash runs/qwha/run_qwha_commonsense.sh --bits 3 --group_size 64 --skip_train
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  baseline/QWHA/sqat/configs/qwha_cs170k_int3_g64_ep1_span_bcal.yaml \
    "$@"
