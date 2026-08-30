#!/bin/bash
# =============================================================================
# run_qeft_commonsense.sh — QEFT baseline on Commonsense-170k (8-task suite)
#
# QEFT (Lee et al., EMNLP 2024 Findings, arXiv:2410.08661), implemented against this repo's own
# GPTQ / permutation / data code (baseline/QEFT/sqat/) and run on this repo's training recipe and
# balanced in-domain calibration. The task is fixed HERE rather than passed as a flag, because
# the config decides what the model trains on and --dataset decides what it is scored on: two
# knobs that must agree.
#
#   bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/qeft/run_qeft_commonsense.sh --bits 2 --group_size 32 --with_base \
#       --config baseline/QEFT/sqat/configs/qeft_cs170k_int2_g32_ep1_span_bcal.yaml
#   bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --skip_train
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset commonsense \
    --config  baseline/QEFT/sqat/configs/qeft_cs170k_int3_g64_ep1_span_bcal.yaml \
    "$@"
