#!/bin/bash
# =============================================================================
# run_saltq_alpaca.sh — SALT-Q on Alpaca (the MMLU column)
#
# The fourth fine-tuning task: general instruction tuning on alpaca-cleaned (51,760 records,
# datasets/alpaca/train.json, built by scripts/build_alpaca_dataset.py), scored on MMLU 5-shot.
#
# Unlike commonsense and math, the benchmark is EXTERNAL — it shares nothing with the training
# set, so there is no test split to generate against and no answer extractor. Stage 3 dispatches
# to scripts/eval_mmlu.py (see runs/eval_vllm.sh's MMLU branch), which runs lm-eval-harness in
# the TRAIN env; no vLLM and no fold, because eval_mmlu.py registers this export's boundary
# gathers itself through src.permute_common.lm_eval_model_kwargs.
#
# The task is fixed HERE rather than passed as a flag, because the config decides what the model
# trains on and --dataset decides what it is scored on.
#
#   bash runs/saltq/run_saltq_alpaca.sh --bits 2
#   bash runs/saltq/run_saltq_alpaca.sh --skip_train --checkpoint_dir <ckpt>
# =============================================================================

set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" \
    --dataset alpaca \
    --config  configs/saltq_alpaca_int2_g32_ep3_k256_c4cal_sgptql_zp2x.yaml \
    "$@"
