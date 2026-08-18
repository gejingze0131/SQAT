#!/usr/bin/env bash
set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, so the caller's cwd does
# not matter. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
cd_repo_root

# Whatever env the caller activated. The old default was an absolute path into a conda env
# on a different machine, so this script could not run here at all.
PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" analyze_boundary_salient_channels.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset metamath \
    --n_samples 512 \
    --seq_len 2048 \
    --outlier_log_sigma 2.5 \
    --down_outlier_log_sigma 2.5 \
    --group_size 64 \
    --max_segments 4 \
    --output_dir salient_analysis_out
