#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/jingze/miniconda3/envs/sqat/bin/python}"

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
