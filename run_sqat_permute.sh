#!/bin/bash
# =============================================================================
# run_sqat_permute.sh — SQAT-Permute training + evaluation pipeline
#
# Stages:
#   0. Stage-1 permutation equivalence validation (fast, fp16 only, no training)
#   1. SQAT-Permute training
#   2. Dequant export (quantize → dequant → fp16 weights for evaluation)
#   3. Benchmark evaluation
#
# Usage:
#   bash run_sqat_permute.sh                          # default: all stages
#   bash run_sqat_permute.sh --skip_validate          # skip Stage 0
#   bash run_sqat_permute.sh --skip_train             # export + eval only (needs checkpoint)
#   bash run_sqat_permute.sh --checkpoint_dir <path>  # export-only from saved checkpoint
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG="configs/sqat_permute.yaml"
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=4
BITS=4

MODEL_NAME="meta-llama/Llama-2-7b-hf"
BOUNDARY_SIZES="2 30"    # must match configs/sqat_permute.yaml boundary_sizes
GROUP_K=128
EVAL_GPU=0                # single GPU for evaluation

SKIP_VALIDATE=false
SKIP_TRAIN=false
CHECKPOINT_DIR=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_validate)   SKIP_VALIDATE=true; shift ;;
        --skip_train)      SKIP_TRAIN=true;    shift ;;
        --checkpoint_dir)  CHECKPOINT_DIR="$2"; SKIP_TRAIN=true; shift 2 ;;
        --num_gpus)        NUM_GPUS="$2";       shift 2 ;;
        --config)          CONFIG="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  SQAT-Permute Pipeline"
echo "  Config:      $CONFIG"
echo "  GPUs:        $NUM_GPUS"
echo "  Boundaries:  [$BOUNDARY_SIZES]  group_k=$GROUP_K"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0: Permutation equivalence validation (Stage 1 in the code)
# ---------------------------------------------------------------------------
if [ "$SKIP_VALIDATE" = false ]; then
    echo -e "\n>>> Stage 0: Permutation equivalence validation (fp16, no training)"
    bash run_validation.sh \
        --model_name   "$MODEL_NAME" \
        --boundary_sizes $BOUNDARY_SIZES \
        --group_k      $GROUP_K \
        --n_samples    32 \
        --seq_len      512

    echo ">>> Stage 0 PASSED — proceeding to training"
fi

# ---------------------------------------------------------------------------
# Stage 1: SQAT-Permute training
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: SQAT-Permute training"
    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config   "$CONFIG" \
        --qat_mode sqat_permute \
        --bits     "$BITS" \
        --asymmetric \
        --export_dequant \
        --report_to wandb

    # Locate the freshly-produced output directory
    CHECKPOINT_DIR=$(ls -td outputs/qlora-sqat-permute-*/final 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT_DIR" ]; then
        CHECKPOINT_DIR=$(ls -td outputs/qlora-sqat-permute*/final 2>/dev/null | head -1 || true)
    fi
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Could not locate training output directory. Set --checkpoint_dir manually."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export (if not already triggered by merge_and_save: true in config)
# ---------------------------------------------------------------------------
# The config has export.merge_and_save: true so export runs automatically after
# training. This block handles the explicit export-only path when --skip_train
# or --checkpoint_dir is provided.

if [ "$SKIP_TRAIN" = true ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo -e "\n>>> Stage 2: Export-only from $CHECKPOINT_DIR"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config         "$CONFIG" \
        --qat_mode       sqat_permute \
        --bits           "$BITS" \
        --asymmetric \
        --export_only \
        --export_dequant \
        --checkpoint_dir "$CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: Benchmark evaluation
# ---------------------------------------------------------------------------
echo -e "\n>>> Stage 3: Evaluating exported models"

for eval_dir in outputs/qlora-sqat-permute*-dequant-eval outputs/qlora-sqat-permute*-eval; do
    [ -d "$eval_dir" ] || continue
    echo "  Evaluating $eval_dir"

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_benchmarks.py eval \
        --model_path "$eval_dir" \
        --output_dir results/benchmarks

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_mmlu.py \
        --model_path  "$eval_dir" \
        --num_fewshot 0 \
        --output_dir  results/mmlu
done

echo -e "\n============================================================"
echo "  SQAT-Permute pipeline complete!"
echo "============================================================"
