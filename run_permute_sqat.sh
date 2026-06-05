#!/bin/bash
# =============================================================================
# run_permute_sqat.sh — Permuted Selective-QAT pipeline (validate → train → export → eval)
#
# Pipeline (qat_mode = sqat_permute):
#   Stage 0  Permutation equivalence verification (fp32, no training) — scripts/verify_permute.py
#   Stage 1  Training: permute fp16 base → reload NF4 once → fused Selective-QAT + QLoRA
#            (export to a dequant eval model runs automatically via export.merge_and_save)
#   Stage 2  Export-only (only when --skip_train / --checkpoint_dir is given)
#   Stage 3  Benchmark evaluation (boundary gather auto-applied from sqat_permute_meta.pt)
#
# Usage:
#   bash run_permute_sqat.sh                          # all stages
#   bash run_permute_sqat.sh --skip_validate          # skip Stage 0
#   bash run_permute_sqat.sh --skip_eval              # train+export, no benchmarks
#   bash run_permute_sqat.sh --skip_train             # export + eval from latest checkpoint
#   bash run_permute_sqat.sh --checkpoint_dir <path>  # export + eval from a specific checkpoint
#   bash run_permute_sqat.sh --num_gpus 2 --config configs/sqat_permute.yaml
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (BOUNDARY_SIZES / GROUP_K must match the chosen --config yaml)
# ---------------------------------------------------------------------------
CONFIG="configs/sqat_permute_commonsense.yaml"
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=4
BITS=4

MODEL_NAME="meta-llama/Llama-2-7b-hf"
BOUNDARY_SIZES="2 30"     # must match configs/sqat_permute_commonsense.yaml: qat.sqat_permute.boundary_sizes
GROUP_K=128
EVAL_GPU=0                # single GPU used for export + evaluation
# Group-Hadamard rotation of the salient slice (q/k/v/gate/up); overrides the yaml.
# true  → smooth co-located weight/activation outliers (recommended)
# false → original concentrated-group scheme
ONLINE_GROUP_HADAMARD=false

SKIP_VALIDATE=true
SKIP_TRAIN=false
SKIP_EVAL=false
CHECKPOINT_DIR=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_validate)  SKIP_VALIDATE=true; shift ;;
        --skip_train)     SKIP_TRAIN=true;    shift ;;
        --skip_eval)      SKIP_EVAL=true;     shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; SKIP_TRAIN=true; shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";      shift 2 ;;
        --config)         CONFIG="$2";        shift 2 ;;
        --model_name)     MODEL_NAME="$2";    shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";      shift 2 ;;
        --online_group_hadamard) ONLINE_GROUP_HADAMARD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Map the toggle to the train.py flag (config yaml is the fallback default).
if [ "$ONLINE_GROUP_HADAMARD" = "true" ]; then
    HADAMARD_FLAG="--online_group_hadamard"
else
    HADAMARD_FLAG="--no_online_group_hadamard"
fi

echo "============================================================"
echo "  Permuted Selective-QAT Pipeline"
echo "  Config:      $CONFIG"
echo "  Model:       $MODEL_NAME"
echo "  GPUs:        $NUM_GPUS (train) / cuda:$EVAL_GPU (eval)"
echo "  Boundaries:  [$BOUNDARY_SIZES]   group_k=$GROUP_K   bits=$BITS"
echo "  GroupHadamard: $ONLINE_GROUP_HADAMARD"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0: Permutation equivalence verification (P_k + P4 + Hadamard, fp32)
# ---------------------------------------------------------------------------
if [ "$SKIP_VALIDATE" = false ]; then
    echo -e "\n>>> Stage 0: Permutation equivalence verification (fp32, no training)"
    bash run_validation.sh \
        --model_name     "$MODEL_NAME" \
        --boundary_sizes $BOUNDARY_SIZES \
        --group_k        $GROUP_K
    echo ">>> Stage 0 PASSED — proceeding"
fi

# ---------------------------------------------------------------------------
# Stage 1: Training (export to dequant eval model runs automatically afterwards)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: Permuted Selective-QAT training"
    accelerate launch \
        --config_file   "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config   "$CONFIG" \
        --qat_mode sqat_permute \
        --bits     "$BITS" \
        --asymmetric \
        $HADAMARD_FLAG \
        --export_dequant \
        --report_to wandb

    CHECKPOINT_DIR=$(ls -td outputs/qlora-sqat-permute*/final 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: could not locate training output dir; pass --checkpoint_dir for export/eval."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export-only (reloads the PERMUTED fp16 base recorded in the metadata)
# ---------------------------------------------------------------------------
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
# Stage 3: Benchmark evaluation (eval scripts auto-register the boundary gather)
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: Evaluating exported models"
    shopt -s nullglob
    found=false
    for eval_dir in outputs/qlora-sqat-permute*-dequant-eval outputs/qlora-sqat-permute*-eval; do
        [ -d "$eval_dir" ] || continue
        found=true
        echo "  Evaluating $eval_dir"
        # CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_benchmarks.py eval \
        #     --model_path "$eval_dir" \
        #     --output_dir results/benchmarks
        # CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_mmlu.py \
        #     --model_path  "$eval_dir" \
        #     --num_fewshot 0 \
        #     --output_dir  results/mmlu
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_math.py \
            --model_path  "$eval_dir" \
            --num_fewshot 5 \
            --output_dir  results/math

    done
    shopt -u nullglob
    [ "$found" = true ] || echo "  (no exported eval dirs found under outputs/qlora-sqat-permute*)"
fi

echo -e "\n============================================================"
echo "  Permuted Selective-QAT pipeline complete!"
echo "============================================================"
