#!/bin/bash
# =============================================================================
# run_permute_fp16_ablation.sh — Permuted Selective-QAT UPPER-BOUND ablation
#
# Runs the IDENTICAL permute algorithm as run_permute_sqat.sh (same calibration
# set, saliency selection, P_k / P4 transforms, boundary gathers), but
# instead of QAT-protecting the salient slice it keeps that slice at FULL fp16.
# The non-salient columns are still GPTQ'd. The ONLY difference vs. our method is:
#   ours      : salient slice = low-bit QAT-protected weight
#   this (UB) : salient slice = fp16 (full precision)
#
# NO training. Starts from a PLAIN QLoRA checkpoint (qat_mode=none) whose adapter
# was trained on the ORIGINAL un-permuted base. The adapter is merged into the
# dense fp16 base, then the permute transforms are applied to the merged weights
# (a pure equivalence transform), then PTQ (fp16 salient + GPTQ non-salient).
#
# All meta (boundary_sizes, group_k, group_size, top_k_ratio, q_bits, ...) is read
# from the SAME configs/sqat_permute_${DATASET_NAME}.yaml as the permute-SQAT run,
# so nothing-but-the-method differs — a rigorous upper-bound comparison.
#
# Usage:
#   bash run_permute_fp16_ablation.sh
#   bash run_permute_fp16_ablation.sh --checkpoint_dir outputs/qlora-none-math-3bit-none/final
#   bash run_permute_fp16_ablation.sh --skip_eval
#   bash run_permute_fp16_ablation.sh --config configs/sqat_permute_math.yaml --eval_gpu 1
# =============================================================================

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config — must match the permute-SQAT run it is compared against
# ---------------------------------------------------------------------------
DATASET_NAME="math"  # "math" or "commonsense" (must match the config yaml)
CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
BITS=3
EVAL_GPU=0

# A PLAIN QLoRA (qat_mode=none) checkpoint — adapter trained on the original base.
CHECKPOINT_DIR="outputs/qlora-none-${DATASET_NAME}-${BITS}bit-none/final"
SKIP_EVAL=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --config)         CONFIG="$2";         shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";       shift 2 ;;
        --skip_eval)      SKIP_EVAL=true;      shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: plain-QLoRA checkpoint not found at '$CHECKPOINT_DIR'."
    echo "       Pass --checkpoint_dir <path>/final (a qat_mode=none QLoRA checkpoint)."
    exit 1
fi
if [ ! -f "$CHECKPOINT_DIR/adapter_model.safetensors" ] && \
   [ ! -f "$CHECKPOINT_DIR/adapter_model.bin" ]; then
    echo "ERROR: $CHECKPOINT_DIR has no adapter_model.* — not a QLoRA adapter checkpoint."
    exit 1
fi

EVAL_DIR="outputs/qlora-permute-fp16salient-ablation-${DATASET_NAME}-${BITS}bit-dequant-eval"

echo "============================================================"
echo "  Permuted Selective-QAT UPPER-BOUND ablation (fp16 salient)"
echo "  Config:      $CONFIG"
echo "  Checkpoint:  $CHECKPOINT_DIR  (plain QLoRA)"
echo "  Export dir:  $EVAL_DIR"
echo "  Eval GPU:    cuda:$EVAL_GPU"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Export — permute the merged weights, fp16 salient + GPTQ non-salient
# ---------------------------------------------------------------------------
echo -e "\n>>> Stage 1: fp16-salient ablation export"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/export_permute_fp16_ablation.py \
    --config         "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --bits           "$BITS" \
    --asymmetric \
    --output_dir     "$EVAL_DIR"

# ---------------------------------------------------------------------------
# Stage 2: Benchmark evaluation (eval scripts auto-register the boundary gather)
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 2: Evaluating $EVAL_DIR"
    if [ "$DATASET_NAME" = "commonsense" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_benchmarks.py eval \
            --model_path "$EVAL_DIR" \
            --output_dir results/benchmarks
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_mmlu.py \
            --model_path  "$EVAL_DIR" \
            --num_fewshot 0 \
            --output_dir  results/mmlu
    elif [ "$DATASET_NAME" = "math" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_math.py \
            --model_path  "$EVAL_DIR" \
            --num_fewshot 5 \
            --output_dir  results/math
    else
        echo "Unknown DATASET_NAME: $DATASET_NAME"; exit 1
    fi
fi

echo -e "\n============================================================"
echo "  fp16-salient ablation pipeline complete!"
echo "============================================================"
