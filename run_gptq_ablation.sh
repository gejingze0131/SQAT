#!/bin/bash
# =============================================================================
# run_gptq_ablation.sh — Selective-QAT ABLATION: full GPTQ on the merged weights
#
# Takes a TRAINED sqat_permute checkpoint, merges the LoRA into the (permuted)
# fp16 base, and quantizes the WHOLE weight with GPTQ — NO salient slice kept on
# the canonical grid, NO AWQ scaling. The permuted base + boundary gathers are
# kept so the model still runs correctly; only the Selective-QAT export-side
# protection is removed.
#
# Purpose: isolate what the Selective-QAT salient handling contributes. Compare
#   permuted-SQAT (salient slice canonical-grid + GPTQ non-salient + AWQ)   <- run_permute_sqat.sh
#   vs this ablation (everything GPTQ'd, like vanilla GPTQ on the trained model).
# Both start from the SAME checkpoint, so the delta is exactly the SQAT export
# protection (on a model whose LoRA was still SQAT-trained).
#
# Usage:
#   bash run_gptq_ablation.sh                          # auto-detect latest sqat_permute checkpoint
#   bash run_gptq_ablation.sh --checkpoint_dir <path>  # a specific checkpoint
#   bash run_gptq_ablation.sh --skip_eval              # export only
#   bash run_gptq_ablation.sh --config configs/sqat_permute_math.yaml --eval_gpu 1
# =============================================================================

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config — must match the run that produced the checkpoint
# ---------------------------------------------------------------------------
DATASET_NAME="commonsense" # "math" or "commonsense" (must match the config yaml)
CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
BITS=4
EVAL_GPU=0

CHECKPOINT_DIR=""          # empty → auto-detect latest outputs/qlora-sqat-permute*/final
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

if [ -z "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR=$(ls -td outputs/qlora-sqat-permute*/final 2>/dev/null | head -1 || true)
fi
if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: no sqat_permute checkpoint found; pass --checkpoint_dir <path>/final."
    exit 1
fi
if [ ! -f "$CHECKPOINT_DIR/sqat_permute_meta.pt" ]; then
    echo "ERROR: $CHECKPOINT_DIR has no sqat_permute_meta.pt — not a sqat_permute checkpoint."
    exit 1
fi

EVAL_DIR="outputs/qlora-sqat-permute-gptqfull-ablation-${DATASET_NAME}-${BITS}bit-dequant-eval"

echo "============================================================"
echo "  Selective-QAT Ablation: FULL GPTQ on merged weights"
echo "  Config:      $CONFIG"
echo "  Checkpoint:  $CHECKPOINT_DIR"
echo "  Export dir:  $EVAL_DIR"
echo "  Eval GPU:    cuda:$EVAL_GPU"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Export-only — full GPTQ (no salient slice, no AWQ)
# ---------------------------------------------------------------------------
echo -e "\n>>> Stage 1: Full-GPTQ ablation export"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
    --config           "$CONFIG" \
    --qat_mode         sqat_permute \
    --bits             "$BITS" \
    --asymmetric \
    --export_only \
    --export_dequant \
    --gptq_full \
    --checkpoint_dir   "$CHECKPOINT_DIR" \
    --merge_output_dir "$EVAL_DIR"

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
echo "  GPTQ ablation pipeline complete!"
echo "============================================================"
