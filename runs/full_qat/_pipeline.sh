#!/bin/bash
# =============================================================================
# runs/full_qat/_pipeline.sh — Full QAT (qat_mode = full) pipeline engine
#
# Not meant to be run directly — the per-task entry scripts fix --dataset and the
# matching --config together, which is the pair that must never disagree:
#   runs/full_qat/run_full_qat_math.sh / run_full_qat_commonsense.sh
#
# Full QAT fake-quantizes EVERY target linear's base weight on each forward pass
# (groupwise STE), so the LoRA adapter is trained against the actual INT4 grid
# instead of the fp16 dequantization of NF4. This is the "quantize everything"
# upper-effort baseline to compare against permuted Selective-QAT.
#
# Like runs/qlora/_pipeline.sh, it deliberately reads the SAME
# configs/sqat_permute_${DATASET_NAME}.yaml so that every parameter UNRELATED to
# the QAT method (model, LoRA, dataset, training hyper-params, group_size,
# symmetric, ...) stays identical to the permuted-SQAT run — only the QAT method
# differs. The sqat_permute-only sections (boundary_sizes, group_k, gptq,
# awq_scale) are simply ignored by FullQAT.
#
# Two export variants are produced and BOTH are benchmarked:
#   - export_dequant     : INT4 quantize -> dequantize, the realistic deployed
#                          accuracy of the full-QAT model (the headline number)
#   - export_merged_only : merge LoRA into the FP16 base with NO quantization
#                          (the full-QAT model evaluated without quant error)
#
# Pipeline:
#   Stage 1  Training (auto-exports the dequant eval model via export.merge_and_save)
#   Stage 1b Export merged-only (FP16) from the final checkpoint
#   Stage 2  Export-only (both variants) when --skip_train / --checkpoint_dir is given
#   Stage 3  Benchmark evaluation of BOTH exported models
#
# Usage:
#   bash runs/full_qat/run_full_qat_math.sh                          # all stages
#   bash runs/full_qat/run_full_qat_math.sh --skip_eval              # train+export, no benchmarks
#   bash runs/full_qat/run_full_qat_math.sh --skip_train             # export + eval from latest checkpoint
#   bash runs/full_qat/run_full_qat_math.sh --checkpoint_dir <path>  # export + eval from a specific checkpoint
#   bash runs/full_qat/run_full_qat_math.sh --num_gpus 2 --config configs/sqat_permute_math.yaml
# =============================================================================

set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, and refuses a --config
# whose training task disagrees with --dataset. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
cd_repo_root

# Avoid CUDA allocator fragmentation (see runs/permute_sqat/_pipeline.sh for rationale).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config — read the SAME sqat_permute config to keep all non-method params equal
# ---------------------------------------------------------------------------
DATASET_NAME="math" # "math" or "commonsense" (must match the config yaml)
# Empty => resolved from DATASET_NAME after parsing, so --config and
# --dataset cannot depend on the order they were passed in.
CONFIG=""
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=2
BITS=3

MODEL_NAME="meta-llama/Llama-2-7b-hf"
EVAL_GPU=0                # single GPU used for export + evaluation

# Dedicated output dir so a full-QAT run never clobbers a permuted/none run.
# Empty => derived from DATASET_NAME after parsing. Declared eagerly, it baked in
# the DEFAULT task and a --dataset override never reached it.
OUTPUT_DIR=""

# LSQ / LSQ+ learnable quantization scale (LR-QAT style). true → --enable_lsq,
# false → --no_enable_lsq (original per-step min-max scale). The flag is threaded
# through the train AND export-only commands so the export reads the learned grid.
ENABLE_LSQ=true

SKIP_TRAIN=false
SKIP_EVAL=false
CHECKPOINT_DIR=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_train)     SKIP_TRAIN=true;    shift ;;
        --skip_eval)      SKIP_EVAL=true;     shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; SKIP_TRAIN=true; shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";      shift 2 ;;
        --config)         CONFIG="$2";        shift 2 ;;
        # Every entry script under this folder documents --bits, and the README says it works
        # everywhere; this engine used to reject it with "Unknown argument: --bits" and exit 1.
        --bits)           BITS="$2";          shift 2 ;;
        --dataset)        DATASET_NAME="$2";  shift 2 ;;
        --model_name)     MODEL_NAME="$2";    shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";      shift 2 ;;
        --enable_lsq)     ENABLE_LSQ=true;    shift ;;
        --no_enable_lsq)  ENABLE_LSQ=false;   shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolved here rather than at declaration: --config and --dataset can now be passed in either
# order without one silently overwriting the other.
[ -n "$CONFIG" ] || CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="outputs/qlora-full-${DATASET_NAME}"

# Fail in two seconds rather than after a 20-hour train + a meaningless score. Only when this run
# will actually evaluate — a --skip_eval run is free to train on anything.
if [ "$SKIP_EVAL" = false ]; then
    assert_config_matches_dataset "$CONFIG" "$DATASET_NAME"
fi

# Resolve the LSQ flag passed to scripts/train.py (train + export-only commands).
if [ "$ENABLE_LSQ" = true ]; then
    LSQ_FLAG="--enable_lsq"
else
    LSQ_FLAG="--no_enable_lsq"
fi

DEQUANT_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-full-dequant-eval"
MERGED_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-full-merged-eval"

echo "============================================================"
echo "  Full QAT (qat_mode=full) Pipeline"
echo "  Config:      $CONFIG"
echo "  Model:       $MODEL_NAME"
echo "  GPUs:        $NUM_GPUS (train) / cuda:$EVAL_GPU (eval)"
echo "  Bits:        $BITS"
echo "  Output dir:  $OUTPUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Training (auto-exports the dequant eval model afterwards)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: Full QAT ${BITS}-bit training"
    accelerate launch \
        --config_file   "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config     "$CONFIG" \
        --qat_mode   full \
        --bits       "$BITS" \
        --asymmetric \
        $LSQ_FLAG \
        --output_dir "$OUTPUT_DIR" \
        --export_dequant \
        --report_to wandb

    CHECKPOINT_DIR="${OUTPUT_DIR}/final"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: expected checkpoint at $CHECKPOINT_DIR not found; pass --checkpoint_dir."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"

    # # --- Stage 1b: Export merged-only (FP16, no quant) --------------------
    # echo -e "\n>>> Stage 1b: Export merged-only (FP16, no quant error)"
    # CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
    #     --config           "$CONFIG" \
    #     --qat_mode         full \
    #     --bits             "$BITS" \
    #     --asymmetric \
    #     --export_only \
    #     --export_merged_only \
    #     --checkpoint_dir   "$CHECKPOINT_DIR" \
    #     --merge_output_dir "$MERGED_EVAL_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export-only (both variants) from an existing checkpoint
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = true ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo -e "\n>>> Stage 2: Export-only from $CHECKPOINT_DIR"

    echo "  (a) dequant export (INT4 quant->dequant, realistic deployed accuracy)"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config           "$CONFIG" \
        --qat_mode         full \
        --bits             "$BITS" \
        --asymmetric \
        $LSQ_FLAG \
        --export_only \
        --export_dequant \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --merge_output_dir "$DEQUANT_EVAL_DIR"

    # echo "  (b) merged-only export (FP16, no quant error)"
    # CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
    #     --config           "$CONFIG" \
    #     --qat_mode         full \
    #     --bits             "$BITS" \
    #     --asymmetric \
    #     --export_only \
    #     --export_merged_only \
    #     --checkpoint_dir   "$CHECKPOINT_DIR" \
    #     --merge_output_dir "$MERGED_EVAL_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: Benchmark evaluation of BOTH exported models
# ---------------------------------------------------------------------------
eval_one() {
    local eval_dir="$1"
    [ -d "$eval_dir" ] || { echo "  (skip) $eval_dir not found"; return; }
    echo "  Evaluating $eval_dir"
    if [ "$DATASET_NAME" = "commonsense" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_benchmarks.py eval \
            --model_path "$eval_dir" \
            --output_dir results/benchmarks
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_mmlu.py \
            --model_path  "$eval_dir" \
            --num_fewshot 0 \
            --output_dir  results/mmlu
    elif [ "$DATASET_NAME" = "math" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_math.py \
            --model_path  "$eval_dir" \
            --num_fewshot 5 \
            --output_dir  results/math
    else
        echo "Unknown DATASET_NAME: $DATASET_NAME"
        exit 1
    fi
}

if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: Evaluating exported models"
    eval_one "$DEQUANT_EVAL_DIR"   # INT4 quant->dequant (full-QAT deployed accuracy)
    eval_one "$MERGED_EVAL_DIR"    # FP16 merged (no quant error)
fi

echo -e "\n============================================================"
echo "  Full QAT pipeline complete!"
echo "============================================================"
