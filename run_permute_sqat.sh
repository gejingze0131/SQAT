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
#   bash run_permute_sqat.sh --resume_from <ckpt>     # CONTINUE training from a Trainer checkpoint
#                                                     #   (reuses permuted base; raise num_epochs to
#                                                     #    train past the original max_steps)
#   bash run_permute_sqat.sh --num_gpus 2 --config configs/sqat_permute.yaml
# =============================================================================

set -euo pipefail

# Fix A: avoid CUDA allocator fragmentation (the OOM showed several GB "reserved
# but unallocated"). expandable_segments lets the caching allocator grow/shrink
# segments so variable-length batches don't strand memory. Applies to train,
# export and eval below.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config (BOUNDARY_SIZES / GROUP_K must match the chosen --config yaml)
# ---------------------------------------------------------------------------
DATASET_NAME="math" # math or commonsense (must match the config yaml, which controls the boundary gather)
CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=4
BITS=3          # 2 / 3 / 4  (must match configs/*.yaml model.quant_bits; base stays NF4)

MODEL_NAME="meta-llama/Llama-2-7b-hf"
BOUNDARY_SIZES="2 30"     # must match configs/sqat_permute_${DATASET_NAME}.yaml: qat.sqat_permute.boundary_sizes
GROUP_K=128
EVAL_GPU=0                # single GPU used for export + evaluation
# Improvement 2: AWQ-style per-channel scaling of the salient slice; overrides the yaml.
# true  → quantize the salient slice in the amplified space W*S (q/k/v share S1, gate/up S2,
#         down S3), bake 1/S into the dense weight (better grid, no runtime cost)
# false → no salient scaling (original scheme)
AWQ_SCALE=true
# Improvement 1: GPTQ (OBS) export for the ~97% non-salient columns; overrides the yaml.
# true  → non-salient cols use GPTQ error compensation (salient slice stays on canonical grid)
# false → non-salient cols use plain RTN (original export)
# Composes with AWQ_SCALE (salient slice fixed to the amplified grid, non-salient GPTQ'd).
GPTQ_NONSALIENT=true
# LSQ / LSQ+ learnable quantization scale (LR-QAT style) on the salient slice; overrides the yaml.
# true  → the salient slice [0:group_k] is fakequant'd with a LEARNED scale[,zp] (asym→scale+zp),
#         init current_minmax, learned in the AWQ-amplified space when AWQ_SCALE=true. Aimed at
#         low-bit (W2/W3) stability. Composes with AWQ_SCALE and GPTQ_NONSALIENT.
# false → per-step min-max scale (original scheme).
# The export reads the learned scales from sqat_permute_meta.pt, so Stage-2 export needs no flag.
ENABLE_LSQ=true

SKIP_VALIDATE=true
SKIP_TRAIN=false
SKIP_EVAL=false
# CHECKPOINT_DIR="outputs/qlora-sqat-permute-4bit-sqat_permute/final"
CHECKPOINT_DIR=""
# 从某个 Trainer checkpoint 恢复继续训练（例：outputs/qlora-sqat-permute-3bit-sqat_permute/checkpoint-6000）。
# 留空 = 全新训练。sqat_permute 恢复时会复用已有的 <output_dir>/permuted_fp16_base（不重新生成 permute，
# 否则与 checkpoint 的 LoRA 不匹配）。注意：若 checkpoint 已接近 max_steps，想多训需在 config 调大 num_epochs。
RESUME_FROM=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_validate)  SKIP_VALIDATE=true; shift ;;
        --skip_train)     SKIP_TRAIN=true;    shift ;;
        --skip_eval)      SKIP_EVAL=true;     shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; SKIP_TRAIN=true; shift 2 ;;
        --resume_from)    RESUME_FROM="$2";   shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";      shift 2 ;;
        --config)         CONFIG="$2";        shift 2 ;;
        --model_name)     MODEL_NAME="$2";    shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";      shift 2 ;;
        --awq_scale)              AWQ_SCALE="$2";            shift 2 ;;
        --gptq_nonsalient)        GPTQ_NONSALIENT="$2";      shift 2 ;;
        --enable_lsq)             ENABLE_LSQ="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Map the toggle to the train.py flag (config yaml is the fallback default).
if [ "$AWQ_SCALE" = "true" ]; then
    AWQ_FLAG="--awq_scale"
else
    AWQ_FLAG="--no_awq_scale"
fi

# Map the GPTQ toggle to the train.py flag (config yaml is the fallback default).
if [ "$GPTQ_NONSALIENT" = "true" ]; then
    GPTQ_FLAG="--gptq_nonsalient"
else
    GPTQ_FLAG="--no_gptq_nonsalient"
fi

# Map the LSQ toggle to the train.py flag (config yaml is the fallback default).
if [ "$ENABLE_LSQ" = "true" ]; then
    LSQ_FLAG="--enable_lsq"
else
    LSQ_FLAG="--no_enable_lsq"
fi

# Resume: continue TRAINING from the checkpoint (reuses the permuted base; does NOT skip train).
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    if [ ! -d "$RESUME_FROM" ]; then
        echo "ERROR: --resume_from checkpoint dir not found: $RESUME_FROM"; exit 1
    fi
    RESUME_FLAG="--resume_from_checkpoint $RESUME_FROM"
    SKIP_TRAIN=false
fi

echo "============================================================"
echo "  Permuted Selective-QAT Pipeline"
echo "  Config:      $CONFIG"
echo "  Model:       $MODEL_NAME"
echo "  GPUs:        $NUM_GPUS (train) / cuda:$EVAL_GPU (eval)"
echo "  Boundaries:  [$BOUNDARY_SIZES]   group_k=$GROUP_K   bits=$BITS"
echo "  AWQ-scale:     $AWQ_SCALE"
echo "  GPTQ non-sal:  $GPTQ_NONSALIENT"
echo "  LSQ:           $ENABLE_LSQ"
[ -n "$RESUME_FROM" ] && echo "  Resume from:   $RESUME_FROM"
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
        $AWQ_FLAG \
        $GPTQ_FLAG \
        $LSQ_FLAG \
        $RESUME_FLAG \
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
        $GPTQ_FLAG \
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
    for eval_dir in outputs/qlora-sqat-permute*-dequant-eval; do
        [ -d "$eval_dir" ] || continue
        found=true
        echo "  Evaluating $eval_dir"
        if [ $DATASET_NAME = "commonsense" ]; then
            # For commonsense, we run both benchmarks to see if the permutation has any effect on one but not the other.
             CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_benchmarks.py eval \
                --model_path "$eval_dir" \
                --output_dir results/benchmarks
             CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_mmlu.py \
                --model_path  "$eval_dir" \
                --num_fewshot 0 \
                --output_dir  results/mmlu
        elif [ $DATASET_NAME = "math" ]; then
            CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_math.py \
                --model_path  "$eval_dir" \
                --num_fewshot 5 \
                --output_dir  results/math
        fi
    done
    shopt -u nullglob
    [ "$found" = true ] || echo "  (no exported eval dirs found under outputs/qlora-sqat-permute*)"
fi

echo -e "\n============================================================"
echo "  Permuted Selective-QAT pipeline complete!"
echo "============================================================"
