#!/bin/bash
# =============================================================================
# runs/qlora/_pipeline.sh — plain QLoRA baseline (qat_mode = none) pipeline engine
#
# Not meant to be run directly — the per-task entry scripts fix --dataset and the
# matching --config together, which is the pair that must never disagree:
#   runs/qlora/run_qlora_math.sh / run_qlora_commonsense.sh
#
# Mirrors runs/permute_sqat/_pipeline.sh but with qat_mode=none (no Selective-QAT, no
# permutation). It deliberately reads the SAME configs/sqat_permute_${DATASET_NAME}.yaml
# so that every parameter UNRELATED to the sqat_permute method (model, LoRA,
# dataset, training hyper-params, group_size, ...) stays identical to the
# permuted-SQAT run — only the QAT method differs. This makes the two runs a
# fair apples-to-apples comparison.
#
# Two export variants are produced and BOTH are benchmarked:
#   - export_dequant     : INT4 quantize -> dequantize, simulates the quant
#                          error of the deployed model (lower / realistic bound)
#   - export_merged_only : merge LoRA into the FP16 base with NO quantization,
#                          simulates the FP16 accuracy UPPER bound
#
# Pipeline:
#   Stage 1  Training (auto-exports the dequant eval model via export.merge_and_save)
#   Stage 1b Export merged-only (FP16 upper bound) from the final checkpoint
#   Stage 2  Export-only (both variants) when --skip_train / --checkpoint_dir is given
#   Stage 3  Generative evaluation of BOTH exported models through vLLM, in the `vllm`
#            conda env (runs/eval_vllm.sh)
#
# Usage:
#   bash runs/qlora/run_qlora_math.sh                          # all stages
#   bash runs/qlora/run_qlora_math.sh --skip_eval              # train+export, no benchmarks
#   bash runs/qlora/run_qlora_math.sh --skip_train             # export + eval from latest checkpoint
#   bash runs/qlora/run_qlora_math.sh --checkpoint_dir <path>  # export + eval from a specific checkpoint
#   bash runs/qlora/run_qlora_math.sh --num_gpus 2 --config configs/sqat_permute_math.yaml
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
EVAL_GPU=0                # single GPU used for EXPORT (one dense fp16 model per card)
EVAL_GPUS="0,1,2,3"       # GPUs vLLM evaluates on; >1 id => tensor-parallel, matching
                          # runs/saltq/_pipeline.sh so the suites are run identically across methods

# Dedicated output dir so a plain-QLoRA run never clobbers a real sqat_permute run.
# Empty => derived from DATASET_NAME after parsing. Declared eagerly, it baked in
# the DEFAULT task and a --dataset override never reached it.
OUTPUT_DIR=""

SKIP_TRAIN=false
SKIP_EVAL=false
# CHECKPOINT_DIR="outputs/qlora-none-commonsense-4bit-none/final"
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
        --model_name)     MODEL_NAME="$2";    shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";      shift 2 ;;
        --eval_gpus)      EVAL_GPUS="$2";     shift 2 ;;
        --dataset)        DATASET_NAME="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolved here rather than at declaration: --config and --dataset can now be passed in either
# order without one silently overwriting the other.
[ -n "$CONFIG" ] || CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="outputs/qlora-none-${DATASET_NAME}"

# Fail in two seconds rather than after a 20-hour train + a meaningless score. Only when this run
# will actually evaluate — a --skip_eval run is free to train on anything.
if [ "$SKIP_EVAL" = false ]; then
    assert_config_matches_dataset "$CONFIG" "$DATASET_NAME"
fi

DEQUANT_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-none-dequant-eval"
MERGED_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-none-merged-eval"

echo "============================================================"
echo "  Plain QLoRA Baseline (qat_mode=none) Pipeline"
echo "  Config:      $CONFIG"
echo "  Model:       $MODEL_NAME"
echo "  GPUs:        $NUM_GPUS (train) / [$EVAL_GPUS] (eval, vLLM tensor-parallel)"
echo "  Bits:        $BITS"
echo "  Output dir:  $OUTPUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Training (auto-exports the dequant eval model afterwards)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: QLoRA ${BITS}-bit baseline training"
    accelerate launch \
        --config_file   "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config     "$CONFIG" \
        --qat_mode   none \
        --bits       "$BITS" \
        --output_dir "$OUTPUT_DIR" \
        --export_dequant \
        --report_to wandb

    # scripts/train.py writes to "<output_dir>-<bits>bit-<qat_mode>", NOT to <output_dir>. This
    # line omitted the suffix and killed the run AFTER a full epoch and the dequant export, with
    # "expected checkpoint at .../final not found". Line 70's commented example and the two
    # EVAL_DIRs above always had it right; only this one line disagreed.
    CHECKPOINT_DIR="${OUTPUT_DIR}-${BITS}bit-none/final"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: expected checkpoint at $CHECKPOINT_DIR not found; pass --checkpoint_dir."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"

    # --- Stage 1b: Export merged-only (FP16 upper bound) -------------------
    echo -e "\n>>> Stage 1b: Export merged-only (FP16 upper bound)"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config           "$CONFIG" \
        --qat_mode         none \
        --bits             "$BITS" \
        --export_only \
        --export_merged_only \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --merge_output_dir "$MERGED_EVAL_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export-only (both variants) from an existing checkpoint
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = true ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo -e "\n>>> Stage 2: Export-only from $CHECKPOINT_DIR"

    echo "  (a) dequant export (INT-b quant->dequant, realistic bound)"
    if [ -d "$DEQUANT_EVAL_DIR" ]; then
      echo "      already at $DEQUANT_EVAL_DIR — skipping"
    else
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config           "$CONFIG" \
        --qat_mode         none \
        --bits             "$BITS" \
        --export_only \
        --export_dequant \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --merge_output_dir "$DEQUANT_EVAL_DIR"
    fi

    # (b) is what makes this pipeline a CONTROL and not just another baseline: the FP16 merge
    # separates "the data pipeline cannot teach this task" from "quantization is what breaks it".
    # It was commented out, so a --skip_train rerun silently produced only half the comparison.
    if [ -d "$MERGED_EVAL_DIR" ]; then
        echo "  (b) merged-only export already at $MERGED_EVAL_DIR — skipping"
    else
        echo "  (b) merged-only export (FP16 upper bound)"
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
            --config           "$CONFIG" \
            --qat_mode         none \
            --bits             "$BITS" \
            --export_only \
            --export_merged_only \
            --checkpoint_dir   "$CHECKPOINT_DIR" \
            --merge_output_dir "$MERGED_EVAL_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: Generative evaluation through vLLM (separate conda env)
#
# Both exported variants are scored on their OWN generations, the same metric the SALT-Q run
# reports, so this baseline and the method are directly comparable.
# runs/eval_vllm.sh hops conda envs (vLLM pins its own torch) and folds any residual permutation
# into the weights first.
# ---------------------------------------------------------------------------
eval_one() {
    local eval_dir="$1"
    [ -d "$eval_dir" ] || { echo "  (skip) $eval_dir not found"; return; }
    bash runs/eval_vllm.sh \
        --model_path "$eval_dir" \
        --dataset    "$DATASET_NAME" \
        --gpus       "$EVAL_GPUS" \
        --tag        "$(basename "$eval_dir")"
}

if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: Generative evaluation (vLLM)"
    eval_one "$DEQUANT_EVAL_DIR"   # INT4 quant->dequant (realistic bound)
    eval_one "$MERGED_EVAL_DIR"    # FP16 merged (upper bound)
fi

echo -e "\n============================================================"
echo "  Plain QLoRA baseline pipeline complete!"
echo "============================================================"
