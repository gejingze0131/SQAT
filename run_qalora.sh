#!/bin/bash
# =============================================================================
# run_qalora.sh — QA-LoRA (qat_mode = qalora) pipeline
#
# QA-LoRA (Xu et al., 2023 — arXiv:2309.14717), faithful to the paper/official repo:
# the base is a REAL GPTQ INT-b g{group_size} model quantized ONCE directly from fp16
# (NO NF4, no double-quant — a pre-step builds outputs/.../qalora_intb_base and training
# loads it frozen in fp16). LoRA A is resized to consume ONE average-pooled activation per
# quantization group (AvgPool1d(group_size); A: [rank, ceil(in_features/group_size)]). The
# adapter is a plain low-rank path ADDED ON TOP of the frozen quantized base and is NEVER
# fake-quantized:
#     y = base_layer(x) + scaling * B( A( avgpool_g(x) ) )
# Because a pooled-input adapter's delta is constant within each input group, at deploy it
# folds EXACTLY into the affine zero-points (paper Eq. 7): deployed = W_base_intb + expand(delta),
# the base ints are unchanged, only the dequantized zero-point shifts. Training == deploy
# bit-for-bit (verified). NOT a QAT-on-merged-weight scheme — the adapter stays outside the
# quantizer (that is the whole point of QA-LoRA's mergeability).
#
# COST vs the NF4 baselines: the frozen fp16 base is ~13GB/GPU (vs ~4GB NF4) and the GPTQ base
# build writes a ~13GB checkpoint to disk — ensure free disk + GPU headroom (lower batch if OOM).
#
# Like run_full_qlora.sh / run_none_qlora.sh, it deliberately reads the SAME
# configs/sqat_permute_${DATASET_NAME}.yaml so that every parameter UNRELATED to
# the QAT method (model, LoRA, dataset, training hyper-params, group_size,
# symmetric, ...) stays identical to the permuted-SQAT run — only the QAT method
# differs. The sqat_permute-only sections (boundary_sizes, group_k, gptq,
# awq_scale, lsq) are simply ignored by QA-LoRA.
#
# Notes specific to QA-LoRA:
#   - Asymmetric only. --qat_mode qalora forces symmetric=False (and the handler
#     raises on symmetric=True), so --asymmetric is passed for clarity.
#   - lora.dropout MUST be 0.0 (the group adapter folds into static zero-points and
#     cannot represent activation dropout) — the shared config already sets dropout: 0.0.
#   - LSQ does NOT apply to QA-LoRA: the base uses a fixed min-max affine grid and the
#     adapter is never quantized, so the config's lsq.enabled is ignored (no LSQ flag).
#   - The dequant export path is REQUIRED (the merged model is a dense quantize->dequant
#     weight), so --export_dequant is always passed for the quantized variant.
#
# Two export variants are produced and BOTH are benchmarked:
#   - export_dequant     : dequant_b(W_base) + adapter-in-zero-point — the deployed pure
#                          INT-b QA-LoRA model (headline number)
#   - export_merged_only : merge the group adapter into the ORIGINAL base (NF4-dequant, no
#                          INT-b quant) — the no-INT-b-quant reference, same convention as the
#                          none/full merged-eval siblings
#
# Pipeline:
#   Stage 1  Training (auto-exports the dequant eval model via export.merge_and_save)
#   Stage 1b Export merged-only (FP16) from the final checkpoint
#   Stage 2  Export-only (both variants) when --skip_train / --checkpoint_dir is given
#   Stage 3  Benchmark evaluation of BOTH exported models
#
# Usage:
#   bash run_qalora.sh                          # all stages
#   bash run_qalora.sh --skip_eval              # train+export, no benchmarks
#   bash run_qalora.sh --skip_train             # export + eval from latest checkpoint
#   bash run_qalora.sh --checkpoint_dir <path>  # export + eval from a specific checkpoint
#   bash run_qalora.sh --num_gpus 2 --config configs/sqat_permute_math.yaml --bits 2
#   bash run_qalora.sh --eval_gpus 0 --note "..."   # 1-GPU eval; stamp a note on the CSV rows
#
# Results are appended to results_saltq.csv (--results_csv) via scripts/collect_saltq_results.py,
# the same table SALT-Q writes to, so a QA-LoRA control row sits next to the run it controls for.
# =============================================================================

set -euo pipefail

# Avoid CUDA allocator fragmentation (see run_permute_sqat.sh for rationale).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config — read the SAME sqat_permute config to keep all non-method params equal
# ---------------------------------------------------------------------------
DATASET_NAME="math" # "math" or "commonsense" (must match the config yaml)
CONFIG="configs/sqat_permute_${DATASET_NAME}.yaml"
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=4
BITS=2            # 2 / 3 / 4 (must match configs/*.yaml model.quant_bits; base stays NF4)

MODEL_NAME="meta-llama/Llama-2-7b-hf"
EVAL_GPU=0                # single GPU used for EXPORT (one dense fp16 model must fit on one card)
EVAL_GPUS="0,1,2,3"       # GPUs used for EVALUATION; >1 id => data-parallel eval, same as
                          # run_saltq.sh. Evaluating on 1 GPU while SALT-Q evaluates on 4 does not
                          # change the score, only the wall clock — but keeping them identical
                          # removes one more thing that has to be argued about.

# Dedicated output dir so a QA-LoRA run never clobbers a permuted/full/none run.
OUTPUT_DIR="outputs/qlora-qalora-${DATASET_NAME}"

# Every eval run is folded into the same long-format table SALT-Q writes to, so the QA-LoRA row
# lands next to the SALT-Q row it is the control for.
RESULTS_CSV="results_saltq.csv"
RESULTS_NOTE=""

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
        --bits)           BITS="$2";          shift 2 ;;
        --model_name)     MODEL_NAME="$2";    shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --eval_gpu)       EVAL_GPU="$2";      shift 2 ;;
        --eval_gpus)      EVAL_GPUS="$2";     shift 2 ;;
        --results_csv)    RESULTS_CSV="$2";   shift 2 ;;
        --note)           RESULTS_NOTE="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DEQUANT_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-qalora-dequant-eval"
MERGED_EVAL_DIR="${OUTPUT_DIR}-${BITS}bit-qalora-merged-eval"

# Build the eval launch prefix. >1 GPU id => data-parallel eval via `accelerate launch` (plain DDP
# data sharding); a single id => one-GPU `python`. Identical to run_saltq.sh.
EVAL_NUM_GPUS=$(awk -F',' '{print NF}' <<< "$EVAL_GPUS")
if [ "$EVAL_NUM_GPUS" -gt 1 ]; then
    EVAL_LAUNCH="accelerate launch --num_processes $EVAL_NUM_GPUS --num_machines 1 \
        --gpu_ids $EVAL_GPUS --mixed_precision no --main_process_port ${EVAL_PORT:-29600}"
else
    EVAL_LAUNCH="env CUDA_VISIBLE_DEVICES=$EVAL_GPUS python"
fi

echo "============================================================"
echo "  QA-LoRA (qat_mode=qalora) Pipeline"
echo "  Config:      $CONFIG"
echo "  Model:       $MODEL_NAME"
echo "  GPUs:        $NUM_GPUS (train) / cuda:$EVAL_GPU (export) / [$EVAL_GPUS] (eval, ${EVAL_NUM_GPUS}-way DP)"
echo "  Bits:        $BITS  (affine asymmetric, group-wise)"
echo "  Output dir:  $OUTPUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Training (auto-exports the dequant eval model afterwards)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: QA-LoRA ${BITS}-bit training"
    accelerate launch \
        --config_file   "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config     "$CONFIG" \
        --qat_mode   qalora \
        --bits       "$BITS" \
        --asymmetric \
        --output_dir "$OUTPUT_DIR" \
        --export_dequant \
        --report_to wandb

    # src/trainer.py appends "-{bits}bit-{qat_mode}" to the configured output_dir, so the actual
    # checkpoint lives at "${OUTPUT_DIR}-${BITS}bit-qalora/final" (NOT "${OUTPUT_DIR}/final").
    CHECKPOINT_DIR="${OUTPUT_DIR}-${BITS}bit-qalora/final"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: expected checkpoint at $CHECKPOINT_DIR not found; pass --checkpoint_dir."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export-only (both variants) from an existing checkpoint
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = true ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo -e "\n>>> Stage 2: Export-only from $CHECKPOINT_DIR"

    echo "  (a) dequant export (dequant_b(W_base) + adapter-in-zero-point, deployed INT-b accuracy)"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config           "$CONFIG" \
        --qat_mode         qalora \
        --bits             "$BITS" \
        --asymmetric \
        --export_only \
        --export_dequant \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --merge_output_dir "$DEQUANT_EVAL_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: Benchmark evaluation of BOTH exported models
# ---------------------------------------------------------------------------
eval_one() {
    local eval_dir="$1"
    [ -d "$eval_dir" ] || { echo "  (skip) $eval_dir not found"; return; }
    echo "  Evaluating $eval_dir"
    if [ "$DATASET_NAME" = "commonsense" ]; then
        $EVAL_LAUNCH scripts/eval_benchmarks.py eval \
            --model_path "$eval_dir" \
            --output_dir results/benchmarks
        $EVAL_LAUNCH scripts/eval_mmlu.py \
            --model_path  "$eval_dir" \
            --num_fewshot 0 \
            --output_dir  results/mmlu
    elif [ "$DATASET_NAME" = "math" ]; then
        $EVAL_LAUNCH scripts/eval_math.py \
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
    eval_one "$DEQUANT_EVAL_DIR"   # INT quant->dequant (QA-LoRA deployed accuracy)
    eval_one "$MERGED_EVAL_DIR"    # FP16 merged (no quant error)

    # Fold the timestamped lm-eval JSONs into the SAME long-format table SALT-Q writes to, so the
    # QA-LoRA control sits next to the run it controls for. Idempotent — re-running never
    # duplicates rows. --filter qalora keeps this invocation from re-ingesting the SALT-Q JSONs
    # (they are already in the table with their own run context); _infer_method maps the
    # "...-qalora-dequant-eval" dir name to the "QA-LoRA" label.
    echo -e "\n>>> Collecting results into $RESULTS_CSV"
    python scripts/collect_saltq_results.py \
        --results_dir results \
        --csv         "$RESULTS_CSV" \
        --config      "$CONFIG" \
        --filter      qalora \
        --note        "${RESULTS_NOTE}"
fi

echo -e "\n============================================================"
echo "  QA-LoRA pipeline complete!"
echo "============================================================"
