#!/bin/bash
# =============================================================================
# run_saltq.sh — SALT-Q pipeline (train -> export -> eval) on MetaMathQA
#
# SALT-Q = Saliency-Allocated Low-bit Trainability. Per weight matrix, after the shared offline
# permutation has moved the salient input channels to the front:
#
#   cols [0 : group_k)    ~1-2%    FULL weight freedom  -> nn.Parameter weights + LSQ fakequant
#   cols [group_k : in)  ~98-99%   AFFINE freedom only  -> FROZEN GPTQ codes, trainable (s, z)
#   the integer codes                                    -> ZERO freedom
#
# There is no adapter and no merge anywhere: the trained salient weights land on the quantization
# grid by construction, and the trained (s, z) go straight into the metadata slots a GPTQ
# checkpoint already carries. The export asserts deployed == trained with max|Δ| == 0.
#
# Stages:
#   Stage 1  Training. Three offline steps run first, automatically, on rank 0:
#              a) permuted fp16 base   <output_dir>/permuted_fp16_base   (~13 GB, shared with
#                                                                         sqat_permute)
#              b) frozen-code base     <output_dir>/saltq_base           (~7 GB, GPTQ)
#            Both are REUSED (never regenerated) on resume and on export — the codes are a
#            one-shot discrete choice, so a fresh base invalidates every trained tensor.
#            The merge-free export runs automatically at the end of training.
#   Stage 2  Export-only (only when --skip_train / --checkpoint_dir is given)
#   Stage 3  Benchmark evaluation (boundary gather auto-applied from sqat_permute_meta.pt)
#
# Usage:
#   bash run_saltq.sh                              # train + export + eval
#   bash run_saltq.sh --skip_eval                  # train + export only
#   bash run_saltq.sh --skip_train                 # export + eval from the latest checkpoint
#   bash run_saltq.sh --checkpoint_dir <path>      # export + eval from a specific checkpoint
#   bash run_saltq.sh --resume_from <ckpt>         # CONTINUE training (reuses both bases)
#   bash run_saltq.sh --salient_lr 1e-5            # sweep the key new hyper-parameter
#   bash run_saltq.sh --num_gpus 2 --config configs/saltq.yaml
# =============================================================================

set -euo pipefail

# Avoid CUDA allocator fragmentation. SALT-Q rebuilds one [out, in] fp32 weight per projection on
# every forward, so the allocator sees a steady churn of large short-lived blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG="configs/saltq.yaml"
# Which benchmark suite Stage 3 runs. "math" -> gsm8k via eval_math.py into results/math and the
# long-format results_saltq.csv. "commonsense" -> the 8-task commonsense suite + MMLU via
# eval_benchmarks.py / eval_mmlu.py into results/commonsense_170k. The CSV collector parses gsm8k
# specifically, so it only runs for math.
DATASET_NAME="math"
# Optional explicit lm-eval task list for the commonsense path. Empty => the 8-task suite in
# eval_benchmarks.DEFAULT_TASKS plus MMLU. Set it (e.g. "commonsense_qa") to score exactly one
# task and skip MMLU — useful when the fine-tuning target IS that task.
EVAL_TASKS=""
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=4                # 3 x RTX 6000 Ada (46 GB)
BITS=2                    # first run: INT3 g64, where a Permuted-SQAT MetaMath baseline exists.
                          # INT2 (where GPTQ-only non-salient collapses) is the target regime next.

MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_ROOT="outputs/saltq"     # must match training.output_dir in the yaml
EVAL_GPU=0                # single GPU used for export
EVAL_GPUS="0,1,2,3"         # GPUs used for evaluation; >1 id => data-parallel eval

# The single most important new hyper-parameter: these are REAL WEIGHTS, not an adapter, so a
# LoRA-scale lr (2e-4) destroys them. Empty = take the value from the yaml.
SALIENT_LR=""
# lr for the quantization params (s, z) of BOTH segments. The non-salient 98%'s entire task
# adaptation lives here. Empty = take the value from the yaml.
SCALES_LR=""
# Train the RMSNorm weights too (they stay fp16 at deploy, so it is free extra freedom).
TRAIN_LAYERNORMS=false

# Every eval run is folded into this one long-format table (one row per run/task/metric).
RESULTS_CSV="results_saltq.csv"
RESULTS_NOTE=""

SKIP_TRAIN=false
SKIP_EVAL=false
CHECKPOINT_DIR=""
RESUME_FROM=""
SALTQ_BASE_DIR=""         # explicit frozen-code base; defaults to <output_dir>/saltq_base
PERMUTED_BASE_DIR=""      # explicit permuted fp16 base; defaults to <output_dir>/permuted_fp16_base
                          # (pass this to REUSE another run's permutation when output_dir differs,
                          # e.g. an ablation that must keep the permute step single-variable)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_train)       SKIP_TRAIN=true;        shift ;;
        --skip_eval)        SKIP_EVAL=true;         shift ;;
        --checkpoint_dir)   CHECKPOINT_DIR="$2"; SKIP_TRAIN=true; shift 2 ;;
        --resume_from)      RESUME_FROM="$2";       shift 2 ;;
        --saltq_base_dir)   SALTQ_BASE_DIR="$2";    shift 2 ;;
        --permuted_base_dir) PERMUTED_BASE_DIR="$2"; shift 2 ;;
        --num_gpus)         NUM_GPUS="$2";          shift 2 ;;
        --bits)             BITS="$2";              shift 2 ;;
        --config)           CONFIG="$2";            shift 2 ;;
        --model_name)       MODEL_NAME="$2";        shift 2 ;;
        --output_root)      OUTPUT_ROOT="$2";       shift 2 ;;
        --eval_gpu)         EVAL_GPU="$2";          shift 2 ;;
        --eval_gpus)        EVAL_GPUS="$2";         shift 2 ;;
        --salient_lr)       SALIENT_LR="$2";        shift 2 ;;
        --scales_lr)        SCALES_LR="$2";         shift 2 ;;
        --train_layernorms) TRAIN_LAYERNORMS="$2";  shift 2 ;;
        --results_csv)      RESULTS_CSV="$2";       shift 2 ;;
        --dataset)          DATASET_NAME="$2";      shift 2 ;;
        --eval_tasks)       EVAL_TASKS="$2";        shift 2 ;;
        --note)             RESULTS_NOTE="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_FLAGS=""
[ -n "$SALIENT_LR" ] && EXTRA_FLAGS="$EXTRA_FLAGS --salient_lr $SALIENT_LR"
[ -n "$SCALES_LR" ]  && EXTRA_FLAGS="$EXTRA_FLAGS --saltq_scales_lr $SCALES_LR"
if [ "$TRAIN_LAYERNORMS" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --train_layernorms"
else
    EXTRA_FLAGS="$EXTRA_FLAGS --no_train_layernorms"
fi

SALTQ_BASE_FLAG=""
[ -n "$SALTQ_BASE_DIR" ] && SALTQ_BASE_FLAG="--saltq_base_dir $SALTQ_BASE_DIR"
PERMUTED_BASE_FLAG=""
[ -n "$PERMUTED_BASE_DIR" ] && PERMUTED_BASE_FLAG="--permuted_base_dir $PERMUTED_BASE_DIR"

# Build the eval launch prefix. >1 GPU id => data-parallel eval via `accelerate launch`
# (plain DDP data sharding); a single id => one-GPU `python`.
EVAL_NUM_GPUS=$(awk -F',' '{print NF}' <<< "$EVAL_GPUS")
if [ "$EVAL_NUM_GPUS" -gt 1 ]; then
    EVAL_LAUNCH="accelerate launch --num_processes $EVAL_NUM_GPUS --num_machines 1 \
        --gpu_ids $EVAL_GPUS --mixed_precision no --main_process_port ${EVAL_PORT:-29600}"
else
    EVAL_LAUNCH="env CUDA_VISIBLE_DEVICES=$EVAL_GPUS python"
fi

# Resume: CONTINUE training from a checkpoint (reuses both offline bases; does NOT skip train).
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    if [ ! -d "$RESUME_FROM" ]; then
        echo "ERROR: --resume_from checkpoint dir not found: $RESUME_FROM"; exit 1
    fi
    RESUME_FLAG="--resume_from_checkpoint $RESUME_FROM"
    SKIP_TRAIN=false
fi

echo "============================================================"
echo "  SALT-Q — Saliency-Allocated Low-bit Trainability"
echo "  Config:        $CONFIG"
echo "  Model:         $MODEL_NAME"
echo "  Bits:          INT$BITS (asymmetric affine)"
echo "  GPUs:          $NUM_GPUS (train) / [$EVAL_GPUS] (eval, $EVAL_NUM_GPUS-way data-parallel)"
echo "  salient_lr:    ${SALIENT_LR:-<from yaml>}   (REAL weights — not an adapter lr)"
echo "  scales_lr:     ${SCALES_LR:-<from yaml>}   ((s, z) of both segments)"
echo "  train_LN:      $TRAIN_LAYERNORMS"
[ -n "$RESUME_FROM" ]    && echo "  Resume from:   $RESUME_FROM"
[ -n "$SALTQ_BASE_DIR" ] && echo "  Frozen codes:  $SALTQ_BASE_DIR"
[ -n "$PERMUTED_BASE_DIR" ] && echo "  Permuted base: $PERMUTED_BASE_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Training (offline permute + GPTQ code freezing run first, automatically)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: SALT-Q training"
    accelerate launch \
        --config_file   "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config   "$CONFIG" \
        --qat_mode saltq \
        --bits     "$BITS" \
        --asymmetric \
        $EXTRA_FLAGS \
        $SALTQ_BASE_FLAG \
        $PERMUTED_BASE_FLAG \
        $RESUME_FLAG \
        --report_to wandb

    CHECKPOINT_DIR=$(ls -td ${OUTPUT_ROOT}-${BITS}bit-saltq/final 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: could not locate training output dir; pass --checkpoint_dir for export/eval."
        exit 1
    fi
    echo ">>> Training done. Checkpoint: $CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: Export-only
#
# A SALT-Q checkpoint contains ONLY the trainable tensors (salient weights + all (s, z)) plus a
# pointer to the frozen-code base — the multi-GB int8 codes are never duplicated per checkpoint.
# The export therefore REQUIRES that base; it is read from saltq_ckpt_meta.pt, or pass
# --saltq_base_dir explicitly.
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = true ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo -e "\n>>> Stage 2: Export-only from $CHECKPOINT_DIR"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/train.py \
        --config         "$CONFIG" \
        --qat_mode       saltq \
        --bits           "$BITS" \
        --asymmetric \
        $SALTQ_BASE_FLAG \
        $PERMUTED_BASE_FLAG \
        --export_only \
        --checkpoint_dir "$CHECKPOINT_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: Benchmark evaluation
# The exported dir ships sqat_permute_meta.pt, so the eval scripts register the runtime residual
# reorder at each segment boundary. Evaluating it with a loader that ignores that metadata gives
# garbage.
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: Evaluating exported models"
    shopt -s nullglob
    found=false
    for eval_dir in ${OUTPUT_ROOT}-${BITS}bit-saltq-deploy-eval; do
        [ -d "$eval_dir" ] || continue
        found=true
        EVAL_OUT="results/commonsense_170k"
        [ -n "$EVAL_TASKS" ] && EVAL_OUT="results/${EVAL_TASKS%% *}"
        echo "  Evaluating $eval_dir  (suite: $DATASET_NAME, tasks: ${EVAL_TASKS:-<default+mmlu>}, out: $EVAL_OUT)"
        if [ "$DATASET_NAME" = "commonsense" ]; then
            # Same 4-way data-parallel launcher as the math path — the exported model ships
            # sqat_permute_meta.pt and both eval scripts register the boundary gathers from it.
            if [ -n "$EVAL_TASKS" ]; then
                $EVAL_LAUNCH scripts/eval_benchmarks.py eval \
                    --model_path  "$eval_dir" \
                    --tasks       $EVAL_TASKS \
                    --output_dir  "$EVAL_OUT"
            else
                $EVAL_LAUNCH scripts/eval_benchmarks.py eval \
                    --model_path  "$eval_dir" \
                    --output_dir  "$EVAL_OUT"
                $EVAL_LAUNCH scripts/eval_mmlu.py \
                    --model_path  "$eval_dir" \
                    --num_fewshot 0 \
                    --output_dir  "$EVAL_OUT"
            fi
        else
            $EVAL_LAUNCH scripts/eval_math.py \
                --model_path  "$eval_dir" \
                --num_fewshot 5 \
                --output_dir  results/math
        fi
    done
    shopt -u nullglob
    [ "$found" = true ] || echo "  (no exported eval dirs found under ${OUTPUT_ROOT}-${BITS}bit-saltq-deploy-eval)"

    # Fold the timestamped lm-eval JSONs into one long-format table. Idempotent, so re-running
    # the pipeline never duplicates rows. Run-level context (bits / group_size / group_k / lrs /
    # the freedom split) is recovered from the artifacts next to the evaluated model.
    if [ "$DATASET_NAME" != "math" ]; then
        echo -e "\n>>> Skipping $RESULTS_CSV: the collector parses gsm8k, and this run evaluated"
        echo "    the $DATASET_NAME suite. Raw JSONs are under results/commonsense_170k/."
        echo -e "\n============================================================"
        echo "  SALT-Q pipeline complete!"
        echo "============================================================"
        exit 0
    fi

    echo -e "\n>>> Collecting results into $RESULTS_CSV"
    # --seed merges the reported MetaMath baselines (source=report) so the new SALT-Q row lands
    # next to what it has to beat. --filter saltq keeps the lm-eval side to SALT-Q only: the older
    # JSONs under results/ are intermediate artifacts from a previous development round and are
    # NOT valid baselines.
    python scripts/collect_saltq_results.py \
        --results_dir results \
        --csv         "$RESULTS_CSV" \
        --config      "$CONFIG" \
        --seed        baselines_metamath.csv \
        --filter      saltq \
        --note        "${RESULTS_NOTE}"
fi

echo -e "\n============================================================"
echo "  SALT-Q pipeline complete!"
echo "============================================================"
