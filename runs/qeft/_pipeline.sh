#!/bin/bash
# =============================================================================
# runs/qeft/_pipeline.sh — QEFT baseline pipeline engine (base -> train -> export -> eval)
#
# Not meant to be run directly — the per-task entry script fixes --dataset and the matching
# --config together, which is the pair that must never disagree:
#   runs/qeft/run_qeft_commonsense.sh
#
# QEFT (Lee et al., EMNLP 2024 Findings, arXiv:2410.08661). Where QA-LoRA folds a group-pooled
# adapter into the affine zero-points and SALT-Q trains real weights on the quantization grid,
# QEFT keeps k FP16 "weak columns" per linear and trains ONLY those:
#
#     y = x @ W_INTb^T  +  x[:, weak] @ W_weak^T           (W_INTb frozen, W_weak fp16)
#
# with one GLOBAL column reordering (OGR) folded offline into the whole network so those columns
# are a contiguous leading slice everywhere except o_proj. There is no adapter and no merge: the
# deployed model is INT-b codes PLUS those fp16 columns, so the row reads "b + fp16".
#
# SAME ENV as the rest of the repo (`saltq`): unlike LoTA-QAF and QWHA, nothing here needs
# upstream's pinned stack — the method is implemented against this repo's own GPTQ, permutation
# and data code (baseline/QEFT/sqat/, see its PROVENANCE.md).
#
# Stages:
#   Stage 0  The mixed-precision base, one GPU: calibrate lambda = diag(2 X^T X) on the BALANCED
#            in-domain set, pick the one global weak-column set, fold its permutation into the
#            model, then GPTQ everything except those columns (they stay fp16). Shared per
#            (bits, group_size, k); SKIPPED when it exists, and refuses to reuse a base built for
#            a different configuration — the weak-column set is a one-shot discrete choice, so a
#            rebuilt base orphans every checkpoint trained against it.
#   Stage 1  Weak-column tuning on this repo's data recipe (src/data.py's PROMPT, loss_span,
#            collator and shuffle seed), lr from QEFT's own paper. DDP, one full base per rank.
#            Checkpoints hold only the trained columns.
#   Stage 2  Dense export: scatter the trained columns back into the base weight. Checked, not
#            assumed — per layer the frozen bulk must be bit-identical to the base and the
#            two-GEMM forward must match the dense one. --with_base also exports the bare
#            mixed-precision base: this method builds its own base, so that is its own floor.
#   Stage 3  Generative evaluation through vLLM (runs/eval_vllm.sh) — the same greedy generation
#            + exact match every other row in RESULTS_SUMMARY.md is scored with — then folded
#            into results_saltq.csv next to them.
#
# Usage:
#   bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/qeft/run_qeft_commonsense.sh --bits 2 --group_size 32 --with_base \
#        --config baseline/QEFT/sqat/configs/qeft_cs170k_int2_g32_ep1_span_bcal.yaml
#   bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --skip_train   # export + eval
# =============================================================================

set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, and refuses a --config
# whose training task disagrees with --dataset. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
cd_repo_root

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QEFT_DIR="$REPO_ROOT/baseline/QEFT/sqat"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG=""
DATASET_NAME="commonsense"
EVAL_TASKS=""
BITS=""
GROUP_SIZE=""

WITH_BASE=false           # also export + score the bare mixed-precision base (this row's floor)
NUM_GPUS=4
EVAL_GPUS="0,1,2,3"
RESUME_FROM=""

RESULTS_CSV="results_saltq.csv"
RESULTS_NOTE=""

SKIP_BASE=false
SKIP_TRAIN=false
SKIP_EXPORT=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";            shift 2 ;;
        --dataset)      DATASET_NAME="$2";      shift 2 ;;
        --bits)         BITS="$2";              shift 2 ;;
        --group_size)   GROUP_SIZE="$2";        shift 2 ;;
        --with_base)    WITH_BASE=true;         shift ;;
        --num_gpus)     NUM_GPUS="$2";          shift 2 ;;
        --resume_from)  RESUME_FROM="$2";       shift 2 ;;
        --skip_base)    SKIP_BASE=true;         shift ;;
        --skip_train)   SKIP_TRAIN=true;        shift ;;
        --skip_export)  SKIP_EXPORT=true;       shift ;;
        --skip_eval)    SKIP_EVAL=true;         shift ;;
        --eval_gpus)    EVAL_GPUS="$2";         shift 2 ;;
        --eval_tasks)   EVAL_TASKS="$2";        shift 2 ;;
        --results_csv)  RESULTS_CSV="$2";       shift 2 ;;
        --note)         RESULTS_NOTE="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -n "$BITS" ]       || { echo "ERROR: --bits is required"; exit 1; }
[ -n "$GROUP_SIZE" ] || { echo "ERROR: --group_size is required"; exit 1; }
[ -n "$CONFIG" ]     || { echo "ERROR: --config is required (the entry script supplies one)"; exit 1; }
[ -f "$CONFIG" ]     || { echo "ERROR: no such config: $CONFIG"; exit 1; }

# Fail in two seconds rather than after a five-hour train and a meaningless score.
if [ "$SKIP_EVAL" = false ]; then
    assert_config_matches_dataset "$CONFIG" "$DATASET_NAME"
fi

# The config decides what is trained; --bits/--group_size decide which base it is trained on. A
# mismatch would train against a 3-bit grid while the log says 2-bit, and never say so.
read -r CFG_BITS CFG_GROUP CFG_K BASE_DIR < <(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
q = c["qeft"]
print(c["model"]["quant_bits"], q["group_size"], q["k"], q["base_dir"])
PY
)
[ "$CFG_BITS" = "$BITS" ] || { echo "ERROR: --bits $BITS but $CONFIG has model.quant_bits: $CFG_BITS" >&2; exit 1; }
[ "$CFG_GROUP" = "$GROUP_SIZE" ] || { echo "ERROR: --group_size $GROUP_SIZE but $CONFIG has qeft.group_size: $CFG_GROUP" >&2; exit 1; }

OUT_DIR="$(config_output_dir "$CONFIG")" || exit 1
FINAL_DIR="$OUT_DIR/final"
EVAL_DIR="${OUT_DIR}-${BITS}bit-qeft-dense-eval"
BASE_EVAL_DIR="${BASE_DIR}-${BITS}bit-qeftbase-eval"

echo "============================================================"
echo "  QEFT — Quantization for Efficient Fine-Tuning (baseline)"
echo "  Config:     $CONFIG"
echo "  Dataset:    $DATASET_NAME"
echo "  Bits:       INT${BITS} g${GROUP_SIZE}, k=${CFG_K} fp16 weak columns (asym, bcal)"
echo "  Base:       $BASE_DIR"
echo "  Output dir: $OUT_DIR"
echo "  GPUs:       $NUM_GPUS (train) / [$EVAL_GPUS] (eval)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0: the mixed-precision base (OGR + GPTQ around the fp16 weak columns)
# ---------------------------------------------------------------------------
if [ "$SKIP_BASE" = false ]; then
    echo -e "\n>>> Stage 0: QEFT base — global weak columns + GPTQ (balanced in-domain calibration)"
    CUDA_VISIBLE_DEVICES="${BASE_GPU:-0}" python "$QEFT_DIR/build_base.py" --config "$CONFIG"
else
    echo -e "\n>>> Stage 0: skipped by flag"
fi
[ -f "$BASE_DIR/qeft_meta.pt" ] || { echo "ERROR: no QEFT base at $BASE_DIR (build it with "\
    "jobs/qeft_prep_*.pbs, or drop --skip_base)"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 1: weak-column tuning (this repo's data recipe, QEFT's own lr)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: QEFT weak-column tuning"
    torchrun --nproc_per_node="$NUM_GPUS" --master_port=$((29500 + RANDOM % 1000)) \
        "$QEFT_DIR/train_commonsense.py" --config "$CONFIG" \
        ${RESUME_FROM:+--resume_from "$RESUME_FROM"}
fi
# Only assert the trained checkpoint exists when a later stage actually needs it. A prep-only
# invocation (--skip_train --skip_export --skip_eval, which is what jobs/qeft_prep_*.pbs runs) has
# nothing to check, and failing here would make the job exit non-zero — which silently leaves any
# `-W depend=afterok:` training job HELD forever.
if [ "$SKIP_EXPORT" = false ] || [ "$SKIP_EVAL" = false ]; then
    [ -f "$FINAL_DIR/qeft_weak_columns.safetensors" ] || {
        echo "ERROR: no trained weak columns at $FINAL_DIR"; exit 1; }
fi

# ---------------------------------------------------------------------------
# Stage 2: dense export (the scatter is checked per layer, not assumed)
# ---------------------------------------------------------------------------
if [ "$SKIP_EXPORT" = false ]; then
    echo -e "\n>>> Stage 2: dense export (+ per-layer equivalence check)"
    CUDA_VISIBLE_DEVICES=0 python "$QEFT_DIR/export_dense.py" \
        --ckpt "$FINAL_DIR" --out "$EVAL_DIR"

    if [ "$WITH_BASE" = true ] && [ ! -f "$BASE_EVAL_DIR/config.json" ]; then
        echo -e "\n>>> Stage 2b: dense export of the bare mixed-precision base (this row's floor)"
        CUDA_VISIBLE_DEVICES=0 python "$QEFT_DIR/export_dense.py" \
            --ckpt none --base_dir "$BASE_DIR" --out "$BASE_EVAL_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: generative evaluation through vLLM, then into the shared results table
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: Generative evaluation (vLLM)"
    for d in "$EVAL_DIR" "$BASE_EVAL_DIR"; do
        [ -d "$d" ] || continue
        bash runs/eval_vllm.sh \
            --model_path "$d" \
            --dataset    "$DATASET_NAME" \
            --gpus       "$EVAL_GPUS" \
            --tag        "$(basename "$d")" \
            ${EVAL_TASKS:+--sub_task $EVAL_TASKS}
    done

    # Idempotent — re-running the pipeline never duplicates rows. --filter qeft keeps this
    # invocation from re-ingesting other methods' summaries; _infer_method maps
    # "-qeft-dense-eval" to "QEFT" and "-qeftbase-eval" to "QEFT base (bcal GPTQ + fp16 weak)".
    echo -e "\n>>> Collecting results into $RESULTS_CSV"
    source ~/miniforge3/etc/profile.d/conda.sh
    conda run -n saltq python scripts/collect_saltq_results.py \
        --results_dir results \
        --csv         "$RESULTS_CSV" \
        --config      "$CONFIG" \
        --filter      qeft \
        --note        "${RESULTS_NOTE}"
fi

echo -e "\n============================================================"
echo "  QEFT pipeline complete!"
echo "============================================================"
