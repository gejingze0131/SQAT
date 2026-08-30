#!/bin/bash
# =============================================================================
# runs/qwha/_pipeline.sh — QWHA baseline pipeline engine (base -> init -> train -> export -> eval)
#
# Not meant to be run directly — the per-task entry script fixes --dataset and the matching
# --config together, which is the pair that must never disagree:
#   runs/qwha/run_qwha_commonsense.sh
#
# QWHA (Jeon et al., arXiv:2509.17428), from the authors' own code under baseline/QWHA/. Where
# QA-LoRA folds a group-pooled adapter into the affine zero-points and SALT-Q trains real weights
# on the quantization grid, QWHA leaves the quantized weights frozen and trains a SPARSE SPECTRUM
# in the Walsh-Hadamard domain:
#
#     y = WHT(x) @ (WHT(W_q) + S)^T / in_features
#
# with S carrying rank*(in + out) trainable values per layer, placed by AdaAlloc where they
# reconstruct the quantization error best. There is no lossless merge into b-bit weights: the
# deployed model is INT-b codes PLUS that fp16 spectrum, so the row reads "b + fp16".
#
# ISOLATED ENV. Upstream pins torch 2.5.1 / transformers 4.51.3 / gptqmodel 2.2.0 and ships its
# own peft fork, which the `saltq` env cannot hold, so stages 0-3 run against $QWHA_ENV (built by
# baseline/QWHA/sqat/setup_env.sh) addressed by interpreter path rather than `conda activate` —
# the caller usually has `saltq` active for the collector. Stage 4 hops to `saltq` + `vllm-eval`
# on its own, inside runs/eval_vllm.sh.
#
# Stages:
#   Stage 0  The GPTQ base. NOT upstream's optimum/wikitext2 path: this repo measured generic
#            calibration as the thing that breaks low-bit bases here (INT2 floor 36.64 on the old
#            first-N BoolQ-only set vs 66.22 on the task-balanced 3500-record in-domain set; C4
#            128x2048 loses the instruction template outright at 30.67). make_bcal_base.py runs
#            OUR GPTQ on the balanced set and packs the grid into GPTQModel's format, asserting
#            the round trip. Shared per width; SKIPPED when it exists, and never rebuilt under a
#            trained adapter — the spectrum was initialized from one specific grid's error.
#   Stage 1  AdaAlloc initialization: quantization error + X^T X on the SAME balanced records,
#            then upstream's per-layer spectrum selection and value refinement. Also skipped when
#            its checkpoint exists.
#   Stage 2  Training, on this repo's data recipe (src/data.py's PROMPT, loss_span, collator and
#            shuffle seed), lr from QWHA's own paper. DDP, one full base per rank.
#   Stage 3  Dense fp16 export. A QWHA layer is exactly one linear map,
#            W_eff = iWHT(WHT(dequant(W_q)) + S), and the export CHECKS that per layer against
#            the live module before writing. --with_base also exports the bare quantized base:
#            this method builds its own base, so that number is its own floor.
#   Stage 4  Generative evaluation through vLLM (runs/eval_vllm.sh) — the same greedy generation
#            + exact match every other row in RESULTS_SUMMARY.md is scored with — then folded
#            into results_saltq.csv next to them.
#
# Usage:
#   bash runs/qwha/run_qwha_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/qwha/run_qwha_commonsense.sh --bits 2 --group_size 32 \
#        --config baseline/QWHA/sqat/configs/qwha_cs170k_int2_g32_ep1_span_bcal.yaml
#   bash runs/qwha/run_qwha_commonsense.sh --bits 3 --group_size 64 --skip_train   # export + eval
# =============================================================================

set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, and refuses a --config
# whose training task disagrees with --dataset. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
cd_repo_root

# QWHA rebuilds one dense [out, in] weight per projection on every forward (dequantize the codes,
# WHT them, add the sparse spectrum), so the allocator sees a steady churn of large short-lived
# blocks — the same reason runs/saltq/_pipeline.sh sets this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWHA_DIR="$REPO_ROOT/baseline/QWHA/sqat"
QWHA_ENV="${QWHA_ENV:-/scratch/users/nus/jingzege/conda_envs/qwha}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG=""
DATASET_NAME="commonsense"
EVAL_TASKS=""
BITS=""
GROUP_SIZE=""

BASE_ROOT="outputs/qwha_bases"
WITH_BASE=false           # also export + score the bare quantized base: this method's own floor
NUM_GPUS=4
EVAL_GPUS="0,1,2,3"

RESULTS_CSV="results_saltq.csv"
RESULTS_NOTE=""

SKIP_QUANT=false
SKIP_INIT=false
SKIP_TRAIN=false
SKIP_EXPORT=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";            shift 2 ;;
        --dataset)      DATASET_NAME="$2";      shift 2 ;;
        --bits)         BITS="$2";              shift 2 ;;
        --group_size)   GROUP_SIZE="$2";        shift 2 ;;
        --base_root)    BASE_ROOT="$2";         shift 2 ;;
        --with_base)    WITH_BASE=true;         shift ;;
        --num_gpus)     NUM_GPUS="$2";          shift 2 ;;
        --skip_quant)   SKIP_QUANT=true;        shift ;;
        --skip_init)    SKIP_INIT=true;         shift ;;
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
[ -x "$QWHA_ENV/bin/python" ] || {
    echo "ERROR: no QWHA env at $QWHA_ENV — run baseline/QWHA/sqat/setup_env.sh" >&2; exit 1; }

# Fail in two seconds rather than after a five-hour train and a meaningless score.
if [ "$SKIP_EVAL" = false ]; then
    assert_config_matches_dataset "$CONFIG" "$DATASET_NAME"
fi

# The isolated env, addressed by interpreter path. PYTHONPATH/PYTHONHOME would leak in from the
# caller's active env, and the compute nodes' Cray stack pulls an old /lib64/libstdc++ into the
# process ahead of the env's own — the same failure runs/eval_vllm.sh documents.
qwha_py() {
    env -u PYTHONPATH -u PYTHONHOME \
        LD_LIBRARY_PATH="$QWHA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$QWHA_ENV/bin/python" "$@"
}
qwha_torchrun() {
    env -u PYTHONPATH -u PYTHONHOME \
        LD_LIBRARY_PATH="$QWHA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$QWHA_ENV/bin/torchrun" "$@"
}

# The config decides what is trained; --bits/--group_size decide which base it is trained on. A
# mismatch would train a 2-bit spectrum against a 3-bit grid and never say so.
read -r CFG_BITS CFG_GROUP BASE_DIR INIT_DIR < <(qwha_py - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
q = c["qwha"]
print(c["model"]["quant_bits"], q["group_size"], q["gptq_base_dir"], q["init_ckpt_dir"])
PY
)
[ "$CFG_BITS" = "$BITS" ] || { echo "ERROR: --bits $BITS but $CONFIG has model.quant_bits: $CFG_BITS" >&2; exit 1; }
[ "$CFG_GROUP" = "$GROUP_SIZE" ] || { echo "ERROR: --group_size $GROUP_SIZE but $CONFIG has qwha.group_size: $CFG_GROUP" >&2; exit 1; }

OUT_DIR="$(config_output_dir "$CONFIG")" || exit 1
EVAL_DIR="${OUT_DIR}-${BITS}bit-qwha-dense-eval"
BASE_EVAL_DIR="${BASE_DIR}-${BITS}bit-qwhabase-eval"

echo "============================================================"
echo "  QWHA — Quantization-aware Walsh-Hadamard Adaptation (baseline)"
echo "  Config:     $CONFIG"
echo "  Dataset:    $DATASET_NAME"
echo "  Bits:       INT${BITS} g${GROUP_SIZE} (asymmetric, balanced in-domain calibration)"
echo "  GPTQ base:  $BASE_DIR"
echo "  Init ckpt:  $INIT_DIR"
echo "  Output dir: $OUT_DIR"
echo "  Env:        $QWHA_ENV"
echo "  GPUs:       $NUM_GPUS (train) / [$EVAL_GPUS] (eval)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0: the balanced-calibration GPTQ base (shared per width, never rebuilt under an adapter)
# ---------------------------------------------------------------------------
if [ "$SKIP_QUANT" = false ] && [ ! -f "$BASE_DIR/quantize_config.json" ]; then
    echo -e "\n>>> Stage 0: GPTQ base (this repo's grid, balanced in-domain calibration)"
    qwha_py "$QWHA_DIR/make_bcal_base.py" --config "$CONFIG" --out "$BASE_ROOT"
else
    echo -e "\n>>> Stage 0: base present at $BASE_DIR (skipped)"
fi
[ -f "$BASE_DIR/quantize_config.json" ] || { echo "ERROR: no quantized base at $BASE_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 1: AdaAlloc initialization on the same balanced records
# ---------------------------------------------------------------------------
if [ "$SKIP_INIT" = false ] && [ ! -f "$INIT_DIR/adapter_model.safetensors" ]; then
    echo -e "\n>>> Stage 1: QWHA AdaAlloc initialization (balanced in-domain X^T X)"
    qwha_py "$QWHA_DIR/init_adapter.py" --config "$CONFIG" --calib balanced
elif [ -f "$INIT_DIR/adapter_model.safetensors" ]; then
    echo -e "\n>>> Stage 1: initialized adapter present at $INIT_DIR (skipped)"
else
    echo -e "\n>>> Stage 1: skipped by flag, but $INIT_DIR does not exist"
fi
[ -f "$INIT_DIR/adapter_model.safetensors" ] || { echo "ERROR: no initialized adapter at $INIT_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 2: training (spectrum only, this repo's data recipe)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 2: QWHA training"
    qwha_torchrun --nproc_per_node="$NUM_GPUS" --master_port=$((29500 + RANDOM % 1000)) \
        "$QWHA_DIR/train_commonsense.py" --config "$CONFIG"
fi
[ -f "$OUT_DIR/adapter_model.safetensors" ] || { echo "ERROR: no trained adapter at $OUT_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 3: dense fp16 export (the identity is checked per layer, not assumed)
# ---------------------------------------------------------------------------
if [ "$SKIP_EXPORT" = false ]; then
    echo -e "\n>>> Stage 3: dense fp16 export (+ per-layer equivalence check)"
    CUDA_VISIBLE_DEVICES=0 qwha_py "$QWHA_DIR/export_dense.py" \
        --adapter_dir "$OUT_DIR" --out "$EVAL_DIR"

    if [ "$WITH_BASE" = true ] && [ ! -f "$BASE_EVAL_DIR/config.json" ]; then
        echo -e "\n>>> Stage 3b: dense export of the bare quantized base (this method's floor)"
        CUDA_VISIBLE_DEVICES=0 qwha_py "$QWHA_DIR/export_dense.py" \
            --adapter_dir none --config "$CONFIG" --out "$BASE_EVAL_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4: generative evaluation through vLLM, then into the shared results table
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 4: Generative evaluation (vLLM)"
    for d in "$EVAL_DIR" "$BASE_EVAL_DIR"; do
        [ -d "$d" ] || continue
        bash runs/eval_vllm.sh \
            --model_path "$d" \
            --dataset    "$DATASET_NAME" \
            --gpus       "$EVAL_GPUS" \
            --tag        "$(basename "$d")" \
            ${EVAL_TASKS:+--sub_task $EVAL_TASKS}
    done

    # Idempotent — re-running the pipeline never duplicates rows. --filter qwha keeps this
    # invocation from re-ingesting other methods' summaries; _infer_method maps
    # "-qwha-dense-eval" to "QWHA" and "-qwhabase-eval" to "QWHA base (bcal GPTQ)".
    echo -e "\n>>> Collecting results into $RESULTS_CSV"
    source ~/miniforge3/etc/profile.d/conda.sh
    conda run -n saltq python scripts/collect_saltq_results.py \
        --results_dir results \
        --csv         "$RESULTS_CSV" \
        --config      "$CONFIG" \
        --filter      qwha \
        --note        "${RESULTS_NOTE}"
fi

echo -e "\n============================================================"
echo "  QWHA pipeline complete!"
echo "============================================================"
