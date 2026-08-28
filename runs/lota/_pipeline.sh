#!/bin/bash
# =============================================================================
# runs/lota/_pipeline.sh — LoTA-QAF baseline pipeline engine (quantize -> train -> export -> eval)
#
# Not meant to be run directly — the per-task entry scripts fix --dataset and the
# matching --config together, which is the pair that must never disagree:
#   runs/lota/run_lota_commonsense.sh
#
# LoTA-QAF (Chen et al., NeurIPS'25 — arXiv:2505.18724), from the authors' own code under
# baseline/LoTA-QAF/. Where QA-LoRA folds a group-pooled adapter into the affine zero-points and
# SALT-Q trains real weights on the quantization grid, LoTA-QAF moves the INTEGER CODES: a
# ternary adapter's product dW = A_T B_T, thresholded at omega, adds +-1 grid step to every
# weight it selects, and the residual dW - omega*sign() is folded into a per-group offset. The
# merge is lossless by construction — the deployed model is
#
#     W = scales[g] * (q + markers - zeros[g] + mu[g])
#
# i.e. integer codes at the stated width plus a fractional zero-point shift, the same deployment
# contract the QA-LoRA rows carry.
#
# ISOLATED ENV. Upstream pins torch 2.6 / peft 0.15.1 / gptqmodel 2.1.1-dev, which the `saltq`
# env cannot hold, so stages 0-2 run against $LOTA_ENV (built by
# baseline/LoTA-QAF/sqat/setup_env.sh) addressed by interpreter path rather than `conda
# activate` — the caller usually has `saltq` active for the collector. Stage 3 hops to
# `saltq` + `vllm-eval` on its own, inside runs/eval_vllm.sh.
#
# Stages:
#   Stage 0  GPTQ base. GPTQModel with the paper's recipe (asymmetric, desc_act, 1024 C4
#            sequences). SKIPPED when the base dir exists — it is shared by every run at that
#            width, and rebuilding it under a trained adapter silently invalidates every
#            marker, which was learned against one specific integer grid.
#            --base_dir points at a base built some other way (see
#            baseline/LoTA-QAF/sqat/make_matched_base.py, which packs THIS repo's GPTQ grid
#            into GPTQModel's format).
#   Stage 1  Training. Ternary adapter + t-SignSGD, on this repo's data recipe: src/data.py's
#            PROMPT, loss_span, collator and shuffle seed, so only the method differs from the
#            SALT-Q and QA-LoRA rows. omega and sigma_t stay at the paper's values.
#   Stage 2  Merge + dense export. tests/test_export_consistency.py first: the deployed weight
#            must reproduce the trained CustomLoraLinear forward, or the score measures
#            neither. --with_base also exports the bare GPTQ base — this method trains on its
#            own grid, so the QLoRA-merged-then-GPTQ floor row is NOT its floor.
#   Stage 3  Generative evaluation through vLLM (runs/eval_vllm.sh) — the same greedy
#            generation + exact match every other row in RESULTS_SUMMARY.md is scored with,
#            then folded into results_saltq.csv next to them.
#
# Usage:
#   bash runs/lota/run_lota_commonsense.sh --bits 3 --group_size 64 --with_base
#   bash runs/lota/run_lota_commonsense.sh --bits 2 --group_size 32 \
#        --config baseline/LoTA-QAF/sqat/configs/lota_cs170k_int2_g32_ep1_span.yaml
#   bash runs/lota/run_lota_commonsense.sh --skip_train        # export + eval an existing adapter
#   bash runs/lota/run_lota_commonsense.sh --skip_eval         # quantize + train + export only
# =============================================================================

set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, and refuses a --config
# whose training task disagrees with --dataset. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
cd_repo_root

# LoTA-QAF rebuilds one dense [in, out] weight per projection on every forward (decode the
# codes, add the markers, add the offset), so the allocator sees a steady churn of large
# short-lived blocks — the same reason runs/saltq/_pipeline.sh sets this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOTA_DIR="$REPO_ROOT/baseline/LoTA-QAF/sqat"
LOTA_ENV="${LOTA_ENV:-/scratch/users/nus/jingzege/conda_envs/lota}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Empty => resolved from DATASET_NAME after parsing, so --config and --dataset cannot depend
# on the order they were passed in.
CONFIG=""
DATASET_NAME="commonsense"
EVAL_TASKS=""
BITS=""
GROUP_SIZE=""

BASE_ROOT="outputs/lota_bases"
BASE_DIR_OVERRIDE=""
WITH_BASE=false           # also export + score the bare GPTQ base: this method's own floor

EVAL_GPUS="0"             # GPUs vLLM evaluates on; >1 id => tensor-parallel

RESULTS_CSV="results_saltq.csv"
RESULTS_NOTE=""

SKIP_QUANT=false
SKIP_TRAIN=false
SKIP_EVAL=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";            shift 2 ;;
        --dataset)      DATASET_NAME="$2";      shift 2 ;;
        --bits)         BITS="$2";              shift 2 ;;
        --group_size)   GROUP_SIZE="$2";        shift 2 ;;
        --base_root)    BASE_ROOT="$2";         shift 2 ;;
        --base_dir)     BASE_DIR_OVERRIDE="$2"; SKIP_QUANT=true; shift 2 ;;
        --with_base)    WITH_BASE=true;         shift ;;
        --skip_quant)   SKIP_QUANT=true;        shift ;;
        --skip_train)   SKIP_TRAIN=true;        shift ;;
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
[ -x "$LOTA_ENV/bin/python" ] || {
    echo "ERROR: no LoTA-QAF env at $LOTA_ENV — run baseline/LoTA-QAF/sqat/setup_env.sh" >&2
    exit 1
}

# Fail in two seconds rather than after a five-hour train and a meaningless score. Only when
# this run will actually evaluate — a --skip_eval run is free to train on anything.
if [ "$SKIP_EVAL" = false ]; then
    assert_config_matches_dataset "$CONFIG" "$DATASET_NAME"
fi

# The isolated env, addressed by interpreter path. PYTHONPATH/PYTHONHOME would leak in from the
# caller's active env, and the compute nodes' Cray stack pulls an old /lib64/libstdc++ into the
# process ahead of the env's own — the same failure runs/eval_vllm.sh documents.
lota_py() {
    env -u PYTHONPATH -u PYTHONHOME \
        LD_LIBRARY_PATH="$LOTA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LOTA_ENV/bin/python" "$@"
}

# The config decides what is trained; --bits/--group_size decide which base it is trained on. A
# mismatch would train a 2-bit adapter against a 3-bit grid and never say so.
CFG_BITS="$(lota_py -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['quant_bits'])" "$CONFIG")"
if [ "$CFG_BITS" != "$BITS" ]; then
    echo "ERROR: --bits $BITS but $CONFIG has model.quant_bits: $CFG_BITS" >&2
    exit 1
fi
OUT_DIR="$(config_output_dir "$CONFIG")" || exit 1

# --base_dir points at a base built some other way — make_matched_base.py writes one carrying
# THIS repo's GPTQ grid, so the row can be read against SALT-Q and QA-LoRA with the starting
# grid held fixed instead of only against its own floor.
BASE_DIR="${BASE_DIR_OVERRIDE:-${BASE_ROOT}/Llama-2-7B_int${BITS}_${GROUP_SIZE}_asym}"
ADAPTER_DIR="${OUT_DIR}/final"
EVAL_DIR="${OUT_DIR}-${BITS}bit-lota-dequant-eval"
BASE_EVAL_DIR="${BASE_DIR}-${BITS}bit-lotabase-eval"

echo "============================================================"
echo "  LoTA-QAF — Lossless Ternary Adaptation for QAF (baseline)"
echo "  Config:     $CONFIG"
echo "  Dataset:    $DATASET_NAME"
echo "  Bits:       INT${BITS} g${GROUP_SIZE} (asymmetric, desc_act)"
echo "  GPTQ base:  $BASE_DIR"
echo "  Output dir: $OUT_DIR"
echo "  Env:        $LOTA_ENV"
echo "  Eval GPUs:  [$EVAL_GPUS]"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0: the GPTQ base (shared per width; never rebuilt under a trained adapter)
# ---------------------------------------------------------------------------
if [ "$SKIP_QUANT" = false ] && [ ! -f "$BASE_DIR/quantize_config.json" ]; then
    echo -e "\n>>> Stage 0: quantizing the GPTQ base (GPTQModel, the paper's recipe)"
    lota_py "$LOTA_DIR/quantize_base.py" \
        --bits "$BITS" --group_size "$GROUP_SIZE" --out "$BASE_ROOT" --cpu_cache
else
    echo -e "\n>>> Stage 0: base present at $BASE_DIR (skipped)"
fi
[ -f "$BASE_DIR/quantize_config.json" ] || { echo "ERROR: no quantized base at $BASE_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 1: training (ternary adapter + t-SignSGD, this repo's data recipe)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: LoTA-QAF training"
    lota_py "$LOTA_DIR/train_lota.py" --config "$CONFIG" --quantized_model_dir "$BASE_DIR"
fi
[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || {
    echo "ERROR: no trained adapter at $ADAPTER_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 2: merge into the integer codes, then export a dense fp16 checkpoint
#
# The consistency check runs FIRST and is not optional: training runs upstream's
# CustomLoraLinear forward while the reported number comes from the dense checkpoint, so if the
# two disagree the score describes neither. It also fails when no markers fired at any omega —
# a check that never crossed the threshold has said nothing about the integer merge.
# ---------------------------------------------------------------------------
echo -e "\n>>> Stage 2: merging the ternary adaptation into the integer codes"
lota_py "$LOTA_DIR/tests/test_export_consistency.py" \
    --quantized_model_dir "$BASE_DIR" --adapter_dir "$ADAPTER_DIR"
lota_py "$LOTA_DIR/export_lota_dense.py" \
    --quantized_model_dir "$BASE_DIR" --adapter_dir "$ADAPTER_DIR" --out "$EVAL_DIR"

if [ "$WITH_BASE" = true ] && [ ! -f "$BASE_EVAL_DIR/config.json" ]; then
    echo -e "\n>>> Stage 2b: dense export of the bare GPTQ base (this method's own floor)"
    lota_py "$LOTA_DIR/export_lota_dense.py" \
        --quantized_model_dir "$BASE_DIR" --adapter_dir none --out "$BASE_EVAL_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: generative evaluation through vLLM (separate conda env)
#
# The model is scored on its OWN generations through runs/eval_vllm.sh — the identical seam the
# SALT-Q and QA-LoRA rows go through, so the number lands in the same table without an argument
# about the harness. Upstream instead scores through gptqmodel's kernels under lm-eval; see
# baseline/LoTA-QAF/sqat/PROVENANCE.md for why that path is not used here.
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

    # Fold the per-run summaries into the SAME long-format table SALT-Q writes to. Idempotent —
    # re-running the pipeline never duplicates rows. --filter lota keeps this invocation from
    # re-ingesting other methods' summaries; _infer_method maps "-lota-dequant-eval" to
    # "LoTA-QAF" and "-lotabase-eval" to "LoTA-QAF base (GPTQModel)".
    echo -e "\n>>> Collecting results into $RESULTS_CSV"
    source ~/miniforge3/etc/profile.d/conda.sh
    conda run -n saltq python scripts/collect_saltq_results.py \
        --results_dir results \
        --csv         "$RESULTS_CSV" \
        --config      "$CONFIG" \
        --filter      lota \
        --note        "${RESULTS_NOTE}"
fi

echo -e "\n============================================================"
echo "  LoTA-QAF pipeline complete!"
echo "============================================================"
