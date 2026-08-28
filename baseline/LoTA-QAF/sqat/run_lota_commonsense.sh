#!/bin/bash
# =============================================================================
# baseline/LoTA-QAF/run_lota_commonsense.sh — the LoTA-QAF row, end to end
#
# Four stages, in the order their failure modes matter:
#
#   0  QUANTIZE   GPTQModel, the paper's recipe (asym, desc_act, 1024 C4 sequences). Skipped
#                 if the base dir already exists — it is shared by every run at that width and
#                 rebuilding it would silently change what the adapter was trained against.
#   1  TRAIN      ternary adapter + t-SignSGD, this repo's data recipe (train_lota.py).
#   2  EXPORT     merge the ternary adaptation into the integer codes and write a dense fp16
#                 checkpoint (export_lota_dense.py). --with_base also exports the bare GPTQ
#                 base, which is this method's own floor and not our GPTQ floor row.
#   3  EVALUATE   ../../runs/eval_vllm.sh — the SAME greedy vLLM generation + exact match every
#                 other row in RESULTS_SUMMARY.md is scored with, then folded into
#                 results_saltq.csv next to them.
#
# Stages 0-2 run in the isolated `lota` env (torch 2.6 / peft 0.15.1 / gptqmodel 2.1.1-dev,
# see setup_env.sh); stage 3 hops to `saltq` + `vllm-eval` on its own.
#
#   bash run_lota_commonsense.sh --config configs/lota_cs170k_int3_g64_ep1_span.yaml --bits 3 --group_size 64
#   bash run_lota_commonsense.sh --config configs/lota_cs170k_int2_g32_ep1_span.yaml --bits 2 --group_size 32 --with_base
#   bash run_lota_commonsense.sh --config ... --skip_train      # export + eval an existing adapter
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQAT_ROOT="$(cd "$HERE/../../.." && pwd)"
LOTA_ENV="${LOTA_ENV:-/scratch/users/nus/jingzege/conda_envs/lota}"

CONFIG=""
BITS=""
GROUP_SIZE=""
BASE_ROOT="outputs/lota_bases"
SKIP_QUANT=false
SKIP_TRAIN=false
SKIP_EVAL=false
WITH_BASE=false
EVAL_GPUS="0"
RESULTS_NOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --bits)        BITS="$2";        shift 2 ;;
        --group_size)  GROUP_SIZE="$2";  shift 2 ;;
        --base_root)   BASE_ROOT="$2";   shift 2 ;;
        --skip_quant)  SKIP_QUANT=true;  shift ;;
        --skip_train)  SKIP_TRAIN=true;  shift ;;
        --skip_eval)   SKIP_EVAL=true;   shift ;;
        --with_base)   WITH_BASE=true;   shift ;;
        --eval_gpus)   EVAL_GPUS="$2";   shift 2 ;;
        --note)        RESULTS_NOTE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -n "$CONFIG" ]     || { echo "ERROR: --config is required"; exit 1; }
[ -n "$BITS" ]       || { echo "ERROR: --bits is required"; exit 1; }
[ -n "$GROUP_SIZE" ] || { echo "ERROR: --group_size is required"; exit 1; }
# --config is written relative to this directory (configs/lota_*.yaml) but the job runs from
# the repo root, so resolve it here rather than depending on the caller's cwd.
if [ -f "$HERE/$CONFIG" ]; then
    CONFIG="$HERE/$CONFIG"
elif [ ! -f "$CONFIG" ]; then
    echo "ERROR: no such config: $CONFIG" >&2
    exit 1
fi
[ -x "$LOTA_ENV/bin/python" ] || { echo "ERROR: no lota env at $LOTA_ENV; run setup_env.sh" >&2; exit 1; }

# The isolated env is addressed by its interpreter path rather than `conda activate`, so the
# caller can have `saltq` active (the PBS jobs do, for the collector). PYTHONPATH/PYTHONHOME
# would still leak across, and the compute nodes' Cray stack pulls an old /lib64/libstdc++ into
# the process ahead of the env's own -- the same failure runs/eval_vllm.sh documents.
lota_py() {
    env -u PYTHONPATH -u PYTHONHOME \
        LD_LIBRARY_PATH="$LOTA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LOTA_ENV/bin/python" "$@"
}


# The config decides what is trained; --bits/--group_size decide which base it is trained on.
# A mismatch would train a 2-bit adapter against a 3-bit grid and never say so.
CFG_BITS="$(lota_py -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['quant_bits'])" "$CONFIG")"
if [ "$CFG_BITS" != "$BITS" ]; then
    echo "ERROR: --bits $BITS but $CONFIG has model.quant_bits: $CFG_BITS" >&2
    exit 1
fi
OUT_DIR="$(lota_py -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['training']['output_dir'])" "$CONFIG")"

cd "$SQAT_ROOT"
BASE_DIR="${BASE_ROOT}/Llama-2-7B_int${BITS}_${GROUP_SIZE}_asym"
ADAPTER_DIR="${OUT_DIR}/final"
EVAL_DIR="${OUT_DIR}-${BITS}bit-lota-dequant-eval"
BASE_EVAL_DIR="${BASE_ROOT}/Llama-2-7B_int${BITS}_${GROUP_SIZE}_asym-${BITS}bit-lotabase-eval"

echo "============================================================"
echo "  LoTA-QAF pipeline"
echo "  Config:   $CONFIG"
echo "  Bits:     INT${BITS} g${GROUP_SIZE} (asymmetric, desc_act)"
echo "  Base:     $BASE_DIR"
echo "  Output:   $OUT_DIR"
echo "  Env:      $LOTA_ENV"
echo "============================================================"

# --- Stage 0: GPTQ base ------------------------------------------------------
if [ "$SKIP_QUANT" = false ] && [ ! -f "$BASE_DIR/quantize_config.json" ]; then
    echo -e "\n>>> Stage 0: quantizing the GPTQ base (GPTQModel, 1024 C4 sequences)"
    lota_py "$HERE/quantize_base.py" --bits "$BITS" --group_size "$GROUP_SIZE" --out "$BASE_ROOT"
else
    echo -e "\n>>> Stage 0: base present at $BASE_DIR (skipped)"
fi
[ -f "$BASE_DIR/quantize_config.json" ] || { echo "ERROR: no quantized base at $BASE_DIR"; exit 1; }

# --- Stage 1: training -------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n>>> Stage 1: LoTA-QAF training"
    lota_py "$HERE/train_lota.py" --config "$CONFIG" --quantized_model_dir "$BASE_DIR"
fi
[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || {
    echo "ERROR: no trained adapter at $ADAPTER_DIR"; exit 1; }

# --- Stage 2: merge + dense export ------------------------------------------
echo -e "\n>>> Stage 2: merging the ternary adaptation into the integer codes"
lota_py "$HERE/tests/test_export_consistency.py" \
    --quantized_model_dir "$BASE_DIR" --adapter_dir "$ADAPTER_DIR"
lota_py "$HERE/export_lota_dense.py" \
    --quantized_model_dir "$BASE_DIR" --adapter_dir "$ADAPTER_DIR" --out "$EVAL_DIR"

if [ "$WITH_BASE" = true ] && [ ! -f "$BASE_EVAL_DIR/config.json" ]; then
    echo -e "\n>>> Stage 2b: dense export of the bare GPTQ base (this method's own floor)"
    lota_py "$HERE/export_lota_dense.py" \
        --quantized_model_dir "$BASE_DIR" --adapter_dir none --out "$BASE_EVAL_DIR"
fi

# --- Stage 3: the shared generative evaluation -------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Stage 3: generative evaluation (vLLM), the shared harness"
    for d in "$EVAL_DIR" "$BASE_EVAL_DIR"; do
        [ -d "$d" ] || continue
        bash runs/eval_vllm.sh --model_path "$d" --dataset commonsense \
            --gpus "$EVAL_GPUS" --tag "$(basename "$d")"
    done

    echo -e "\n>>> Collecting into results_saltq.csv"
    source ~/miniforge3/etc/profile.d/conda.sh
    conda run -n saltq python scripts/collect_saltq_results.py \
        --results_dir results --csv results_saltq.csv --config "$CONFIG" \
        --filter lota --note "$RESULTS_NOTE"
fi

echo -e "\n============================================================"
echo "  LoTA-QAF pipeline complete"
echo "============================================================"
