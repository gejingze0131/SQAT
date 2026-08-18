#!/bin/bash
# =============================================================================
# runs/eval_vllm.sh — generative evaluation of an exported model, in a separate conda env
#
# The training env (saltq) cannot hold vLLM: vLLM pins its own torch build, and installing it
# next to the training stack would silently move the torch/transformers versions every trained
# checkpoint in outputs/ was produced with. So evaluation runs in its own env and this script
# is the seam — it hops envs, it is not meant to be sourced into a training process.
#
# Three stages, in order, because each one's failure mode is invisible in the next:
#
#   1. FOLD (train env). A SALT-Q / SQAT-Permute export is not a self-contained Llama: its
#      residual stream is permuted per segment and needs a BoundaryGatherHook at every
#      boundary. vLLM cannot register one, so the checkpoint has to be folded back to the
#      original channel order first (scripts/export_vllm_ready.py — exact, verified by
#      scripts/test_unpermute_fold.py). Checkpoints with no perm metadata are copied through.
#   2. GENERATE (vllm env). Greedy sampling on datasets/<name>/test.json, whose `instruction`
#      is already the full training prompt.
#   3. SCORE. Exact-match through the reference extractors in scripts/test_acc.py.
#
# Usage:
#   bash runs/eval_vllm.sh --model_path outputs/saltq-3bit-saltq-deploy-eval --dataset commonsense
#   bash runs/eval_vllm.sh --model_path <dir> --dataset math --tag saltq_int2_g32
#   bash runs/eval_vllm.sh --model_path <dir> --sub_task boolq piqa      # one-off subset
#
# Outputs (under --output_dir, default results/<dataset>_vllm/):
#   <tag>.jsonl   one {type, query, output, answer} per test record
#   <tag>.json    per-task accuracy + unweighted mean
# =============================================================================

set -euo pipefail

# Addresses configs/ scripts/ outputs/ datasets/ from the repo root, so the caller's cwd does
# not matter. See runs/lib/common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd_repo_root

MODEL_PATH=""
DATASET_NAME="commonsense"        # commonsense | math
TAG=""
OUTPUT_DIR=""
SUB_TASK=""
GPUS=""                           # empty => inherit CUDA_VISIBLE_DEVICES
MAX_TOKENS=1024
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.90
TRAIN_ENV="${TRAIN_ENV:-saltq}"
VLLM_ENV="${VLLM_ENV:-vllm-eval}"
FORCE_FOLD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)     MODEL_PATH="$2";    shift 2 ;;
        --dataset)        DATASET_NAME="$2";  shift 2 ;;
        --tag)            TAG="$2";           shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --gpus)           GPUS="$2";          shift 2 ;;
        --max_tokens)     MAX_TOKENS="$2";    shift 2 ;;
        --max_model_len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu_mem_util)   GPU_MEM_UTIL="$2";  shift 2 ;;
        --train_env)      TRAIN_ENV="$2";     shift 2 ;;
        --vllm_env)       VLLM_ENV="$2";      shift 2 ;;
        --force_fold)     FORCE_FOLD=true;    shift ;;
        --sub_task)       shift
                          while [[ $# -gt 0 && "$1" != --* ]]; do
                              SUB_TASK="$SUB_TASK $1"; shift
                          done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -n "$MODEL_PATH" ] || { echo "ERROR: --model_path is required"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "ERROR: no such model dir: $MODEL_PATH"; exit 1; }

case "$DATASET_NAME" in
    commonsense) DATA_PATH="datasets/commonsense" ;;
    math)        DATA_PATH="datasets/metamath" ;;
    *) echo "ERROR: --dataset must be 'commonsense' or 'math' (got '$DATASET_NAME')"; exit 1 ;;
esac
[ -d "$DATA_PATH" ] || { echo "ERROR: missing dataset dir $DATA_PATH"; exit 1; }

MODEL_PATH="${MODEL_PATH%/}"
[ -n "$TAG" ]        || TAG="$(basename "$MODEL_PATH")"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="results/${DATASET_NAME}_vllm"
[ -n "$GPUS" ]       && export CUDA_VISIBLE_DEVICES="$GPUS"

VLLM_MODEL_DIR="${MODEL_PATH}-vllm"
RESPONSE_FILE="${OUTPUT_DIR}/${TAG}.jsonl"
SUMMARY_FILE="${OUTPUT_DIR}/${TAG}.json"

mkdir -p "$OUTPUT_DIR"

# `conda activate` reads variables (PS1, ...) that `set -u` treats as fatal, so every hop is
# wrapped. Failing to find conda here is a setup error, not something to discover halfway
# through a 20-minute generation run.
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Source it first:"
    echo "  source ~/miniforge3/etc/profile.d/conda.sh"
    exit 1
fi
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

# The compute nodes start with a Cray/system LD_LIBRARY_PATH already populated, and something
# in that stack pulls /lib64/libstdc++.so.6 (CXXABI up to 1.3.11) into the process before the
# env's own (1.3.17) is reached. Once a soname is loaded the loader reuses it, so the env's
# libicui18n — pulled in by sqlite3, which vLLM's engine imports — then fails to resolve
# CXXABI_1.3.15 and the whole engine import dies. Putting the active env's lib dir first makes
# the FIRST libstdc++ loaded the new one. Rebuilt from the pristine value on every hop so the
# two envs cannot leak libraries into each other.
_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
activate_env() {
    set +u
    conda activate "$1"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${_ORIG_LD_LIBRARY_PATH:+:$_ORIG_LD_LIBRARY_PATH}"
    set -u
}
deactivate_env() { set +u; conda deactivate; set -u; }

echo "============================================================"
echo "  Generative evaluation (vLLM)"
echo "  Model:     $MODEL_PATH"
echo "  Dataset:   $DATASET_NAME  ($DATA_PATH/test.json)"
echo "  Sub-tasks: ${SUB_TASK:-all}"
echo "  Envs:      fold=$TRAIN_ENV  generate=$VLLM_ENV"
echo "  Output:    $RESPONSE_FILE"
echo "============================================================"

# --- Stage 1: fold the residual permutation (train env) ---------------------
echo -e "\n>>> Stage 1/3: preparing a hook-free checkpoint"
activate_env "$TRAIN_ENV"
FOLD_FLAGS=""
[ "$FORCE_FOLD" = true ] && FOLD_FLAGS="--force"
python scripts/export_vllm_ready.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$VLLM_MODEL_DIR" \
    $FOLD_FLAGS
deactivate_env

# --- Stage 2: generate (vllm env) -------------------------------------------
echo -e "\n>>> Stage 2/3: generating with vLLM"
activate_env "$VLLM_ENV"

# Two seconds here instead of discovering a broken env after the model is on the GPUs. Both
# things that have actually broken are covered: vLLM's engine imports sqlite3, which drags in
# the env's libicui18n and its libstdc++ requirement, and a torch built for CUDA 13 kills every
# worker at engine start on this cluster's 12.8 driver.
# vllm._C is imported EXPLICITLY: `import vllm` succeeds on a wheel built against the wrong
# CUDA and only the compiled extension fails, so a package-level import proves nothing.
if ! python -c "import sqlite3, torch, vllm, vllm._C; from vllm.v1.engine.core_client import EngineCoreClient; assert torch.version.cuda.startswith('12.')" 2>/dev/null; then
    echo "ERROR: the '$VLLM_ENV' env cannot import vLLM's engine on this node." >&2
    echo "  CONDA_PREFIX     = ${CONDA_PREFIX:-unset}" >&2
    echo "  LD_LIBRARY_PATH  = ${LD_LIBRARY_PATH:-unset}" >&2
    echo "  --- full traceback ---" >&2
    python -c "import sqlite3, torch, vllm, vllm._C; from vllm.v1.engine.core_client import EngineCoreClient; assert torch.version.cuda.startswith('12.')" >&2
    exit 1
fi
# shellcheck disable=SC2086
python scripts/gen_vllm.py \
    --model          "$VLLM_MODEL_DIR" \
    --data_path      "$DATA_PATH" \
    --dataset_split  test \
    --output_file    "$RESPONSE_FILE" \
    --max_tokens     "$MAX_TOKENS" \
    --max_model_len  "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    ${SUB_TASK:+--sub_task $SUB_TASK}

# --- Stage 3: score ---------------------------------------------------------
# Pure stdlib, so it runs in whichever env is active.
echo -e "\n>>> Stage 3/3: scoring"
python scripts/test_acc.py \
    --input_file  "$RESPONSE_FILE" \
    --output_json "$SUMMARY_FILE" \
    --dataset     "$DATASET_NAME" \
    --model_path  "$MODEL_PATH"
deactivate_env

echo -e "\n============================================================"
echo "  Done. Summary: $SUMMARY_FILE"
echo "============================================================"
