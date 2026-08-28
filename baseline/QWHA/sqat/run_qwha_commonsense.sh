#!/bin/bash
# =============================================================================
# One QWHA cell, end to end: GPTQ base -> AdaAlloc init -> fine-tune -> dense export -> eval.
#
#   bash baseline/QWHA/sqat/run_qwha_commonsense.sh --config <yaml> [--tag <name>]
#
# Every stage is idempotent (it skips when its output already exists), so a job that runs out of
# walltime can simply be resubmitted. Training runs in the isolated `qwha` env; evaluation hands
# the dense fp16 export to runs/eval_vllm.sh, which hops to the vllm env on its own -- the same
# path, prompts and extractors as every other row in RESULTS_SUMMARY.md.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SQAT_DIR="$REPO_ROOT/baseline/QWHA/sqat"
QWHA_ENV="${QWHA_ENV:-/scratch/users/nus/jingzege/conda_envs/qwha}"
export QWHA_CACHE_PATH="${QWHA_CACHE_PATH:-/scratch/users/nus/jingzege/SQAT_outputs/qwha_cache}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

CONFIG=""; TAG=""; NGPU="${NGPU:-4}"
SKIP_QUANT=false; SKIP_INIT=false; SKIP_TRAIN=false; SKIP_EXPORT=false; SKIP_EVAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --ngpu)        NGPU="$2"; shift 2 ;;
        --skip_quant)  SKIP_QUANT=true; shift ;;
        --skip_init)   SKIP_INIT=true; shift ;;
        --skip_train)  SKIP_TRAIN=true; shift ;;
        --skip_export) SKIP_EXPORT=true; shift ;;
        --skip_eval)   SKIP_EVAL=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$CONFIG" ] || { echo "ERROR: --config is required"; exit 1; }
cd "$REPO_ROOT"
[ -f "$CONFIG" ] || { echo "ERROR: no such config: $CONFIG"; exit 1; }

read -r BITS GROUP RANK OUT_DIR < <(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["model"]["quant_bits"], c["qwha"]["group_size"], c["qwha"]["rank"],
      c["training"]["output_dir"])
PY
)
[ -n "$TAG" ] || TAG="$(basename "$OUT_DIR")"
DENSE_DIR="${OUT_DIR}-dense"

echo "=============================================================="
echo "  QWHA baseline — INT${BITS} g${GROUP} rank${RANK}"
echo "  config     $CONFIG"
echo "  adapter    $OUT_DIR"
echo "  dense      $DENSE_DIR"
echo "  cache      $QWHA_CACHE_PATH"
echo "=============================================================="

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$QWHA_ENV"
set -u

if [ "$SKIP_QUANT" = false ]; then
    echo ">>> Stage 1/5: plain GPTQ INT${BITS} g${GROUP} base"
    python "$SQAT_DIR/quantize_base.py" -b "$BITS" -g "$GROUP"
fi

if [ "$SKIP_INIT" = false ]; then
    echo ">>> Stage 2/5: QWHA AdaAlloc initialization"
    INIT_DIR="$(python -c "
import sys; sys.path.insert(0, '$SQAT_DIR')
from qwha_common import init_ckpt_dir
print(init_ckpt_dir('meta-llama/Llama-2-7b-hf', $BITS, $GROUP, $RANK))")"
    if [ -f "$INIT_DIR/adapter_model.safetensors" ]; then
        echo "    $INIT_DIR exists — skipping"
    else
        python "$SQAT_DIR/init_adapter.py" -b "$BITS" -g "$GROUP" -r "$RANK"
    fi
fi

if [ "$SKIP_TRAIN" = false ]; then
    echo ">>> Stage 3/5: fine-tuning on Commonsense-170k (${NGPU} GPUs)"
    torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + RANDOM % 1000)) \
        "$SQAT_DIR/train_commonsense.py" --config "$CONFIG"
fi

if [ "$SKIP_EXPORT" = false ]; then
    echo ">>> Stage 4/5: dense fp16 export (+ equivalence check)"
    CUDA_VISIBLE_DEVICES=0 python "$SQAT_DIR/export_dense.py" \
        --adapter_dir "$OUT_DIR" --out "$DENSE_DIR"
fi

set +u; conda deactivate; set -u

if [ "$SKIP_EVAL" = false ]; then
    echo ">>> Stage 5/5: generative evaluation on the eight commonsense test sets"
    bash runs/eval_vllm.sh --model_path "$DENSE_DIR" --dataset commonsense --tag "$TAG"
fi
echo "QWHA cell done: $TAG"
