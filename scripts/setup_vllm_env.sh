#!/bin/bash
# =============================================================================
# setup_vllm_env.sh — build the conda env that run_eval_vllm.sh generates in.
#
# WHY A SEPARATE ENV. vLLM ships its own pinned torch. Installing it into the training env
# would move torch/transformers underneath every checkpoint already in outputs/, and a
# quantization method is exactly the kind of code that changes behaviour when the numerics
# under it change. The two envs never have to agree on anything: they exchange a plain HF
# checkpoint on disk.
#
# WHY CC=gcc. vLLM's llguidance dependency has no wheel for this platform and is built from
# source with Rust. The default `cc` here is the Cray clang wrapper, which injects
# `-plugin-opt=defaults=cray` and friends that rust-lld rejects; the build dies with
# "unknown plugin option 'lto=0'". Pointing Rust's linker at plain gcc is the whole fix.
#
#   bash scripts/setup_vllm_env.sh              # create env `vllm` with python 3.12
#   bash scripts/setup_vllm_env.sh myenv 3.12   # or name it yourself
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-vllm}"
PY_VERSION="${2:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Source it first:"
    echo "  source ~/miniforge3/etc/profile.d/conda.sh"
    exit 1
fi

set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> conda env '$ENV_NAME' already exists; installing into it."
else
    echo ">>> Creating conda env '$ENV_NAME' (python $PY_VERSION)"
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

set +u; conda activate "$ENV_NAME"; set -u

export CC=gcc CXX=g++
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc
export RUSTFLAGS="-C linker=gcc"

echo ">>> Installing vLLM (llguidance is built from source; this takes a while)"
pip install --no-cache-dir -r requirements-vllm.txt

python -c "import vllm; print('vllm', vllm.__version__)"

echo
echo "Done. Evaluate with:"
echo "  bash run_eval_vllm.sh --model_path <exported dir> --dataset commonsense"
