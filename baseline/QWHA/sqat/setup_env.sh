#!/bin/bash
# =============================================================================
# baseline/QWHA/sqat/setup_env.sh -- build the ISOLATED conda env for the QWHA baseline.
#
# QWHA pins torch 2.5.1 / transformers 4.51.3 / peft(fork) / gptqmodel 2.2.0. The training env
# (saltq) sits on torch 2.11 + transformers 5.14 and the vllm env on its own torch build, so
# QWHA gets a third env. Nothing in this env is ever imported by src/ -- the seam is the
# exported fp16 checkpoint, which runs/eval_vllm.sh reads in the vllm env.
#
# Built on the LOGIN node (internet, no GPU): every CUDA extension is cross-compiled for
# sm_80 (A100-SXM4-40GB, the only GPU in this cluster's gpu pool) via TORCH_CUDA_ARCH_LIST.
# =============================================================================
# NOT -u: conda-forge's compiler packages ship activate/deactivate hooks that read
# CONDA_BACKUP_CXX unguarded, and conda re-runs them after every `conda install` in the env.
set -eo pipefail

# Overridable so the env can be rebuilt anywhere; the defaults are this cluster's.
#   ENV_PREFIX            where the env goes (a prefix path, or export CONDA_ENV_NAME instead)
#   TORCH_CUDA_ARCH_LIST  compute capability to compile fast-hadamard-transform for.
#                         8.0 = A100, 8.6 = A6000/3090, 8.9 = L40S/4090, 9.0 = H100.
#   BUILD_GPTQMODEL_CUDA  1 to also build gptqmodel's CUDA kernels (not needed: QWHA only calls
#                         QuantLinear.dequantize_weight(), which the torch kernel implements)
ENV_PREFIX="${ENV_PREFIX:-/scratch/users/nus/jingzege/conda_envs/qwha}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
BUILD_GPTQMODEL_CUDA="${BUILD_GPTQMODEL_CUDA:-0}"
# The vendored upstream tree (this script lives in baseline/QWHA/sqat/).
QWHA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base 2>/dev/null || echo ~/miniforge3)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
fi
conda activate "$ENV_PREFIX"

# nvcc + CUDA headers for the two source builds below. 12.4 matches the torch wheel's cu124.
conda install -y -p "$ENV_PREFIX" -c nvidia/label/cuda-12.4.0 cuda-toolkit

# The login node ships GCC 8.5; torch 2.5's headers refuse anything below GCC 9 ("You're trying
# to build PyTorch with a too old version of GCC"), which is what killed the first attempt at
# fast-hadamard-transform. GCC 12 from conda-forge, used as BOTH the C++ compiler and nvcc's
# host compiler.
conda install -y -p "$ENV_PREFIX" -c conda-forge gxx_linux-64=12 gcc_linux-64=12

export CUDA_HOME="$ENV_PREFIX"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export MAX_JOBS="${MAX_JOBS:-16}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

pip install --upgrade pip setuptools wheel packaging ninja
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# pyproject.toml of the QWHA repo, minus the two source builds and minus openai (sft_utils
# imports openai_object, which only exists in openai<1; it is unused on our path).
pip install \
    accelerate==1.6.0 click==8.1.8 datasets==3.4.1 device-smi==0.4.1 fire==0.7.0 \
    huggingface-hub==0.31.1 joblib==1.4.2 lm-eval==0.4.8 logbar==0.0.4 \
    matplotlib==3.10.0 numpy==2.2.2 optimum==1.24.0 pandas==2.2.3 pillow==11.1.0 \
    pyarrow==19.0.1 pyyaml==6.0.2 regex==2024.11.6 requests==2.32.3 safetensors==0.5.3 \
    scikit-learn==1.6.1 scipy==1.15.2 sentencepiece==0.2.0 tabulate==0.9.0 \
    tokenicer==0.0.4 tokenizers==0.21.1 tqdm==4.67.1 transformers==4.51.3 \
    "triton>=3.1.0" wandb==0.19.9 threadpoolctl

# ~1 h of nvcc on the login node; sm_80 only (TORCH_CUDA_ARCH_LIST above), since that is the
# whole gpu pool.
pip install --no-build-isolation \
    "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.0.4.post1"

# gptqmodel WITHOUT its CUDA extensions. Its setup.py skips them when TORCH_CUDA_ARCH_LIST is
# unset and no GPU is visible -- which is the login node -- turning a long, fragile build into a
# pure-python install. QWHA only ever calls QuantLinear.dequantize_weight(), which the torch
# kernel implements for 2/3/4/8 bits; the smoke test is what proves that end to end.
if [ "$BUILD_GPTQMODEL_CUDA" = "1" ]; then
    pip install --no-build-isolation --no-deps gptqmodel==2.2.0
else
    # Hiding both the arch list AND the GPUs is what makes setup.py take the pure-python path:
    # it only skips the extensions when TORCH_CUDA_ARCH_LIST is unset AND torch sees no device.
    env -u TORCH_CUDA_ARCH_LIST CUDA_VISIBLE_DEVICES= \
        pip install --no-build-isolation --no-deps gptqmodel==2.2.0
fi

# --no-deps: the fork's setup.py pins a transformers range that would otherwise move the pinned
# 4.51.3 underneath the rest of the env.
pip install --no-build-isolation --no-deps -e "$QWHA_DIR/peft"

python - <<'PY'
import torch, transformers, peft, gptqmodel, fast_hadamard_transform
from peft import QWHAConfig
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| gptqmodel", gptqmodel.__version__)
print("QWHAConfig import OK; fast_hadamard_transform OK")
PY
echo "ENV BUILD OK: $ENV_PREFIX"
