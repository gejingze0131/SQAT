#!/bin/bash
# =============================================================================
# baseline/LoTA-QAF/setup_env.sh — build the isolated env this baseline runs in
#
# LoTA-QAF is pinned to a stack the SQAT training env cannot hold: torch 2.6.0, peft 0.15.1
# and gptqmodel 2.1.1-dev (upstream requirements.txt / environment.yml). The `saltq` env runs
# torch 2.11 + peft 0.20, and every checkpoint in outputs/ was produced with it — so this
# baseline gets its own prefix, the same way `vllm-eval` does for evaluation.
#
# Two of the three pieces are VENDORED as editable checkouts under vendor/, because LoTA-QAF
# does not ship importable modules: LoTA/layer.py and LoTA/adapter.py are fragments whose
# docstrings name the library file they belong in. patches/apply_lota_patches.py splices them
# in mechanically, so the modification is a reviewable diff in a checkout at a pinned commit
# rather than an untracked edit to site-packages.
#
# Run on a LOGIN node (no GPU visible): GPTQModel's setup.py probes for a CUDA device and
# skips its CUDA extensions when it finds none, which is what we want — the TORCH and TRITON
# backends are pure PyTorch/Triton, and there is no nvcc on this cluster's login nodes anyway.
#
#   bash baseline/LoTA-QAF/sqat/setup_env.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # baseline/LoTA-QAF/sqat
UPSTREAM="$(cd "$HERE/.." && pwd)"                        # the LoTA-QAF checkout
PREFIX="${LOTA_ENV:-/scratch/users/nus/jingzege/conda_envs/lota}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/users/nus/jingzege/cache/pip}"

# Pinned upstreams. GPTQModel has no 2.1.1-dev tag; this is the last commit on main carrying
# __version__ = "2.1.1-dev", i.e. what `gptqmodel==2.1.1.dev0` in requirements.txt resolved to.
GPTQMODEL_COMMIT="8cd2e26220aac6c2aa40573322c168357e472c5f"
PEFT_TAG="v0.15.1"

source ~/miniforge3/etc/profile.d/conda.sh

echo "=== [1/6] conda env at $PREFIX ==="
[ -d "$PREFIX" ] || conda create -y -p "$PREFIX" python=3.10

echo "=== [2/6] torch 2.6.0 (cu124) ==="
conda run -p "$PREFIX" pip install --no-input torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/6] pinned deps (upstream requirements.txt versions) ==="
# flash_attn and vllm are deliberately absent: the model runs with sdpa here, and evaluation
# happens in the `vllm-eval` env through the shared harness.
conda run -p "$PREFIX" pip install --no-input \
  transformers==4.51.3 tokenizers==0.21.1 accelerate==1.5.2 datasets==3.5.0 \
  safetensors==0.5.3 huggingface_hub==0.30.0 numpy==2.1.3 sentencepiece==0.2.0 \
  protobuf==6.30.2 threadpoolctl==3.6.0 device-smi==0.4.1 hf_transfer==0.1.9 \
  random_word==1.0.13 tokenicer==0.0.4 logbar==0.0.4 pillow==11.1.0 \
  packaging==24.2 pyyaml==6.0.2 scipy==1.15.2 tqdm==4.67.1

echo "=== [4/6] vendored peft $PEFT_TAG ==="
if [ ! -d "$UPSTREAM/vendor/peft" ]; then
    mkdir -p "$UPSTREAM/vendor"
    git clone --filter=blob:none --branch "$PEFT_TAG" --depth 1 \
        https://github.com/huggingface/peft.git "$UPSTREAM/vendor/peft"
fi
conda run -p "$PREFIX" pip install --no-input --no-deps -e "$UPSTREAM/vendor/peft"

echo "=== [5/6] vendored gptqmodel @ $GPTQMODEL_COMMIT ==="
if [ ! -d "$UPSTREAM/vendor/GPTQModel" ]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/ModelCloud/GPTQModel.git "$UPSTREAM/vendor/GPTQModel"
fi
git -C "$UPSTREAM/vendor/GPTQModel" checkout -q "$GPTQMODEL_COMMIT"
( cd "$UPSTREAM/vendor/GPTQModel" && conda run -p "$PREFIX" pip install --no-input --no-deps --no-build-isolation -e . )

echo "=== [6/6] LoTA-QAF patches ==="
conda run -p "$PREFIX" python "$HERE/patches/apply_lota_patches.py"

conda run -p "$PREFIX" python -c "
import torch, transformers, peft, gptqmodel, triton
from peft.tuners.lora.layer import CustomLoraLinear, IntLinear
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| peft', peft.__version__, '| gptqmodel', gptqmodel.__version__, '| triton', triton.__version__)
print('LoTA patch OK:', CustomLoraLinear.__name__, IntLinear.__name__)
"
echo "env ready: $PREFIX"
