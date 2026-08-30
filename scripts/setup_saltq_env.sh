#!/bin/bash
# =============================================================================
# setup_saltq_env.sh — build the conda env that training, the offline permute + GPTQ pre-steps,
# export and result collection run in (`saltq`). Evaluation lives in a SEPARATE env built by
# scripts/setup_vllm_env.sh; see README.md "Two conda envs, on purpose".
#
# Pins are the versions every number in results_saltq.csv was produced with (2026-08). torch comes
# from the cu128 index EXCLUSIVELY so the CUDA runtime wheels match the cluster's driver; the rest
# is pinned on PyPI. A newer torch/transformers is not "probably fine" for a quantization method —
# the fakequant numerics and the GPTQ Hessians move with them.
#
#   bash scripts/setup_saltq_env.sh              # create env `saltq`, python 3.11
#   bash scripts/setup_saltq_env.sh myenv 3.11   # or name it yourself
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-saltq}"
PY_VERSION="${2:-3.11}"

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

# PHASE 1 — the CUDA-matched runtime, from the cu128 index only (--index-url, not --extra-index-url,
# so pip cannot fall back to the CPU wheel on PyPI).
pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# PHASE 2 — the training stack, pinned. bitsandbytes is the NF4 path of the QLoRA bounds; the SALT-Q
# / QA-LoRA / mixed-precision code needs only torch + transformers + accelerate + peft + safetensors.
pip install --no-cache-dir \
    transformers==5.14.1 \
    accelerate==1.14.0 \
    peft==0.20.0 \
    bitsandbytes==0.50.0 \
    datasets==5.0.1 \
    safetensors==0.8.0 \
    numpy==2.4.6 \
    scipy==1.17.1 \
    matplotlib==3.11.1 \
    pyyaml==6.0.3 \
    wandb==0.28.1 \
    tqdm \
    sentencepiece \
    lm-eval

echo ">>> Verifying"
python - <<'PY'
import torch, transformers, accelerate, peft, safetensors, datasets
print(f"torch {torch.__version__} (cuda {torch.version.cuda}, available={torch.cuda.is_available()})")
print(f"transformers {transformers.__version__}  accelerate {accelerate.__version__}  peft {peft.__version__}")
print(f"safetensors {safetensors.__version__}  datasets {datasets.__version__}")
PY
echo ">>> Done. Activate with:  conda activate $ENV_NAME"
echo ">>> The eval env is separate:  bash scripts/setup_vllm_env.sh"
