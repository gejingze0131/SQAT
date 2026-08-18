#!/bin/bash
# =============================================================================
# setup_vllm_env.sh — build the conda env that runs/eval_vllm.sh generates in.
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
#   bash scripts/setup_vllm_env.sh              # create env `vllm-eval`, python 3.12
#   bash scripts/setup_vllm_env.sh myenv 3.12   # or name it yourself
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-vllm-eval}"
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

# PHASE 1 — the CUDA-matched runtime, from the cu128 index EXCLUSIVELY (--index-url, not
# --extra-index-url). Both indexes carry this torch version and pip's choice between them is
# arbitrary; getting the PyPI one means a cu13 build and "driver too old" at engine start.
# The versions are the ones the pinned vllm requires — move them together.
# Installing the trio first also stops the vLLM resolve in phase 2 from pulling its own.
echo ">>> Phase 1/2: torch runtime built for CUDA 12.8"
pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# PHASE 2 — vLLM itself. llguidance is built from source here (see CC=gcc above).
echo ">>> Phase 2/2: vLLM (llguidance builds from source; this takes a while)"
pip install --no-cache-dir -r requirements-vllm.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Import the chain that actually breaks when any of the above is mismatched:
#   * sqlite3    — vLLM's engine imports it, which drags in the env's libicui18n and its
#                  libstdc++ requirement (see the LD_LIBRARY_PATH note in runs/eval_vllm.sh)
#   * torch      — must report a 12.x build, or the driver rejects it at engine start
#   * vllm._C    — the COMPILED extension. `import vllm` succeeds on a wheel built against
#                  the wrong CUDA and only the extension fails, with
#                  "libcudart.so.13: cannot open shared object file", so it has to be
#                  imported explicitly here rather than trusted to a package-level import.
echo ">>> Verifying"
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python - <<'PYCHECK'
import sqlite3, torch, torchaudio, transformers, vllm
from vllm.v1.engine.core_client import EngineCoreClient
import vllm._C  # the COMPILED extension — this is what fails on a CUDA-13 wheel
assert torch.version.cuda.startswith("12."), f"torch is built for CUDA {torch.version.cuda}, not 12.x"
assert transformers.__version__ < "5", (
    f"transformers {transformers.__version__}: vLLM 0.11 reads tokenizer attributes the 5.x "
    f"API dropped, and only finds out mid-generation")
print(f"  vllm {vllm.__version__} | torch {torch.__version__} ({torch.version.cuda}) "
      f"| transformers {transformers.__version__}")
PYCHECK

echo
echo "Done. Evaluate with:"
echo "  bash runs/eval_vllm.sh --model_path <exported dir> --dataset commonsense"
