#!/bin/bash
ENV_PREFIX=/scratch/users/nus/jingzege/conda_envs/qwha
QWHA_DIR=/home/users/nus/jingzege/projects/SQAT/baseline/QWHA
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
export CUDA_HOME="$ENV_PREFIX"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
export PIP_CACHE_DIR=/scratch/users/nus/jingzege/cache/pip
echo "START $(date)"
echo "===== fast-hadamard-transform ====="
pip install --no-build-isolation -v \
    "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.0.4.post1"
echo "fht rc=$?  $(date)"
echo "===== gptqmodel ====="
pip install --no-build-isolation gptqmodel==2.2.0
echo "gptqmodel rc=$?  $(date)"
echo "===== peft fork ====="
pip install -e "$QWHA_DIR/peft"
echo "peft rc=$?"
echo "===== verify ====="
python - <<'PY'
import torch, transformers, peft, gptqmodel, fast_hadamard_transform
from peft import QWHAConfig
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| gptqmodel", gptqmodel.__version__)
print("VERIFY OK")
PY
echo "DONE $(date)"
