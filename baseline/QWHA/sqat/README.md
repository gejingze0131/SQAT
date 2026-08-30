# QWHA baseline — environment and how to run it

Reproduces **QWHA** (*Quantization-Aware Walsh-Hadamard Adaptation*,
[arXiv:2509.17428](https://arxiv.org/abs/2509.17428), upstream
[vantaa89/QWHA](https://github.com/vantaa89/QWHA) @ `fc8d288`) on Llama-2-7B / Commonsense-170k,
inside this repo's experimental cell. What is upstream and what is ours — and why each departure
exists — is in [PROVENANCE.md](PROVENANCE.md). This file is only about **building the environment
and running the pipeline**.

QWHA needs its own environment: upstream pins torch 2.5.1 / transformers 4.51.3 / gptqmodel 2.2.0
and ships a **fork of peft** (the `QWHAConfig` / `QWHALayer` tuner lives inside it). The repo's
training env (`saltq`: torch 2.11, transformers 5.14, peft 0.20) can hold exactly one peft, and
every checkpoint under `outputs/` was produced with that one — so QWHA gets a third env, the same
way `baseline/LoTA-QAF/` does. Only the training loop lives there; the quantization grid, the data
recipe and the evaluation are shared with every other row (see PROVENANCE.md).

---

## 1. One command

```bash
bash baseline/QWHA/sqat/setup_env.sh          # ~30 min, mostly the fast-hadamard-transform build
```

Overridable, so it can be rebuilt anywhere:

```bash
ENV_PREFIX=/path/to/envs/qwha \
TORCH_CUDA_ARCH_LIST=8.0 \      # 8.0 A100 · 8.6 A6000/3090 · 8.9 L40S/4090 · 9.0 H100
MAX_JOBS=16 \
bash baseline/QWHA/sqat/setup_env.sh
```

It ends by importing torch / transformers / peft / gptqmodel / fast_hadamard_transform and
printing `ENV BUILD OK`. Anything else means it did not finish.

**Prerequisites**

| | |
|---|---|
| conda / mamba | any recent one (`miniforge3` here) |
| NVIDIA driver | must support the CUDA 12.4 runtime — ≥ 550 recommended, ≥ 525 works via CUDA 12.x minor-version compatibility (this cluster: 570.124.06) |
| system CUDA | **not needed** — nvcc comes from conda |
| system gcc | **not needed** — gcc 12 comes from conda (see §3) |
| disk | ~13 GB for the env |
| GPU for the build | not needed; the only CUDA code is cross-compiled from `TORCH_CUDA_ARCH_LIST` |

## 2. What the environment contains

| package | version | why pinned |
|---|---|---|
| python | 3.12 | upstream `requires-python >= 3.12` |
| torch | 2.5.1+cu124 | upstream pin |
| transformers | 4.51.3 | upstream pin; gptqmodel 2.2 patches transformers internals and does not survive 5.x |
| peft | 0.14.0.dev0 | **the vendored fork** at `baseline/QWHA/peft` (`pip install -e`), which is where QWHA itself is implemented |
| gptqmodel | 2.2.0 | the QuantLinear the adapter wraps; installed **without** CUDA kernels (§3) |
| fast-hadamard-transform | v1.0.4.post1 | the WHT kernel, compiled from source |
| accelerate / datasets / optimum / lm-eval / … | upstream `pyproject.toml` pins | |

An exact pin list is in [`env/requirements-frozen.txt`](env/requirements-frozen.txt)
(`pip freeze` of the working env).

## 3. The manual equivalent, with the four things that bite

If you would rather run it by hand — or need to adapt it to another cluster — this is exactly what
`setup_env.sh` does, in order.

```bash
ENV=/path/to/envs/qwha
conda create -y -p "$ENV" python=3.12
conda activate "$ENV"

# (a) nvcc + CUDA headers, matching the torch wheel's cu124
conda install -y -p "$ENV" -c nvidia/label/cuda-12.4.0 cuda-toolkit

# (b) a modern host compiler — see gotcha 1
conda install -y -p "$ENV" -c conda-forge gxx_linux-64=12 gcc_linux-64=12

export CUDA_HOME="$ENV"
export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export TORCH_CUDA_ARCH_LIST=8.0        # your GPU's compute capability
export MAX_JOBS=16

pip install --upgrade pip setuptools wheel packaging ninja
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# (c) upstream's pinned deps (its pyproject minus openai, which is unused and pins openai<1)
pip install accelerate==1.6.0 click==8.1.8 datasets==3.4.1 device-smi==0.4.1 fire==0.7.0 \
    huggingface-hub==0.31.1 joblib==1.4.2 lm-eval==0.4.8 logbar==0.0.4 matplotlib==3.10.0 \
    numpy==2.2.2 optimum==1.24.0 pandas==2.2.3 pillow==11.1.0 pyarrow==19.0.1 pyyaml==6.0.2 \
    regex==2024.11.6 requests==2.32.3 safetensors==0.5.3 scikit-learn==1.6.1 scipy==1.15.2 \
    sentencepiece==0.2.0 tabulate==0.9.0 tokenicer==0.0.4 tokenizers==0.21.1 tqdm==4.67.1 \
    transformers==4.51.3 "triton>=3.1.0" wandb==0.19.9 threadpoolctl

# (d) the WHT kernel (~1 h of nvcc; the one genuinely slow step)
pip install --no-build-isolation \
    "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.0.4.post1"

# (e) gptqmodel WITHOUT its CUDA kernels — see gotcha 3
env -u TORCH_CUDA_ARCH_LIST CUDA_VISIBLE_DEVICES= \
    pip install --no-build-isolation --no-deps gptqmodel==2.2.0

# (f) the vendored peft fork — --no-deps so its transformers range cannot move the 4.51.3 pin
pip install --no-build-isolation --no-deps -e baseline/QWHA/peft
```

**Gotcha 1 — the host compiler.** torch ≥ 2.5's headers refuse GCC < 9 (`"You're trying to build
PyTorch with a too old version of GCC"`), and RHEL 8 ships 8.5. Install gcc/gxx 12 from conda-forge
and point **both** `CXX` and nvcc (`-ccbin`) at it, or the fast-hadamard build dies after a few
minutes of compiling.

**Gotcha 2 — `set -u` and conda's compiler hooks.** conda-forge's compiler packages install
activate/deactivate hooks that read `CONDA_BACKUP_CXX` unguarded, and conda re-runs them after
every `conda install` in that env. An installer script using `set -u` will abort there
(`CONDA_BACKUP_CXX: unbound variable`). `setup_env.sh` therefore uses `set -eo pipefail`, not
`-euo`.

**Gotcha 3 — gptqmodel's CUDA extensions.** Building them is slow and fragile, and QWHA never
needs them: it only calls `QuantLinear.dequantize_weight()`, which the pure-torch kernel
implements for 2/3/4/8 bits (the kernel auto-selection logs `TorchQuantLinear`). `setup.py` skips
the extensions only when `TORCH_CUDA_ARCH_LIST` is unset **and** torch sees no device — hence
`env -u TORCH_CUDA_ARCH_LIST CUDA_VISIBLE_DEVICES=`. Pass `BUILD_GPTQMODEL_CUDA=1` to
`setup_env.sh` if you want them anyway.

**Gotcha 4 — long installs dying silently.** On a login node under an agent/session manager, a
`nohup`'d (even `setsid`'d) pip can be killed when the session exits, leaving a log that just
stops mid-install with no error. Run the build in a terminal you keep open, under `tmux`/`screen`,
or as a batch job.

## 4. Exact-pin reproduction

```bash
conda create -y -p "$ENV" python=3.12 && conda activate "$ENV"
conda install -y -p "$ENV" -c nvidia/label/cuda-12.4.0 cuda-toolkit
conda install -y -p "$ENV" -c conda-forge gxx_linux-64=12 gcc_linux-64=12
export CUDA_HOME="$ENV" CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" \
       CXX="$ENV/bin/x86_64-conda-linux-gnu-g++" NVCC_PREPEND_FLAGS="-ccbin $CXX" \
       TORCH_CUDA_ARCH_LIST=8.0
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation -r baseline/QWHA/sqat/env/requirements-frozen.txt
pip install --no-build-isolation --no-deps -e baseline/QWHA/peft
```

The frozen list carries `fast_hadamard_transform` as a git URL, so it still compiles (§3 d), and
`gptqmodel==2.2.0+cu124torch2.5`, which is the local version tag the pure-python build stamps —
drop the `+cu124torch2.5` suffix if pip refuses it on your machine.

## 5. Verify

```bash
conda activate "$ENV"
python -c "
import torch, transformers, peft, gptqmodel, fast_hadamard_transform
from peft import QWHAConfig
print(torch.__version__, transformers.__version__, peft.__version__, gptqmodel.__version__)"
```

Expected: `2.5.1+cu124 4.51.3 0.14.0 2.2.0`.

Then the real acceptance test — the whole pipeline on a 2-layer random Llama (~10 min, 2 GPUs):

```bash
bash baseline/QWHA/sqat/smoke_test.sh                 # or: qsub jobs/qwha_smoke.pbs
BITS=2 GROUP=32 SMOKE_ROOT=/tmp/qwha_smoke_int2 bash baseline/QWHA/sqat/smoke_test.sh
```

It runs every stage the 7B run does — balanced-calibration GPTQ base (packed into GPTQModel's
format, round trip asserted), AdaAlloc initialization, DDP training on this repo's commonsense
cell, dense export with a per-layer equivalence check — and prints `SMOKE TEST OK`.

## 6. Data, and offline compute nodes

```bash
bash baseline/QWHA/sqat/prefetch_data.sh              # login node, once
```

Downloads and **builds** the wikitext-2 datasets cache (used by upstream's calibration path) and
verifies it loads under `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` — a bare hub snapshot is not
enough for `load_dataset`, which raises `OfflineModeIsEnabled` without a prepared cache. The
reported rows calibrate on `datasets/commonsense/train.json`, which is already local, and the base
model comes from `HF_HOME`; prefetch that too if your nodes have no route out.

## 7. Run it

```bash
# 1 GPU: the balanced-calibration GPTQ base + the AdaAlloc initialization (shared per width)
qsub jobs/qwha_prep_int3_g64_bcal.pbs

# 4 GPUs: train -> dense export -> vLLM eval -> results_saltq.csv, all into one log
qsub -W depend=afterok:<prep-id> jobs/cs_qwha_int3_g64_span_bcal.pbs
```

or directly:

```bash
bash runs/qwha/run_qwha_commonsense.sh --bits 3 --group_size 64 --with_base
bash runs/qwha/run_qwha_commonsense.sh --bits 2 --group_size 32 --with_base \
    --config baseline/QWHA/sqat/configs/qwha_cs170k_int2_g32_ep1_span_bcal.yaml
```

Every stage is idempotent — it skips when its output exists — so a job that hits the walltime can
be resubmitted. Artefacts: `outputs/qwha_bases/` (base + initialized adapter),
`outputs/qwha_cs170k_*` (trained adapter), `outputs/qwha_cs170k_*-dense-eval` (what vLLM reads),
scores in `results/commonsense_vllm/` and `results_saltq.csv` (`--filter qwha`), logs in
`logs/commonsense_170k/`.

**Memory** (A100-40GB): training needs gradient checkpointing on — the QWHA forward materializes
one dense `[out, in]` weight per projection, and without recompute they all stay alive for the
backward. It is on by default in the configs. The AdaAlloc initialization is the CPU-RAM-hungry
stage (~110 GB peak: fp32 base + per-layer quantization errors + float64 XᵀX buffers).
