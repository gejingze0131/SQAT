#!/bin/bash
# =============================================================================
# jobs/local/_env.sh — sourced by every script under jobs/local/.
#
# jobs/*.pbs is the cluster half of this repo: a scheduler script per experiment, submitted
# with qsub, each one calling a runs/ entry script. This directory is the same idea for the
# local 3x RTX 6000 Ada (48 GB) box, where there is no scheduler — a job is just a bash script
# you background. Everything a #PBS header used to carry lives here instead:
#
#   conda            the PBS scripts said `source ~/miniforge3/...`; this box has anaconda3,
#                    so conda_bootstrap (runs/lib/common.sh) finds whichever is installed.
#   ngpus=4 -> 3     ACCEL_CONFIG points the runs/ pipelines at accelerate_config_local.yaml,
#                    which declares 3 processes over gpu_ids 0,1,2. Pass --num_gpus 3 too:
#                    the pipelines forward it to `accelerate launch --num_processes`.
#   offline caches   compute nodes had no route out and every job ran HF_HUB_OFFLINE=1; kept
#                    here on purpose, so a run can never silently re-resolve a model revision
#                    mid-experiment. Prefetch into ~/.cache/huggingface first.
#
# Usage from a job script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# and then call a runs/ entry script exactly as the .pbs files do.
# =============================================================================

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../runs/lib/common.sh"
cd_repo_root

conda_bootstrap
set +u; conda activate "${SALTQ_ENV:-saltq}"; set -u

export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local.yaml}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The dataloader forks workers off a process that already built a fast tokenizer; leaving Rust
# parallelism on there prints a deadlock warning per worker per epoch and can actually hang.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# A single-process `accelerate launch` execs `python scripts/train.py` WITHOUT -u (the multi-proc
# path adds it), so through a `| tee` pipe the trainer's stdout is block-buffered: the tqdm bar
# (stderr) advances live while the {'loss': ...} lines sit in a 4 KB buffer and only land at
# process exit. That made a healthy 1-GPU run look like it was logging nothing for hours
# (2026-09-06, the seg32 ablation arm). wandb's own files/output.log captured them all along,
# which is how it was diagnosed. Unbuffer stdout so the file on disk matches what is happening.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# How many cards the training launch gets, and which the vLLM eval uses. They differ, and the
# difference is not cosmetic: vLLM's tensor-parallel size must divide the model's attention-head
# counts. Qwen2.5-32B has 40 heads / 8 KV heads, so TP=3 is REJECTED at engine start and TP=2 is
# the largest that works — while the 65 GB fp16 export does not fit on one 48 GB card, so TP=1
# is out too. Llama-2-7B (32/32 heads) is happy on any of the three.
LOCAL_NUM_GPUS="${LOCAL_NUM_GPUS:-3}"
LOCAL_EVAL_GPUS="${LOCAL_EVAL_GPUS:-0,1}"
LOCAL_EVAL_GPU="${LOCAL_EVAL_GPU:-0}"

echo "============================================================"
echo "  local job: $(basename "${0}")"
echo "  repo:      $(pwd)"
echo "  env:       ${CONDA_DEFAULT_ENV}   accelerate: ${ACCEL_CONFIG}"
echo "  gpus:      train=${LOCAL_NUM_GPUS}  eval=[${LOCAL_EVAL_GPUS}]  export=${LOCAL_EVAL_GPU}"
echo "  started:   $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# The same static preflight every jobs/*.pbs runs before spending a queue slot: it catches a
# runs/ entry whose engine, common.sh path, --config or documented flags have drifted. Instant,
# and the alternative is finding out in stage 3 after an epoch of training.
bash scripts/test_runs_wiring.sh > /dev/null || {
    echo "ERROR: runs/ wiring preflight failed — rerun 'bash scripts/test_runs_wiring.sh' to see it." >&2
    exit 1
}

# CUDA must be USABLE, not merely present. nvidia-smi answering is not evidence: it speaks the
# management plane (/dev/nvidiactl, enumeration, temperatures) while cuInit builds the per-device
# primary context, and a kernel OOM-kill can wedge the second while the first stays perfectly
# healthy. That is not hypothetical here — it happened on 2026-08-31, and `rmmod nvidia_uvm &&
# modprobe nvidia_uvm` (root) was the only fix.
#
# Without this check a resubmission is WORSE than a crash: training dies fast and loudly, but
# scripts/export_gptq_dequant.py picks its device as `cuda if is_available() else cpu` and would
# quietly start a ~1-2 DAY CPU run instead of the ~1 hour GPU one. Nine call sites share that
# pattern. Fail in two seconds instead.
if [ "${LOCAL_SKIP_CUDA_CHECK:-0}" != 1 ]; then
python - <<'PYCUDA' || exit 1
import sys
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    torch.zeros(8, device="cuda").sum().item()          # a real context + a real kernel
except Exception as e:
    import ctypes
    try:
        rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
    except Exception:
        rc = "n/a"
    sys.stderr.write(
        f"ERROR: CUDA is not usable ({type(e).__name__}: {str(e).splitlines()[0][:90]}).\n"
        f"       cuInit(0) -> {rc}  (0 = OK, 3 = NOT_INITIALIZED)\n"
        f"       nvidia-smi answering does NOT mean compute works — it is a different plane.\n"
        f"       If cuInit is 3 after a kernel OOM-kill, the driver needs a reset (needs root):\n"
        f"           sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\n"
        f"       Refusing to start: the export steps would silently fall back to CPU and run\n"
        f"       for days. Set LOCAL_SKIP_CUDA_CHECK=1 to override.\n")
    sys.exit(1)
PYCUDA
fi

