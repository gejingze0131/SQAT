#!/bin/bash
# =============================================================================
# Login-node prefetch for the QWHA baseline. Compute nodes have no route out, so both places
# upstream reaches for wikitext2 -- optimum's GPTQ calibration (train split) and the AdaAlloc
# initialization's X^T X pass (test split) -- must find a BUILT datasets cache, not just a hub
# snapshot: under HF_DATASETS_OFFLINE a bare snapshot still raises OfflineModeIsEnabled.
# =============================================================================
set -eo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/users/nus/jingzege/conda_envs/qwha
export HF_HUB_DISABLE_XET=1

python - <<'PY'
from datasets import load_dataset
for split in ("train", "test"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    print(f"prepared wikitext-2-raw-v1 {split}: {len(ds)} rows")
PY

echo "--- verifying the offline path ---"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from datasets import load_dataset
for split in ("train", "test"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    print(f"offline load OK: {split} {len(ds)} rows")
PY
echo "PREFETCH OK"
