#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/users/nus/jingzege/conda_envs/qwha
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd /home/users/nus/jingzege/projects/SQAT
python - <<'PY'
import sys
sys.path.insert(0, "baseline/QWHA/sqat")
from datasets import load_dataset
print("wikitext2 test rows:", len(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")), flush=True)
print("wikitext2 train rows:", len(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")), flush=True)
import qwha_common as qc
print("gptq dir:", qc.gptq_base_dir("meta-llama/Llama-2-7b-hf", 3, 64), flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=2048, padding_side="right")
tok.pad_token = tok.eos_token
cfg = {"data": {"train_dataset": "datasets/commonsense", "train_split": "train", "val_split": None,
                "dataset_field": ["instruction","output"], "sub_task": None, "shuffle_dataset": True,
                "loss_span": "instruction+response", "num_proc": 8, "max_train_samples": 2000},
       "training": {"seed": 42}}
tr, ev, coll = qc.load_sqat_data_module(cfg, tok)
print("train records:", len(tr), flush=True)
b = coll([tr[0], tr[1]])
print("batch:", {k: tuple(v.shape) for k, v in b.items()}, flush=True)
print("PRECHECK OK")
PY
