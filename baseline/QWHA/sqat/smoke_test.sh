#!/bin/bash
# =============================================================================
# End-to-end smoke test of the QWHA baseline harness on a 2-layer random Llama.
#
# Exercises every seam that costs GPU-hours to discover at 7B: gptqmodel quantization at the
# cell's bit width, the peft-fork adapter build, the AdaAlloc initialization, DDP training on
# this repo's commonsense cell, and the dense export's equivalence check. ~10 minutes on 2 GPUs.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SQAT_DIR="$REPO_ROOT/baseline/QWHA/sqat"
QWHA_ENV="${QWHA_ENV:-/scratch/users/nus/jingzege/conda_envs/qwha}"
SMOKE_ROOT="${SMOKE_ROOT:-/scratch/users/nus/jingzege/SQAT_outputs/qwha_smoke}"
export QWHA_CACHE_PATH="$SMOKE_ROOT/cache"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
BITS="${BITS:-3}"; GROUP="${GROUP:-64}"; RANK="${RANK:-8}"; NGPU="${NGPU:-2}"

cd "$REPO_ROOT"
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$QWHA_ENV"; set -u

TINY="$SMOKE_ROOT/tiny-llama"
python - "$TINY" <<'PY'
import os, sys, torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
out = sys.argv[1]
if os.path.isdir(out):
    print(f"[smoke] {out} exists"); sys.exit(0)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
cfg = LlamaConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                  num_attention_heads=8, num_key_value_heads=8, vocab_size=len(tok),
                  max_position_embeddings=512, torch_dtype="float16")
torch.manual_seed(0)
model = LlamaForCausalLM(cfg).to(torch.float16)
model.save_pretrained(out); tok.save_pretrained(out)
print(f"[smoke] wrote {out}")
PY

python "$SQAT_DIR/quantize_base.py"  -m "$TINY" -b "$BITS" -g "$GROUP"
python "$SQAT_DIR/init_adapter.py"   -m "$TINY" -b "$BITS" -g "$GROUP" -r "$RANK"

CFG="$SMOKE_ROOT/smoke.yaml"
python - "$CFG" "$TINY" "$BITS" "$GROUP" "$RANK" "$SMOKE_ROOT" <<'PY'
import sys, yaml
cfg_path, tiny, bits, group, rank, root = sys.argv[1:7]
cfg = {
    "model": {"name": tiny, "quant_bits": int(bits), "max_seq_len": 512},
    "qwha": {"group_size": int(group), "rank": int(rank), "scale": 4000.0},
    "data": {"train_dataset": "datasets/commonsense", "train_split": "train", "val_split": None,
             "dataset_field": ["instruction", "output"], "sub_task": None,
             "shuffle_dataset": True, "loss_span": "instruction+response", "num_proc": 4,
             "max_train_samples": 200},
    "training": {"output_dir": f"{root}/adapter", "num_epochs": 1,
                 "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
                 "group_by_length": False, "learning_rate": 3.0e-5, "weight_decay": 0.01,
                 "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "max_grad_norm": 0.3,
                 "fp16": False, "bf16": True, "gradient_checkpointing": True,
                 "logging_steps": 5, "save_steps": 500, "save_total_limit": 1,
                 "dataloader_num_workers": 2, "report_to": "none", "seed": 42},
}
yaml.safe_dump(cfg, open(cfg_path, "w"))
print(f"[smoke] wrote {cfg_path}")
PY

torchrun --nproc_per_node="$NGPU" --master_port=$((29800 + RANDOM % 100)) \
    "$SQAT_DIR/train_commonsense.py" --config "$CFG"

CUDA_VISIBLE_DEVICES=0 python "$SQAT_DIR/export_dense.py" \
    --adapter_dir "$SMOKE_ROOT/adapter" --out "$SMOKE_ROOT/dense"

echo "SMOKE TEST OK"
