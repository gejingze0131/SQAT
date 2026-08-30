#!/bin/bash
# =============================================================================
# End-to-end smoke test of the QWHA baseline harness on a 2-layer random Llama.
#
# Exercises exactly the stages the 7B run goes through, in the same order and through the same
# scripts: the balanced-calibration GPTQ base (our grid, packed into GPTQModel's format, with
# the round-trip assertion), the AdaAlloc initialization on those same records, DDP training on
# this repo's commonsense cell, and the dense export's per-layer equivalence check. ~10 minutes
# on 2 GPUs; every failure it catches otherwise surfaces hours into a queued job.
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SQAT_DIR="$REPO_ROOT/baseline/QWHA/sqat"
QWHA_ENV="${QWHA_ENV:-/scratch/users/nus/jingzege/conda_envs/qwha}"
SMOKE_ROOT="${SMOKE_ROOT:-/scratch/users/nus/jingzege/SQAT_outputs/qwha_smoke}"
export QWHA_CACHE_PATH="$SMOKE_ROOT/cache"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
BITS="${BITS:-3}"; GROUP="${GROUP:-64}"; RANK="${RANK:-8}"; NGPU="${NGPU:-2}"
CALIB="${CALIB:-200}"          # balanced calibration records (8 tasks -> 25 each)

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

CFG="$SMOKE_ROOT/smoke.yaml"
python - "$CFG" "$TINY" "$BITS" "$GROUP" "$RANK" "$SMOKE_ROOT" "$CALIB" <<'PY'
import sys, yaml
cfg_path, tiny, bits, group, rank, root, calib = sys.argv[1:8]
bits, group, rank, calib = int(bits), int(group), int(rank), int(calib)
cfg = {
    "model": {"name": tiny, "quant_bits": bits, "max_seq_len": 512},
    "qwha": {"group_size": group, "rank": rank, "scale": 4000.0,
             "gptq_base_dir": f"{root}/bases/tiny_int{bits}_{group}_asym_bcal",
             "init_ckpt_dir": f"{root}/bases/tiny_int{bits}_{group}_asym_bcal-init-rank{rank}",
             "init_calib_batch_size": 8},
    "qat": {"symmetric": False,
            "sqat": {"calibration_samples": calib, "calibration_sampling": "balanced",
                     "calibration_seq_len": 512},
            "gptq": {"nsamples": calib, "percdamp": 0.01, "blocksize": 128, "batch_size": 8}},
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

echo "===== stage 0: balanced-calibration GPTQ base (packed, round-trip asserted) ====="
python "$SQAT_DIR/make_bcal_base.py" --config "$CFG" --out "$SMOKE_ROOT/bases" --tag tiny

echo "===== stage 1: AdaAlloc init on the same balanced records ====="
python "$SQAT_DIR/init_adapter.py" --config "$CFG" --calib balanced

echo "===== stage 1b: the same init off the cached X^T X roots ====="
# The roots are the expensive half of stage 1 (~3 h on the 7B) and depend on the model and the
# calibration set only, so the second width reads them from cache instead of re-earning them.
# A cache that reproduces a DIFFERENT initialization would be worse than no cache at all, so
# the cached path is required to land on the same adapter, bit for bit.
INIT_DIR=$(python - "$CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["qwha"]["init_ckpt_dir"])
PY
)
mv "$INIT_DIR" "$INIT_DIR-fresh"
python "$SQAT_DIR/init_adapter.py" --config "$CFG" --calib balanced
cmp "$INIT_DIR/adapter_model.safetensors" "$INIT_DIR-fresh/adapter_model.safetensors" \
    || { echo "FAIL: cached roots gave a different adapter than the computed ones"; exit 1; }
echo "[smoke] cached-root init is bit-identical to the computed one"
rm -rf "$INIT_DIR-fresh"

echo "===== stage 2: DDP training on this repo's commonsense cell ====="
torchrun --nproc_per_node="$NGPU" --master_port=$((29800 + RANDOM % 100)) \
    "$SQAT_DIR/train_commonsense.py" --config "$CFG"

echo "===== stage 3: dense export + per-layer equivalence check ====="
CUDA_VISIBLE_DEVICES=0 python "$SQAT_DIR/export_dense.py" \
    --adapter_dir "$SMOKE_ROOT/adapter" --out "$SMOKE_ROOT/dense"
CUDA_VISIBLE_DEVICES=0 python "$SQAT_DIR/export_dense.py" \
    --adapter_dir none --config "$CFG" --out "$SMOKE_ROOT/dense_base"

echo "SMOKE TEST OK"
