#!/bin/bash
# =============================================================================
# End-to-end smoke test of the QEFT baseline on a 2-layer random Llama.
#
# Exercises exactly the stages the 7B run goes through, in the same order and through the same
# scripts: the mixed-precision base (balanced in-domain calibration, global weak-column selection,
# OGR fold, GPTQ around the fp16 columns), DDP weak-column tuning on this repo's commonsense
# cell, and the dense export's per-layer equivalence check. ~10 minutes on 2 GPUs; every failure
# it catches otherwise surfaces hours into a queued job.
#
#   bash baseline/QEFT/sqat/smoke_test.sh                    # INT3 g32, k=64
#   BITS=2 GROUP=32 K=64 bash baseline/QEFT/sqat/smoke_test.sh
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
QEFT_DIR="$REPO_ROOT/baseline/QEFT/sqat"
SMOKE_ROOT="${SMOKE_ROOT:-/scratch/users/nus/jingzege/SQAT_outputs/qeft_smoke}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
BITS="${BITS:-3}"; GROUP="${GROUP:-32}"; K="${K:-64}"; NGPU="${NGPU:-2}"
CALIB="${CALIB:-200}"          # balanced calibration records (8 tasks -> 25 each)

cd "$REPO_ROOT"
mkdir -p "$SMOKE_ROOT"

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
                  max_position_embeddings=512, torch_dtype="bfloat16",
                  tie_word_embeddings=False)
torch.manual_seed(0)
LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out)
tok.save_pretrained(out)
print(f"[smoke] wrote {out}")
PY

CFG="$SMOKE_ROOT/smoke.yaml"
python - "$CFG" "$TINY" "$BITS" "$GROUP" "$K" "$SMOKE_ROOT" "$CALIB" <<'PY'
import sys, yaml
cfg_path, tiny, bits, group, k, root, calib = sys.argv[1:8]
bits, group, k, calib = int(bits), int(group), int(k), int(calib)
cfg = {
    "model": {"name": tiny, "quant_bits": bits, "dtype": "bfloat16", "max_seq_len": 512},
    "qeft": {"k": k, "group_size": group, "oproj_weak": True,
             "base_dir": f"{root}/base_int{bits}_g{group}_k{k}",
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]},
    "qat": {"symmetric": False,
            "sqat": {"calibration_samples": calib, "calibration_sampling": "balanced",
                     "calibration_seq_len": 512},
            "gptq": {"nsamples": calib, "percdamp": 0.01, "blocksize": 128, "batch_size": 8}},
    "data": {"train_dataset": "datasets/commonsense", "train_split": "train", "val_split": None,
             "dataset_field": ["instruction", "output"], "sub_task": None,
             "shuffle_dataset": True, "loss_span": "instruction+response", "num_proc": 4,
             "max_train_samples": 200},
    "training": {"output_dir": f"{root}/run", "num_epochs": 1,
                 "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
                 "group_by_length": False, "learning_rate": 5.0e-6, "weight_decay": 0.01,
                 "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "max_grad_norm": 0.3,
                 "fp16": False, "bf16": True, "gradient_checkpointing": True,
                 "logging_steps": 5, "save_steps": 500, "save_total_limit": 1,
                 "dataloader_num_workers": 2, "report_to": "none", "seed": 42},
}
yaml.safe_dump(cfg, open(cfg_path, "w"))
print(f"[smoke] wrote {cfg_path}")
PY

echo "===== unit tests (selection, salient_ids, QEFTLinear, checkpoint I/O) ====="
python "$REPO_ROOT/scripts/test_qeft.py"

echo "===== stage 0: mixed-precision base (OGR + GPTQ around the fp16 weak columns) ====="
python "$QEFT_DIR/build_base.py" --config "$CFG" --force

echo "===== stage 0b: the base build is idempotent ====="
python "$QEFT_DIR/build_base.py" --config "$CFG"

echo "===== stage 1: weak-column tuning ($NGPU ranks) ====="
torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + RANDOM % 1000)) \
    "$QEFT_DIR/train_commonsense.py" --config "$CFG"

echo "===== stage 2: dense export + per-layer equivalence check ====="
RUN_DIR="$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['training']['output_dir'])")"
python "$QEFT_DIR/export_dense.py" --ckpt "$RUN_DIR/final" --out "$SMOKE_ROOT/export"
python "$QEFT_DIR/export_dense.py" --ckpt none \
    --base_dir "$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['qeft']['base_dir'])")" \
    --out "$SMOKE_ROOT/export_base"

echo "===== stage 3: the exported model is a plain Llama, and it differs from the base ====="
python - "$SMOKE_ROOT/export" "$SMOKE_ROOT/export_base" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM
trained, base = (AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
                 for p in sys.argv[1:3])
x = torch.randint(0, 100, (2, 32))
with torch.no_grad():
    a, b = trained(input_ids=x).logits, base(input_ids=x).logits
d = (a - b).abs().max().item()
print(f"[smoke] trained vs untrained base: max |Δlogits| = {d:.3e}")
assert d > 0, "tuning changed nothing — the weak columns are not reaching the export"
n_weak = sum(1 for n, _ in trained.named_parameters() if "weak" in n)
assert n_weak == 0, "the export must be a plain dense Llama, with no QEFT-specific tensors"
print("[smoke] export is a plain dense Llama")
PY

echo
echo "SMOKE TEST OK"
