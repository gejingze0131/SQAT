# QLoRA + Selective Salient QAT Experiment Framework

## Structure
```
qlora_framework/
├── configs/
│   └── default.yaml          # All experiment configs
├── src/
│   ├── __init__.py
│   ├── model_loader.py       # Model loading (NF4/NF3 + LoRA)
│   ├── data.py               # Dataset loading (CommonsenseQA, etc.)
│   ├── trainer.py            # Training loop with QAT hooks
│   ├── qat_base.py           # QAT interface + Full QAT impl
│   ├── qat_sqat.py           # Selective Salient QAT impl
│   └── export.py             # Weight merge & export for vLLM
├── scripts/
│   ├── train.py              # Main training entry
│   └── eval_mmlu.py          # MMLU 0-shot evaluation via vLLM
├── eval/
│   └── benchmarks.py         # Multi-benchmark harness (BoolQ, etc.)
└── requirements.txt
```

## Quick Start
```bash
# 1. Train QLoRA baseline (4-bit)
accelerate launch --config_file accelerate_config.yaml scripts/train.py --config configs/default.yaml

# 2. Train with SQAT
accelerate launch --num_processes 4 scripts/train.py --config configs/default.yaml --qat_mode sqat

# 3. Export for vLLM
python scripts/train.py --export_only --checkpoint_dir outputs/qlora-4bit

# 4. Evaluate MMLU 0-shot
python scripts/eval_mmlu.py --model_path outputs/qlora-4bit-merged
```
