#!/bin/bash
# ============================================================================
# run_export.sh — Test export functionality of QLoRA models
# ============================================================================
# CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
#     --config configs/default.yaml \
#     --qat_mode full \
#     --bits 4 \
#     --export_only \
#     --checkpoint_dir outputs/qlora-4bit-full/final \
#     --merge_output_dir outputs/qlora-4bit-full/merged \
#     --export_dequant \
#     --report_to wandb

# CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
#     --config configs/default.yaml \
#     --qat_mode sqat \
#     --bits 4 \
#     --export_only \
#     --checkpoint_dir outputs/qlora-4bit-sqat/final \
#     --merge_output_dir outputs/qlora-4bit-sqat/merged \
#     --export_dequant \
#     --report_to wandb

CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --config configs/default.yaml \
    --qat_mode full \
    --bits 4 \
    --export_only \
    --checkpoint_dir outputs/qlora-4bit-full/final \
    --merge_output_dir outputs/qlora-4bit-full-dequant-eval \
    --export_dequant \
    --report_to wandb