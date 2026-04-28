#!/bin/bash
# ============================================================================
# run_experiments.sh — Run full experiment suite
# ============================================================================
set -e

CONFIG="configs/default.yaml"
ACCEL_CONFIG="accelerate_config.yaml"
NUM_GPUS=2
BITS=4

echo "============================================"
echo "  QLoRA Experiment Suite"
echo "============================================"

# ------------------------------------
# Experiment 1: QLoRA baseline (4-bit)
# ------------------------------------
# echo -e "\n>>> Exp 1: QLoRA ${BITS}-bit baseline"
# accelerate launch \
#     --config_file $ACCEL_CONFIG \
#     --num_processes $NUM_GPUS \
#     scripts/train.py \
#     --config $CONFIG \
#     --qat_mode none \
#     --bits $BITS \
#     --export_dequant \
#     --report_to wandb

# # export merged_only 
# CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
#     --config configs/default.yaml \
#     --qat_mode none \
#     --bits $BITS \
#     --export_only \
#     --checkpoint_dir outputs/qlora-${BITS}bit-none/final \
#     --merge_output_dir outputs/qlora-${BITS}bit-none-merged-eval \
#     --export_merged_only \


# ------------------------------------
# Experiment 2: QLoRA + SQAT (${BITS}-bit)
# ------------------------------------
echo -e "\n>>> Exp 2: QLoRA ${BITS}-bit + SQAT"
accelerate launch \
    --config_file $ACCEL_CONFIG \
    --num_processes $NUM_GPUS \
    scripts/train.py \
    --config $CONFIG \
    --qat_mode sqat \
    --bits $BITS \
    --export_dequant \
    --report_to wandb

# ------------------------------------
# Experiment 3: QLoRA + Full QAT (${BITS}-bit)
# ------------------------------------
# echo -e "\n>>> Exp 3: QLoRA ${BITS}-bit + Full QAT"
# accelerate launch \
#     --config_file $ACCEL_CONFIG \
#     --num_processes $NUM_GPUS \
#     scripts/train.py \
#     --config $CONFIG \
#     --qat_mode full \
#     --export_dequant \
#     --bits $BITS \
#     --report_to wandb

# ------------------------------------
# Experiment 4: QLoRA baseline (3-bit)
# ------------------------------------
# echo -e "\n>>> Exp 4: QLoRA ${BITS}-bit baseline"
# accelerate launch \
#     --config_file $ACCEL_CONFIG \
#     --num_processes $NUM_GPUS \
#     scripts/train.py \
#     --config $CONFIG \
#     --qat_mode none \
#     --bits $BITS \
#     --report_to wandb

# ------------------------------------
# Evaluate all
# ------------------------------------
echo -e "\n>>> Evaluating all models on benchmarks"
for dir in outputs/qlora-${BITS}bit-full-*-eval; do
    if [ -d "$dir" ]; then
        echo "  Evaluating $dir"
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_mmlu.py \
            --model_path "$dir" \
            --num_fewshot 0 \
            --output_dir results/mmlu
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py eval \
            --model_path "$dir" \
            --output_dir results/benchmarks
        # CUDA_VISIBLE_DEVICES=1 python scripts/eval_math.py\
        #     --model_path "$dir" \
        #     --num_fewshot 5 \
        #     --output_dir results/math
    fi
done

python scripts/eval_benchmarks.py compare results/benchmarks/qlora-4bit-*-dequant-eval.json

echo "  All experiments complete!"
echo "============================================"
