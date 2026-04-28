# ------------------------------------
BITS=4
echo -e "\n>>> Evaluating all models on MMLU 0-shot"
for dir in outputs/qlora-${BITS}bit-full-*-eval; do
    if [ -d "$dir" ]; then
        echo "  Evaluating $dir"
        # CUDA_VISIBLE_DEVICES=1 python scripts/eval_mmlu.py \
        #     --model_path "$dir" \
        #     --num_fewshot 0 \
        #     --output_dir results/mmlu
        # CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py eval \
        #     --model_path "$dir" \
        #     --output_dir results/benchmarks
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_math.py\
            --model_path "$dir" \
            --tasks gsm8k \
            --num_fewshot 5 \
            --output_dir results/math
    fi
done