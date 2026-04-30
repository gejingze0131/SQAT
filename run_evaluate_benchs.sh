CUDA_VISIBLE_DEVICES=1 python scripts/eval_mmlu.py \
        --model_path outputs/qlora-4bit-sqat-dequant-eval \
        --num_fewshot 0 \
        --output_dir results/mmlu
CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py eval \
        --model_path outputs/qlora-4bit-sqat-dequant-eval  \
        --output_dir results/benchmarks

# python scripts/eval_benchmarks.py compare results/benchmarks/qlora-4bit-sqat-dequant-eval.json