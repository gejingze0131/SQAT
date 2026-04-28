MANIFEST=outputs/llama2_int3_exports/export_manifest.json

readarray -t MODEL_PATHS < <(python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], "r", encoding="utf-8"))

print(m["baseline_model_id"])

for k in ["naive_nf4_int3_dequant_dir", "gptq_int3_dequant_dir"]:
    v = m.get(k)
    if v:
        print(v)
PY
)

for model_path in "${MODEL_PATHS[@]}"; do
    echo "  Evaluating $model_path"

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_mmlu.py \
        --model_path "$model_path" \
        --num_fewshot 0 \
        --output_dir results/mmlu_llama2_int3

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_benchmarks.py eval \
        --model_path "$model_path" \
        --output_dir results/benchmarks_llama2_int3
done

python scripts/eval_benchmarks.py compare results/benchmarks_llama2_int3/*.json