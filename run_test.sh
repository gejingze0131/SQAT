CUDA_VISIBLE_DEVICES=1 python scripts/export_llama2_int3_dequant.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --output-root outputs/llama2_int3_exports_asym \
  --mode naive \
  --naive-bits 3 \
  --naive-group-size 64 \
  --naive-scheme asymmetric \
  --save-tokenizer