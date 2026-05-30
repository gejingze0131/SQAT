CUDA_VISIBLE_DEVICES=0 python analyze_boundary_salient_channels.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset wikitext \
    --n_samples 512 \
    --seq_len 2048 \
    --boundary_sizes 1 1 29 1 \
    --group_k 128 \
    --agg_mode union_first