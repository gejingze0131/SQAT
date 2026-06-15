CUDA_VISIBLE_DEVICES=1 python analyze_boundary_salient_channels.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset metamath \
    --n_samples 512 \
    --seq_len 2048 \
    --outlier_log_sigma 3.0 \
    --group_k_candidates 64 128 256 \
    --max_segments 4 \
    --output_dir salient_analysis_out