#!/bin/bash
# =============================================================================
# jobs/local/cs_qlora_qwen32b_int3_g64_floor.sh
#
# ONE EXTRA ARM for the Qwen2.5-32B commonsense cell: an INT3 g64 GPTQ floor,
# alongside the INT2 g32 floor (85.20) that jobs/local/cs_qlora_qwen32b_int2_g32_span.sh
# already scored. Nothing is trained here.
#
# WHY IT NEEDS ITS OWN SCRIPT. The floor is not "GPTQ over the base model" — step (2) of the
# INT2 job quantizes $MERGED, the fp16 QLoRA MERGE, so the floor and the upper bound differ in
# exactly one thing: whether the finetuned weights were quantized afterwards. An INT3 g64 number
# is only comparable to 85.20 if it comes off the same input, so this script rebuilds that merge
# from the surviving adapter rather than quantizing the stock base.
#
# WHAT SURVIVED. The 65 GB merged export was reclaimed once its score was written (the INT2 job's
# cleanup loop), but outputs/..-2bit-none/final/adapter_model.safetensors (2.1 GB) is intact and
# the merge is deterministic, so the input reconstructs exactly.
#
# CALIBRATION is held identical to the INT2 floor and to what SALT-Q calibrates its frozen codes
# on: 3500 balanced-sampled commonsense records (7B calib variant D). Only --bits and
# --group_size move. g64 at 3 bits is 3+16/64 = 3.25 bits/weight against INT2 g32's 2+16/32 =
# 2.5, so this arm is ABOVE the cell, not a tighter bound on it.
#
# DISK. 65 GB (merge) + 62 GB (INT3 dequant) = 127 GB live, against 328 GB free at the time of
# writing. Both are reclaimed once their scores are on disk; KEEP_EXPORTS=1 opts out.
#
# Run it:
#   nohup setsid bash jobs/local/cs_qlora_qwen32b_int3_g64_floor.sh \
#         > logs/commonsense_170k/qlora_qwen32b_int3_g64_floor.job.log 2>&1 &
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/qlora_none_cs170k_int2_g32_ep1_span_qwen32b.yaml
OUT=outputs/qlora_none_qwen32b_cs170k_int2_ep1_span
CKPT="${OUT}-2bit-none/final"
MERGED="${OUT}-2bit-none-merged-eval"
FLOOR="${OUT}-3bit-g64-none-gptq-eval"
LOGDIR=logs/commonsense_170k
mkdir -p "$LOGDIR"

[ -d "$CKPT" ] || { echo "ERROR: no QLoRA adapter at $CKPT."; exit 1; }
echo "Start time: $(date)"
df -h / | tail -1

# --- (1/3) rebuild the fp16 merge that the floor quantizes --------------------------------
# Same invocation as step (1) of the INT2 job, --bits 2 included: the merge writes dense fp16 and
# never reads the bit width, so keeping the flag identical keeps the input bit-identical too.
if [ -d "$MERGED" ]; then
    echo ">>> (1/3) merged fp16 already at $MERGED"
else
    echo -e "\n>>> (1/3) rebuilding the fp16 merge from the adapter"
    CUDA_VISIBLE_DEVICES="$LOCAL_EVAL_GPU" python scripts/train.py \
        --config "$CONFIG" --qat_mode none --bits 2 \
        --export_only --export_merged_only \
        --checkpoint_dir "$CKPT" --merge_output_dir "$MERGED" \
        2>&1 | tee "$LOGDIR/qlora_qwen32b_int3_g64_floor.merged_export.log"
fi

# --- (2/3) INT3 g64 GPTQ over that merge, bcal 3500 balanced ------------------------------
if [ -d "$FLOOR" ]; then
    echo ">>> (2/3) GPTQ floor already at $FLOOR"
elif [ -d "$MERGED" ]; then
    echo -e "\n>>> (2/3) INT3 g64 GPTQ floor (bcal: 3500 balanced)"
    CUDA_VISIBLE_DEVICES="$LOCAL_EVAL_GPU" python scripts/export_gptq_dequant.py \
        --model_path "$MERGED" --output_dir "$FLOOR" --config "$CONFIG" \
        --bits 3 --group_size 64 \
        --nsamples 3500 --calibration_samples 3500 --calibration_sampling balanced \
        --batch_size 8 \
        2>&1 | tee "$LOGDIR/qlora_qwen32b_int3_g64_floor.gptq.log" || \
        echo "GPTQ floor FAILED."
else
    echo "ERROR: $MERGED missing — cannot build the floor."
fi

# --- (3/3) score it -------------------------------------------------------------------------
# Non-fatal and idempotent, for the reasons in the INT2 job's header: a 22k-prompt generative eval
# of a 32B model is hours, and a re-run must not redo an arm that is already scored.
echo -e "\n>>> (3/3) scoring the INT3 g64 floor"
summary="results/commonsense_vllm/$(basename "$FLOOR").json"
if [ ! -d "$FLOOR" ]; then
    echo "    (skip eval) $FLOOR not found"
elif [ -s "$summary" ]; then
    echo "    (skip eval) already scored: $summary"
else
    bash runs/eval_vllm.sh --model_path "$FLOOR" --dataset commonsense \
        --gpus "$LOCAL_EVAL_GPUS" --tag "$(basename "$FLOOR")" \
        || echo "    eval FAILED for $FLOOR"
fi

# --- reclaim both dense exports once the score is written -----------------------------------
if [ "${KEEP_EXPORTS:-0}" != 1 ]; then
    echo -e "\n>>> Reclaiming dense exports"
    if [ -s "$summary" ]; then
        for d in "$FLOOR" "$MERGED"; do
            echo "    removing $d and ${d}-vllm"
            rm -rf "$d" "${d}-vllm"
        done
    else
        echo "    NO score at $summary — keeping $FLOOR and $MERGED for a retry"
    fi
    df -h / | tail -1
fi

echo "End time: $(date)"
