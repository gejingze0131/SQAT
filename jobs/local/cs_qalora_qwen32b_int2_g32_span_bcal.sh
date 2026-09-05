#!/bin/bash
# =============================================================================
# jobs/local/cs_qalora_qwen32b_int2_g32_span_bcal.sh
#
# QA-LoRA baseline for the Qwen2.5-32B INT2 g32 bcal cell — the method-level control for SALT-Q.
# Scale port of jobs/cs_qalora_int2_span_bcal.pbs (7B: 72.96, vs SALT-Q 74.76, floor 66.22).
#
# THE CELL SO FAR, on Qwen2.5-32B (all at effective batch 81, same data/span/calibration):
#   fp16 upper bound                90.59
#   INT2 g32 GPTQ floor (bcal)      85.20
#   SALT-Q                          89.02
#   QA-LoRA                         <- this run
# At 7B the two methods were 1.80 points apart, which is the number this run is here to place.
#
# WHAT MADE IT RUNNABLE AT 32B. QA-LoRA trains on the real GPTQ INT-b grid, shipped by
# build_qalora_intb_base as a DENSE fp16 checkpoint: 61 GiB, per DDP rank, against 44.4 GiB
# usable. Unlike SALT-Q there was no low-bit path in src/qalora.py at all, so QALoRAPackedLinear
# was added — packed codes (7.3 GiB at INT2) rebuilt per forward through the SAME group_dequantize
# that produced the dense file, held bit-identical by scripts/test_qalora_packed.py. The offline
# base build also streams decoder layers now (stream_layers + on_quantized), so the 61 GiB fp16
# model is never resident on one card. Budget: ~20.3 GiB per GPU.
#
# TWO PASSES, same reasoning as the other 32B jobs: an export/eval failure must not cost training.
#
# Run it:
#   mkdir -p logs/commonsense_170k
#   setsid nohup bash jobs/local/cs_qalora_qwen32b_int2_g32_span_bcal.sh \
#         > logs/commonsense_170k/qalora_qwen32b_int2_g32_span_bcal.job.log 2>&1 < /dev/null &
#
# Env: PASS=1 trains only, PASS=2 exports+evaluates only (default: both).
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/qalora_qwen32b_cs170k_int2_g32_ep1_span_bcal.yaml
OUT=outputs/qalora_qwen32b_cs170k_int2_ep1_span_bcal
CKPT="${OUT}-2bit-qalora/final"
LOGDIR=logs/commonsense_170k
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

common_args=(
    --config     "$CONFIG"
    --bits       2
    --num_gpus   "$LOCAL_NUM_GPUS"
    --output_dir "$OUT"
    --eval_gpu   "$LOCAL_EVAL_GPU"
    --eval_gpus  "$LOCAL_EVAL_GPUS"
)

if [ "$PASS" = both ] || [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: GPTQ INT-b base + QA-LoRA training"
    bash runs/qalora/run_qalora_commonsense.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/qalora_qwen32b_cs170k_int2_g32_ep1_span_bcal.train.log"
fi

if [ "$PASS" = both ] || [ "$PASS" = 2 ]; then
    echo -e "\n>>> PASS 2/2: export + vLLM eval"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — pass 1 did not finish."; exit 1; }
    bash runs/qalora/run_qalora_commonsense.sh "${common_args[@]}" --checkpoint_dir "$CKPT" \
        2>&1 | tee -a "$LOGDIR/qalora_qwen32b_cs170k_int2_g32_ep1_span_bcal.export_eval.log" \
        || echo "export/eval FAILED — the checkpoint at $CKPT is intact, re-run with PASS=2"
fi

echo "End time: $(date)"
