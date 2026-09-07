#!/bin/bash
# =============================================================================
# jobs/local/mm_qalora_int3_g64_span_bcal.sh
#
# QA-LoRA control for the MetaMath INT3 g64 cell on Llama-2-7B — the method-level partner of
# jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh, and the local-box port of
# jobs/mm_qalora_int2_span_bcal.pbs one bit up.
#
# WHY IT IS THE RIGHT CONTROL. QA-LoRA's group-pooled adapter folds at deploy into the affine
# zero-points (paper Eq. 7), i.e. it trains a rank-r restricted parameterization of exactly the
# per-(row, group) z that SALT-Q's non-salient segment trains at full rank — and it trains nothing
# at all in the salient columns. So the SALT-Q minus QA-LoRA difference is precisely "full-rank z
# plus a trainable salient tier" against "low-rank z alone", with the same frozen GPTQ INT-b grid
# underneath both. At INT2 g32 on this dataset that difference was +3.94 GSM8K / +2.64 MATH.
#
# EVERYTHING NON-METHOD IS HELD FIXED against the SALT-Q run: same fp16 checkpoint, same 1000
# balanced MetaMath records into the GPTQ base, same loss_span=instruction+response, same 1 epoch,
# same seed 42, and the same effective batch 80 (2 x 8 x 5) — which is why this trains on two of
# the three cards, see the config header and accelerate_config_local2.yaml. Only lr differs from
# SALT-Q's, necessarily: QA-LoRA's trainable object is an unquantized fp16 adapter, so it has no
# per-bit grid-step law to follow and keeps the cell's 5e-3 / r=64 / alpha=16 unchanged from INT2.
#
# EVAL ON TWO CARDS: vLLM's tensor_parallel_size must divide Llama-2-7B's 32 attention heads, so
# TP=2 works and TP=3 is rejected at engine start. Both arms of the pipeline are scored, when
# present: "-3bit-qalora-dequant-eval" (the deployed INT3 model, the headline number) and
# "-3bit-qalora-merged-eval" (fp16 reference; the pipeline does not build it, so it is skipped).
# The whole math test set is scored — GSM8K 1319 + MATH 5000; runs/qalora/_pipeline.sh has no
# --eval_tasks knob, so this is automatic.
#
# COST NOTE: unlike the NF4 baselines the QA-LoRA base is a DENSE fp16 GPTQ INT-b checkpoint —
# ~13 GB resident per rank and ~13 GB written to outputs/ before training starts. With the two
# export dirs, budget ~40 GB of disk.
#
# Run it:
#   mkdir -p logs/metamath
#   setsid nohup bash jobs/local/mm_qalora_int3_g64_span_bcal.sh \
#         > logs/metamath/qalora_mm_int3_g64_ep1_span_bcal.job.log 2>&1 < /dev/null &
#
# Env knobs: PASS=1 trains (+ the automatic dequant export) and stops; PASS=2 re-exports from the
# saved checkpoint and evaluates; default runs the pipeline end to end in one invocation.
# On an OOM: edit the config's per_device_train_batch_size 8 -> 4 /
# gradient_accumulation_steps 5 -> 10 (same effective batch 80, same T) and start over.
# =============================================================================

# Both must be set BEFORE _env.sh, which defaults them to the 3-GPU box configuration.
export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local2.yaml}"
LOCAL_NUM_GPUS="${LOCAL_NUM_GPUS:-2}"
LOCAL_EVAL_GPUS="${LOCAL_EVAL_GPUS:-0,1}"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/qalora_mm_int3_g64_ep1_span_bcal.yaml
OUT=outputs/qalora_mm_int3_ep1_span_bcal    # must equal training.output_dir in the yaml
CKPT="${OUT}-3bit-qalora/final"
LOGDIR=logs/metamath
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

common_args=(
    --config     "$CONFIG"
    --bits       3
    --num_gpus   "$LOCAL_NUM_GPUS"
    --output_dir "$OUT"
    --eval_gpu   "$LOCAL_EVAL_GPU"
    --eval_gpus  "$LOCAL_EVAL_GPUS"
    --note       "MetaMath INT3 g64 span bcal QA-LoRA control, balanced 1000-record calibration, eff batch 80 (2x8x5)"
)

echo "Start time: $(date)"
df -h / | tail -1

if [ "$PASS" = both ]; then
    echo -e "\n>>> GPTQ INT3 base + QA-LoRA training + dequant export + vLLM eval"
    bash runs/qalora/run_qalora_math.sh "${common_args[@]}" "$@" \
        2>&1 | tee -a "$LOGDIR/qalora_mm_int3_g64_ep1_span_bcal.log"
elif [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: GPTQ INT3 base + QA-LoRA training (dequant export runs at the end)"
    bash runs/qalora/run_qalora_math.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/qalora_mm_int3_g64_ep1_span_bcal.train.log"
else
    echo -e "\n>>> PASS 2/2: re-export from $CKPT + vLLM eval"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — training did not finish."; exit 1; }
    bash runs/qalora/run_qalora_math.sh "${common_args[@]}" --checkpoint_dir "$CKPT" "$@" \
        2>&1 | tee -a "$LOGDIR/qalora_mm_int3_g64_ep1_span_bcal.export_eval.log"
fi

echo "End time: $(date)"
df -h / | tail -1
