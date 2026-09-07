#!/bin/bash
# =============================================================================
# jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh
#
# SALT-Q on Llama-2-7B, MetaMath-395k, INT3 g64, k=128, span supervision, bcal, gptq_latent.
# This is the MetaMath INT3 g64 cell that SALTQ_experiment_tracker_v2.md T10 lists as 未跑,
# and the local-box counterpart of jobs/cs_saltq_mm_int3_sgptql.pbs.
#
# WHAT IT IS COMPARED AGAINST. The same cell one bit lower (T2a, all at effective batch 80,
# same data / span / 1000-record balanced calibration, scored by vLLM greedy generation):
#
#                              GSM8K    MATH
#   QLoRA fp16 upper bound     58.07   10.64
#   SALT-Q  k=256  INT2 g32    56.48   13.68
#   QA-LoRA        INT2 g32    52.54   11.04
#   GPTQ floor     INT2 g32    22.06    2.16
#
# and, from the older lm-eval 5-shot MetaMath table in AGENTS.md §4, the INT3 g64 external
# baselines this cell has that INT2 does not: PSQAT+GPTQ 42.6, LR-QAT / fp16 45.2, SALT-Q 44.8.
# Those are a DIFFERENT harness (lm-eval 5-shot exact-match, not vLLM greedy on the MetaMath test
# split) — do not put them in the same column as the numbers above. The row this run must be read
# against is its own QA-LoRA control, jobs/local/mm_qalora_int3_g64_span_bcal.sh, run on the same
# box at the same effective batch: at INT2 SALT-Q led it by +3.94 GSM8K / +2.64 MATH, and the
# question is whether that survives at a bit width where GPTQ alone already holds the non-salient
# segment together.
#
# TWO CARDS, NOT THREE, and the reason is arithmetic rather than memory: the cell's effective
# batch is 80, 80 is not divisible by 3, and effective batch sets T (optimizer steps per epoch),
# which sets the displacement constant c ~ 0.47*sqrt(T) that the whole lr table in the config was
# derived against. 2 x 8 x 5 = 80 exactly. The third card idles through training and then carries
# the eval. See accelerate_config_local2.yaml.
#
# EVAL RUNS ON TWO CARDS TOO, for a different reason: runs/eval_vllm.sh gives vLLM
# tensor_parallel_size = (number of visible GPUs), and vLLM requires that to divide the model's
# attention-head count. Llama-2-7B has 32 heads, so TP=2 works and TP=3 is rejected at engine
# start. LOCAL_EVAL_GPUS=0,1 below; override it if you want the eval elsewhere.
#
# SCORED ON THE WHOLE math test set (GSM8K 1319 + MATH 5000): NO --eval_tasks flag. The INT2
# sibling passed --eval_tasks gsm8k and had to be re-evaluated later for the MATH half.
#
# DISK: permuted fp16 base ~13 GB + frozen INT3 code base ~7 GB under the output dir, then a
# ~13 GB deploy export plus a hardlinked vLLM copy. ~35 GB.
#
# Run it:
#   mkdir -p logs/metamath
#   setsid nohup bash jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh \
#         > logs/metamath/saltq_mm_int3_g64_ep1_span_bcal_sgptql.job.log 2>&1 < /dev/null &
#
# Env knobs: PASS=1 trains (+ the automatic merge-free export) and stops; PASS=2 re-exports from
# the saved checkpoint and evaluates; default runs the pipeline end to end in one invocation.
# On an OOM: PASS=1 bash <this> --resume_from <ckpt>, optionally after editing the config's
# per_device_train_batch_size 8 -> 4 / gradient_accumulation_steps 5 -> 10 (same 80, same T).
# =============================================================================

# Both must be set BEFORE _env.sh, which defaults them to the 3-GPU box configuration.
export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local2.yaml}"
LOCAL_NUM_GPUS="${LOCAL_NUM_GPUS:-2}"
LOCAL_EVAL_GPUS="${LOCAL_EVAL_GPUS:-0,1}"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/saltq_mm_int3_g64_ep1_span_bcal_sgptql.yaml
OUT=outputs/saltq_mm_int3_g64_ep1_span_bcal_sgptql   # must equal training.output_dir in the yaml
CKPT="${OUT}-3bit-saltq/final"
LOGDIR=logs/metamath
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

common_args=(
    --config      "$CONFIG"
    --bits        3
    --num_gpus    "$LOCAL_NUM_GPUS"
    --output_root "$OUT"
    --eval_gpu    "$LOCAL_EVAL_GPU"
    --eval_gpus   "$LOCAL_EVAL_GPUS"
    --train_layernorms false
    --note        "MetaMath INT3 g64 span bcal gptq_latent, CS-bracketed lrs, balanced 1000-record calibration, eff batch 80 (2x8x5)"
)

echo "Start time: $(date)"
df -h / | tail -1

if [ "$PASS" = both ]; then
    echo -e "\n>>> offline bases + SALT-Q training + merge-free export + vLLM eval"
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql.log"
elif [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: offline bases + SALT-Q training (export runs at the end of it)"
    # "$@" so a resume can be handed through: PASS=1 bash <this> --resume_from <ckpt>
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql.train.log"
else
    echo -e "\n>>> PASS 2/2: re-export from $CKPT + vLLM eval"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — training did not finish."; exit 1; }
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" --checkpoint_dir "$CKPT" "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql.export_eval.log"
fi

echo "End time: $(date)"
df -h / | tail -1
