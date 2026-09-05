#!/bin/bash
# =============================================================================
# jobs/local/cs_saltq_qwen32b_int3_g64_span.sh
#
# SALT-Q on Qwen2.5-32B, Commonsense-170k, INT3 g64, k=128, bcal, sgptq-latent.
# The scale port of configs/saltq_cs170k_int3_g64_ep1_sal5e5_span_bcal_sgptql.yaml (7B: 78.67).
#
# ITS CONTROL ARMS ARE ALREADY MEASURED, both off the SAME fp16 QLoRA merge and the SAME
# 3500 balanced-record calibration (jobs/local/cs_qlora_qwen32b_int2_g32_span.sh built the
# merge and the INT2 floor; jobs/local/cs_qlora_qwen32b_int3_g64_floor.sh built the INT3 floor):
#
#   fp16 merged upper bound                 90.59
#   INT3 g64 GPTQ floor, 3500 balanced      89.96      <- what this run has to beat
#   INT2 g32 GPTQ floor, same merge         85.20      (SALT-Q reached 89.02 there)
#
# READ THAT GAP FIRST. The cell is 0.63 points wide. At 7B the same INT3 g64 cell is 1.70 wide
# (77.75 / 76.05) and SALT-Q scored 78.67 — ABOVE the fp16 bound, so what that run demonstrated
# was not gap recovery but that training on the frozen grid beats the fp16 finetune. That, not
# "closes X% of the gap", is the claim this run either reproduces at 32B or does not. Anything
# within about +-0.6 of 89.96 is inside the width of the cell.
#
# WHAT IS DIFFERENT FROM THE INT2 32B JOB, mechanically: only --bits and the config. The three
# offline/32B fixes are the same and each has its own bit-equivalence test —
#   1. permuted fp16 base   — device_map spreads the 65 GB base over the 3 cards for the
#                             calibration forward (scripts/test_permute_sharded.py)
#   2. frozen-code base     — GPTQ streams decoder layers and absorbs each module as it lands
#                             (scripts/test_gptq_stream.py)
#   3. training             — packed 3-bit codes rebuilt per forward, 8-bit Adam moments
#                             (scripts/test_saltq_packed_wq.py, scripts/test_saltq_e2e.py)
# NOTE the base dirs are NOT shared with the INT2 run: group_k moves 256 -> 128, so the
# permutation and the salient slice both change, and the frozen codes are 3-bit. Both are built
# fresh under this run's own output_dir.
#
# DISK: permuted base ~62 GB + frozen-code base ~35 GB under the output dir, then a ~62 GB
# deploy export and its unpermuted vLLM copy during PASS 2. 318 GB free at the time of writing.
#
# Run it:
#   mkdir -p logs/commonsense_170k
#   nohup setsid bash jobs/local/cs_saltq_qwen32b_int3_g64_span.sh \
#         > logs/commonsense_170k/saltq_qwen32b_int3_g64_span.job.log 2>&1 &
#
# Env knobs: PASS=1 trains only, PASS=2 exports+evaluates only (default: both).
# On an OOM: PASS=1 bash <this> --resume_from <ckpt>, optionally after editing the config's
# per_device_train_batch_size 3 -> 1 / gradient_accumulation_steps 9 -> 27. Effective batch is 81
# either way, so the lr schedule and T are unchanged by that fallback.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/saltq_qwen32b_cs170k_int3_g64_ep1_span_k128_bcal_sgptql.yaml
OUT=outputs/saltq_qwen32b_cs170k_int3_g64_ep1_span_k128_bcal_sgptql
CKPT="${OUT}-3bit-saltq/final"
EVAL_DIR="${OUT}-3bit-saltq-deploy-eval"
LOGDIR=logs/commonsense_170k
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

common_args=(
    --config     "$CONFIG"
    --bits       3
    --num_gpus   "$LOCAL_NUM_GPUS"
    --output_root "$OUT"        # runs/saltq calls it --output_root; must equal training.output_dir
    --eval_gpu   "$LOCAL_EVAL_GPU"
    --eval_gpus  "$LOCAL_EVAL_GPUS"
)

echo "Start time: $(date)"
df -h / | tail -1

if [ "$PASS" = both ] || [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: offline base build + SALT-Q training"
    # "$@" so a resume can be handed through: PASS=1 bash <this> --resume_from <ckpt>
    bash runs/saltq/run_saltq_commonsense.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_qwen32b_cs170k_int3_g64_ep1_span.train.log"
fi

if [ "$PASS" = both ] || [ "$PASS" = 2 ]; then
    echo -e "\n>>> PASS 2/2: merge-free export + vLLM eval"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — pass 1 did not finish."; exit 1; }
    bash runs/saltq/run_saltq_commonsense.sh "${common_args[@]}" --checkpoint_dir "$CKPT" \
        2>&1 | tee "$LOGDIR/saltq_qwen32b_cs170k_int3_g64_ep1_span.export_eval.log" \
        || echo "export/eval FAILED — the checkpoint at $CKPT is intact, re-run with PASS=2"

    # SALT-Q's export is merge-free, so the eval dir is the only dense artefact. Reclaim it once
    # its score is on disk; the frozen-code base + trainable checkpoint rebuild it.
    if [ "${KEEP_EXPORTS:-0}" != 1 ]; then
        summary="results/commonsense_vllm/$(basename "$EVAL_DIR").json"
        if [ -s "$summary" ]; then
            echo ">>> scored ($summary) — removing $EVAL_DIR and ${EVAL_DIR}-vllm"
            rm -rf "$EVAL_DIR" "${EVAL_DIR}-vllm"
        else
            echo ">>> NO score at $summary — keeping $EVAL_DIR"
        fi
        df -h / | tail -1
    fi
fi

echo "End time: $(date)"
