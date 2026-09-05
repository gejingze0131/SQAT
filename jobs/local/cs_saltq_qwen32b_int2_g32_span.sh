#!/bin/bash
# =============================================================================
# jobs/local/cs_saltq_qwen32b_int2_g32_span.sh
#
# SALT-Q on Qwen2.5-32B, Commonsense-170k, INT2 g32, k=256, bcal, sgptq-latent, zp 2x.
# The scale port of the 7B cell that results_saltq.csv's best INT2 row (74.76) comes from.
#
# ITS CONTROL IS ALREADY MEASURED, by jobs/local/cs_qlora_qwen32b_int2_g32_span.sh:
#
#   fp16 merged upper bound                90.59
#   INT2 g32 GPTQ floor, 3500 balanced     85.20      <- what this run has to beat
#   (the 7B cell, for scale: 77.75 / 66.22, and SALT-Q reached 74.76)
#
# READ THAT GAP BEFORE READING THIS RUN. At 7B, quantization cost 11.53 points and SALT-Q
# recovered 8.54 of them. At 32B it costs 5.39. The room the method exists to occupy is roughly
# half as wide here, so "SALT-Q ~= the floor" is a much weaker result than the same gap would be
# at 7B, and a small negative is not evidence of much at all.
#
# THREE OFFLINE STEPS run first on rank 0 (scripts/train.py), each with its own 32B fix:
#   1. permuted fp16 base   — device_map spreads the 65 GB base over the 3 cards for the
#                             calibration forward (qat.saltq.shard_across_gpus)
#   2. frozen-code base     — GPTQ streams decoder layers and absorbs each module as it lands
#                             (stream_layers / on_quantized), instead of 65 GB resident + 124 GB
#                             of fp32 integer levels
#   3. training             — packed 2-bit codes rebuilt per forward (packed_wq) + 8-bit Adam
#                             moments: ~34 GiB/GPU instead of 92
# All three are placement/storage-only and each has a bit-equivalence test:
#   scripts/test_gptq_stream.py, scripts/test_saltq_packed_wq.py,
#   plus scripts/test_saltq_e2e.py --packed_wq over the whole chain.
# EXCEPT step 1: scripts/test_permute_sharded.py fails ~50% of the time and the failure is
# real — the sharded calibration picks a slightly different salient set in the segment
# device_map splits. This run is still self-consistent (the base is built once and read
# back by every later stage); it just cannot be rebuilt byte-identically. See that file.
#
# DISK: permuted base ~65 GB + frozen-code base ~40 GB, both under the run's output_dir.
#
# Run it:
#   mkdir -p logs/commonsense_170k
#   nohup bash jobs/local/cs_saltq_qwen32b_int2_g32_span.sh \
#         > logs/commonsense_170k/saltq_qwen32b_int2_g32_span.job.log 2>&1 &
#
# Env knobs: PASS=1 trains only, PASS=2 exports+evaluates only (default: both).
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/saltq_qwen32b_cs170k_int2_g32_ep1_span_k256_bcal_sgptql_zp2x.yaml
OUT=outputs/saltq_qwen32b_cs170k_int2_g32_ep1_span_k256_bcal_sgptql_zp2x
CKPT="${OUT}-2bit-saltq/final"
EVAL_DIR="${OUT}-2bit-saltq-deploy-eval"
LOGDIR=logs/commonsense_170k
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

common_args=(
    --config     "$CONFIG"
    --bits       2
    --num_gpus   "$LOCAL_NUM_GPUS"
    --output_root "$OUT"        # runs/saltq calls it --output_root; must equal training.output_dir
    --eval_gpu   "$LOCAL_EVAL_GPU"
    --eval_gpus  "$LOCAL_EVAL_GPUS"
)

if [ "$PASS" = both ] || [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: offline base build + SALT-Q training"
    # "$@" so a resume can be handed through: PASS=1 bash <this> --resume_from <ckpt>
    bash runs/saltq/run_saltq_commonsense.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_qwen32b_cs170k_int2_g32_ep1_span.train.log"
fi

if [ "$PASS" = both ] || [ "$PASS" = 2 ]; then
    echo -e "\n>>> PASS 2/2: merge-free export + vLLM eval"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — pass 1 did not finish."; exit 1; }
    bash runs/saltq/run_saltq_commonsense.sh "${common_args[@]}" --checkpoint_dir "$CKPT" \
        2>&1 | tee "$LOGDIR/saltq_qwen32b_cs170k_int2_g32_ep1_span.export_eval.log" \
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
