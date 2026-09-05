#!/bin/bash
# =============================================================================
# jobs/local/cs_qlora_qwen32b_int2_g32_span.sh
#
# QLoRA CONTROL for the Qwen2.5-32B INT2 g32 span cell, on the local 3x48 GB box.
# Local counterpart of jobs/cs_qlora_int2_span_bounds.pbs, which ran the same cell on
# Llama-2-7B over four cluster cards.
#
# WHY THIS RUNS BEFORE ANY SALT-Q RUN ON QWEN. Every row in results_saltq.csv is Llama-2-7B.
# Porting to a 4.6x larger model changes four things at once that a QAT run cannot tell apart
# from a method failure: a different tokenizer, GQA (40 heads / 8 KV), a 152k-vocab lm_head,
# and DDP over 537M LoRA parameters instead of 160M. Plain QLoRA exercises all four and none
# of SALT-Q's machinery, so whatever it reports is about the PIPELINE. The 7B cell established
# the shape of the answer to expect: merged-fp16 high, INT2-dequant far below it, and the gap
# is what SALT-Q is for.
#
# Two exported variants, both scored:
#   ...-2bit-none-merged-eval    fp16 LoRA merge, no quantization   -> upper bound
#   ...-2bit-none-dequant-eval   INT2 g32 quantize->dequantize      -> floor
#
# BATCH SHAPE. 3 per-device x 9 accumulation x 3 GPUs = 81, against the 7B cell's 5 x 4 x 4 = 80,
# so the lr schedule carries over unchanged (T=1822 vs 1845). Gradient checkpointing is on for
# every NF4 base already (src/model_loader.py). Measured: 34.4 GB of 46 per card, ~21 s per
# optimizer step, ~10.5 h for the epoch. See the config header for the rest.
#
# TWO PASSES, and the split is deliberate. The config sets export.merge_and_save=false, so pass
# Every export step is non-fatal on purpose: _env.sh runs under `set -e` with pipefail, and the
# first time the floor was OOM-killed it took steps (3)-(5) with it, so the merged upper bound
# that HAD exported cleanly never got scored. A step that dies must cost only itself.
#
# 1 trains and nothing more (--skip_export drops Stage 1b too) — a 32B in-process export would
# run while the training model, its Adam state and the DDP buckets still hold 34 GB on every
# card, and each dense export is 65 GB on disk. Pass 2 does both exports in fresh single-GPU
# processes off the saved final/ checkpoint, then evaluates. Either pass can be re-run on its
# own: pass 2 skips an export whose output dir already exists.
#
# DISK. Three dense exports at 65 GB each = 195 GB live at the peak. The hook-free copies
# runs/eval_vllm.sh makes are hardlinks (scripts/export_vllm_ready.py), not the 65 GB
# duplicates they used to be, so they cost nothing until something rewrites a file.
#
# THREE ARMS, not two, because the middle one is not a deployment number. The pipeline's
# `-dequant-eval` is plain RTN: src/export.py drops qat.sqat_permute.gptq for every qat_mode that
# is not sqat_permute (it prints a warning saying so), so no Hessian and no calibration data enter
# it. The floor that is actually comparable to SALT-Q is a separate GPTQ pass over the SAME 3500
# balanced-sampled commonsense records SALT-Q calibrates its frozen codes on — 7B calib variant D,
# jobs/cs_qlora_int2_span_floor_calibD.pbs. So:
#
#   ...-2bit-none-merged-eval    fp16 LoRA merge, no quantization      -> upper bound
#   ...-2bit-none-gptq-eval      INT2 g32 GPTQ, 3500 balanced (bcal)   -> THE FLOOR
#   ...-2bit-none-dequant-eval   INT2 g32 RTN                          -> OFF (RTN_ARM=1 opts in)
#
# That third arm needed scripts/export_gptq_dequant.py to stop hoisting the whole dense model onto
# one card (65 GB against 48). It now streams decoder layers one at a time
# (gptq_quantize_model_sequential(stream_layers=True)), which scripts/test_gptq_stream.py holds to
# bitwise agreement with the resident path and to a peak that does not move with depth.
#
# Run it:
#   mkdir -p logs/commonsense_170k
#   nohup bash jobs/local/cs_qlora_qwen32b_int2_g32_span.sh \
#         > logs/commonsense_170k/qlora_qwen32b_int2_g32_span.job.log 2>&1 &
#
# Env knobs: PASS=1 trains only, PASS=2 exports+evaluates only (default: both).
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=configs/qlora_none_cs170k_int2_g32_ep1_span_qwen32b.yaml
OUT=outputs/qlora_none_qwen32b_cs170k_int2_ep1_span
CKPT="${OUT}-2bit-none/final"
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

# --- Pass 1: train only (Stage 1); --skip_export leaves Stage 1b to pass 2 -----------------
if [ "$PASS" = both ] || [ "$PASS" = 1 ]; then
    echo -e "\n>>> PASS 1/2: train only"
    bash runs/qlora/run_qlora_commonsense.sh "${common_args[@]}" --skip_eval --skip_export \
        2>&1 | tee "$LOGDIR/qlora_qwen32b_cs170k_int2_g32_ep1_span.train.log"
fi

# --- Pass 2: exports + evals, ordered by what the cell actually needs ------------------------
# NOT the pipeline's own Stage-2 order. That runs the RTN dequant arm FIRST and, under
# `set -e`, an RTN failure takes the merged export and the GPTQ floor down with it — which is
# exactly what happened: the RTN export was OOM-killed by the kernel at 51% (the merged fp16
# model at 65 GB plus an fp32 CPU copy of all 448 target weights at 124 GB, against 125 GB of
# host RAM; src/export.py now caps that dict at 16, see keep_targets). So the two arms the cell
# is FOR go first, and the RTN reference goes last where it can only cost itself.
if [ "$PASS" = both ] || [ "$PASS" = 2 ]; then
    echo -e "\n>>> PASS 2/2: exports + vLLM evals"
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT — pass 1 did not finish."; exit 1; }

    MERGED="${OUT}-2bit-none-merged-eval"
    FLOOR="${OUT}-2bit-none-gptq-eval"
    RTN="${OUT}-2bit-none-dequant-eval"

    export_only() {   # $1 = extra flag, $2 = destination
        CUDA_VISIBLE_DEVICES="$LOCAL_EVAL_GPU" python scripts/train.py \
            --config "$CONFIG" --qat_mode none --bits 2 \
            --export_only "$1" \
            --checkpoint_dir "$CKPT" --merge_output_dir "$2"
    }
    eval_one() {      # $1 = model dir
        [ -d "$1" ] || { echo "    (skip eval) $1 not found"; return 0; }
        # Idempotent like the export guards above: a 22k-prompt generative eval of a 32B model is
        # hours, and a re-run to build one missing arm must not redo the arms already scored.
        local summary="results/commonsense_vllm/$(basename "$1").json"
        [ -s "$summary" ] && { echo "    (skip eval) already scored: $summary"; return 0; }
        # Non-fatal, for the same reason every export step is: under `set -e` a failed eval used
        # to take the rest of the job with it. The merged upper bound died on a cross-env
        # tokenizer key and the floor — already built, 62 GB on disk — was never scored.
        bash runs/eval_vllm.sh --model_path "$1" --dataset commonsense \
            --gpus "$LOCAL_EVAL_GPUS" --tag "$(basename "$1")" \
            || { echo "    eval FAILED for $1 — continuing to the next arm"; return 0; }
    }

    # (1) fp16 merged upper bound — also the INPUT the GPTQ floor quantizes.
    if [ -d "$MERGED" ]; then echo ">>> merged export already at $MERGED"; else
        echo -e "\n>>> (1/5) merged fp16 export (upper bound)"
        export_only --export_merged_only "$MERGED" \
            2>&1 | tee "$LOGDIR/qlora_qwen32b_cs170k_int2_g32_ep1_span.merged_export.log"
    fi

    # (2) THE FLOOR: INT2 g32 GPTQ over the same 3500 balanced-sampled records SALT-Q calibrates
    # its frozen codes on (7B calib variant D). Streams decoder layers — see the header.
    if [ -d "$FLOOR" ]; then echo ">>> GPTQ floor already at $FLOOR"; elif [ -d "$MERGED" ]; then
        echo -e "\n>>> (2/5) INT2 g32 GPTQ floor (bcal: 3500 balanced)"
        CUDA_VISIBLE_DEVICES="$LOCAL_EVAL_GPU" python scripts/export_gptq_dequant.py \
            --model_path "$MERGED" --output_dir "$FLOOR" --config "$CONFIG" \
            --bits 2 --group_size 32 \
            --nsamples 3500 --calibration_samples 3500 --calibration_sampling balanced \
            --batch_size 8 \
            2>&1 | tee "$LOGDIR/qlora_qwen32b_cs170k_int2_g32_ep1_span.gptq_floor.log" || \
            echo "GPTQ floor FAILED — the upper bound below is still scored."
    else
        echo "ERROR: $MERGED missing — cannot build the GPTQ floor."
    fi

    echo -e "\n>>> (3/5) scoring the upper bound";  eval_one "$MERGED"
    echo -e "\n>>> (4/5) scoring the floor";        eval_one "$FLOOR"

    # (5) RTN reference — OFF by default. Two reasons, and neither is about the method:
    #   * at INT2 the 7B run of this arm emitted "riiriirii..." to the token cap and scored 0.00
    #     on all eight tasks, so it is a reference for other RTN numbers, never a deployment bound
    #     (that is what the GPTQ floor above is for);
    #   * it is the one step still at real risk of exhausting host RAM. _quantize_all_layers keeps
    #     {name: (W_int uint8, scale, zero)} for EVERY target module because the save loop
    #     dequantizes from it — ~39 GB on top of the 65 GB merged model — and a kernel OOM-kill
    #     does not just lose the step, it leaves the nvidia driver unusable (cuInit -> 3,
    #     NOT_INITIALIZED) until nvidia_uvm is reloaded as root. Fixing it properly means making
    #     merge_and_export quantize->dequantize->write back one layer at a time.
    # Set RTN_ARM=1 to opt in.
    if [ "${RTN_ARM:-0}" = 1 ] && [ ! -d "$RTN" ]; then
        echo -e "\n>>> (5/5) INT2 g32 RTN reference arm"
        export_only --export_dequant "$RTN" \
            2>&1 | tee "$LOGDIR/qlora_qwen32b_cs170k_int2_g32_ep1_span.rtn_export.log" || \
            echo "RTN export failed — the two arms above are unaffected."
    fi
    eval_one "$RTN" || true

    # --- Reclaim the dense exports once their scores are on disk ---------------------------
    # Three 65 GB dirs. The hook-free copies runs/eval_vllm.sh makes are hardlinks
    # (scripts/export_vllm_ready.py), so they cost nothing. The scores live in
    # results/commonsense_vllm/<tag>.{json,jsonl} and the adapter in final/ rebuilds any of them.
    # Guarded on the summary JSON existing: a failed eval keeps its export for a retry.
    # Set KEEP_EXPORTS=1 to keep them anyway.
    if [ "${KEEP_EXPORTS:-0}" != 1 ]; then
        echo -e "\n>>> Reclaiming dense exports whose scores are already written"
        for d in "$RTN" "$FLOOR" "$MERGED"; do
            summary="results/commonsense_vllm/$(basename "$d").json"
            if [ -s "$summary" ]; then
                echo "    scored ($summary) — removing $d and ${d}-vllm"
                rm -rf "$d" "${d}-vllm"
            else
                echo "    NO score at $summary — keeping $d"
            fi
        done
        df -h / | tail -1
    fi
fi

echo "End time: $(date)"
