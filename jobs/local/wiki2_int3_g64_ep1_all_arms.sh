#!/bin/bash
# =============================================================================
# jobs/local/wiki2_int3_g64_ep1_all_arms.sh
#
# The WikiText-2 INT3 g64 1-epoch cell, all arms, on ONE card. Local-box port of
# jobs/wiki2_int3_ep1_all_arms.pbs + jobs/wiki2_int3_ep1_floor.pbs, which were written for four
# cluster cards and never ran — SALTQ_experiment_tracker_v2.md T10 still lists "Wiki2 INT3 g64 格"
# as 未跑.
#
#   1. QLoRA        -> the fp16 UPPER bound (merged, unquantized) + the RTN dequant reference
#   2. QLoRA->GPTQ  -> THE FLOOR: that same merged checkpoint through GPTQ INT3 g64
#   3. QA-LoRA      -> pure-INT3 baseline, fresh GPTQ INT3 g64 base
#   4. QEFT         -> mixed-precision baseline (INT3 + k=128 fp16 weak columns, 3.35 eff. bits),
#                      with --with_base so its UNTRAINED base is scored too; at INT2 that base
#                      already beat both pure-2-bit arms' fine-tuned results, so the row is
#                      unreadable without it
#   5. SALT-Q       -> the method, k=128, gptq_latent, z-only
#
# ONE EPOCH ONLY, which is the cell as designed, not a shortcut: the INT2 pair showed that at 3
# epochs this column ranks arms by REGULARIZATION rather than by quantization quality (test/train
# perplexity ratios 6.75 / 3.93 / 1.90 / 1.30, perfectly anti-correlated with held-out PPL). At 1
# epoch every ratio collapses into 1.05-1.68 and the column measures what it is for. All four
# configs already carry num_epochs: 1.
#
# ONE CARD, CARD 2, and this costs the cell nothing. Cards 0 and 1 are held for over a day by the
# MetaMath INT3 campaign. WikiText-2's effective batch is 16 and 16 divides onto one card as
# 4 x 4 accumulation, so T stays 176 and every lr in these configs — each of which carries an
# explicit sqrt(T) factor — transports untouched. What one card costs is wall clock, on a column
# whose training runs are ~180 optimizer steps; the offline base builds dominate either way.
# accelerate_config_local1.yaml pins the trainers, EVAL_GPU/EVAL_GPUS/BASE_GPU/EXPORT_GPU pin
# every other stage.
#
# DISK IS THE BINDING CONSTRAINT, not memory. Five arms x up to three dense 7B artifacts each is
# ~140 GB, against ~215 GB free with the MetaMath campaign also writing. So every dense *-eval
# directory is REMOVED once its perplexity is on disk in results/wikitext2_ppl/ — they rebuild
# from the checkpoint and the offline base, which are kept. KEEP_EXPORTS=1 disables that.
# The QLoRA merged export is the FLOOR's input, so it is reclaimed only after arm 2.
#
# NO ARM CAN KILL ANOTHER. They share no artifact and each is run non-fatally; the exit status of
# each is reported at the end and the script's own status is non-zero if any failed.
#
# Run it:
#   mkdir -p logs/wikitext2
#   setsid nohup bash jobs/local/wiki2_int3_g64_ep1_all_arms.sh \
#         > logs/wikitext2/wiki2_int3_g64_ep1_all_arms.job.log 2>&1 < /dev/null &
#
# Env knobs: ARMS="qlora floor qalora qeft saltq" selects a subset (default: all, in that order);
# KEEP_EXPORTS=1 keeps the dense artifacts; WIKI2_GPU=2 moves the whole cell to another card.
# =============================================================================

WIKI2_GPU="${WIKI2_GPU:-2}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local1.yaml}"   # gpu_ids: "2", 1 process
LOCAL_NUM_GPUS=1
LOCAL_EVAL_GPUS="$WIKI2_GPU"
LOCAL_EVAL_GPU="$WIKI2_GPU"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# _env.sh runs under `set -e`; from here on an arm that fails must not take the rest with it.
set +e

LOGDIR=logs/wikitext2
PPLDIR=results/wikitext2_ppl
ARMS="${ARMS:-qlora floor qalora qeft saltq}"
mkdir -p "$LOGDIR" "$PPLDIR"

QLORA_OUT=outputs/qlora_none_wiki2_int3_g64_ep1
QALORA_OUT=outputs/qalora_wiki2_int3_g64_ep1
QEFT_OUT=outputs/qeft_wiki2_int3_g64_ep1_lr16e5
QEFT_BASE=outputs/qeft_bases/Llama-2-7B_int3_g64_k128_asym_wiki2
SALTQ_OUT=outputs/saltq_wiki2_int3_g64_ep1_bcal_sgptql

declare -A RC

has_arm() { [[ " $ARMS " == *" $1 "* ]]; }

# Reclaim a dense export ONLY once its perplexity exists — an unscored artifact is the one thing
# that cannot be rebuilt for free (it would need the whole arm again to find out it failed).
reclaim() {
    [ "${KEEP_EXPORTS:-0}" = 1 ] && { echo ">>> KEEP_EXPORTS=1 — keeping $*"; return 0; }
    for d in "$@"; do
        [ -d "$d" ] || continue
        if [ -s "$PPLDIR/$(basename "$d").json" ]; then
            echo ">>> scored — reclaiming $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
            rm -rf "$d"
        else
            echo ">>> NO perplexity at $PPLDIR/$(basename "$d").json — KEEPING $d"
        fi
    done
    df -h / | tail -1
}

echo "Start time: $(date)"
echo "arms: $ARMS   card: $WIKI2_GPU"
df -h / | tail -1

# --- 1. QLoRA: fp16 upper bound (+ the RTN dequant reference, an off-cell row) ---------------
if has_arm qlora; then
    echo -e "\n########## 1/5  QLoRA INT3 — fp16 upper bound + the floor's parent ##########"
    bash runs/qlora/run_qlora_wikitext2.sh \
        --config     configs/qlora_none_wiki2_int3_g64_ep1.yaml \
        --bits 3 --num_gpus "$LOCAL_NUM_GPUS" \
        --output_dir "$QLORA_OUT" \
        --eval_gpu   "$LOCAL_EVAL_GPU" --eval_gpus "$LOCAL_EVAL_GPUS" \
        2>&1 | tee "$LOGDIR/qlora_none_wiki2_int3_g64_ep1.log"
    RC[qlora]=$?
    # The merged export is arm 2's INPUT — only the RTN reference can go now.
    reclaim "${QLORA_OUT}-3bit-none-dequant-eval"
fi

# --- 2. THE FLOOR: the same merged checkpoint through GPTQ INT3 g64 --------------------------
# Not the pipeline's own `-dequant-eval`: that one is plain RTN (src/export.py drops the GPTQ
# settings for qat_mode=none and says so), which is a different, much worse artifact. The floor
# SALT-Q has to clear is GPTQ on this task's own 524k in-domain tokens.
if has_arm floor; then
    echo -e "\n########## 2/5  QLoRA -> GPTQ INT3 g64 — THE FLOOR ##########"
    SRC="${QLORA_OUT}-3bit-none-merged-eval"
    DST="${QLORA_OUT}-3bit-none-gptq-eval"
    if [ ! -f "$SRC/config.json" ]; then
        echo "ERROR: $SRC not found — the QLoRA arm has to run first."
        RC[floor]=1
    else
        (
          CUDA_VISIBLE_DEVICES="$WIKI2_GPU" python scripts/export_gptq_dequant.py \
              --model_path "$SRC" --output_dir "$DST" \
              --config configs/qlora_none_wiki2_int3_g64_ep1.yaml --bits 3 --group_size 64 \
              --nsamples 256 --calibration_samples 256 --calibration_seq_len 2048 --batch_size 2 &&
          bash runs/eval_vllm.sh --model_path "$DST" --dataset wikitext2 \
              --gpus "$WIKI2_GPU" --tag "$(basename "$DST")" &&
          python scripts/collect_saltq_results.py \
              --results_dir "$PPLDIR" --csv results_saltq.csv \
              --config configs/qlora_none_wiki2_int3_g64_ep1.yaml \
              --filter "$(basename "$DST")" \
              --note "T2 Wiki2 INT3 g64 1-epoch FLOOR: QLoRA merged fp16 -> GPTQ INT3 g64"
        ) 2>&1 | tee "$LOGDIR/qlora_wiki2_int3_ep1_gptq_floor.log"
        RC[floor]=$?
    fi
    reclaim "$DST" "$SRC"
fi

# --- 3. QA-LoRA: pure INT3, fresh GPTQ INT3 g64 base -----------------------------------------
if has_arm qalora; then
    echo -e "\n########## 3/5  QA-LoRA INT3 g64 ##########"
    bash runs/qalora/run_qalora_wikitext2.sh \
        --config     configs/qalora_wiki2_int3_g64_ep1.yaml \
        --bits 3 --num_gpus "$LOCAL_NUM_GPUS" \
        --output_dir "$QALORA_OUT" \
        --eval_gpu   "$LOCAL_EVAL_GPU" --eval_gpus "$LOCAL_EVAL_GPUS" \
        --note       "T2 Wiki2 INT3 g64 1-epoch cell: merged INT3 artifact" \
        2>&1 | tee "$LOGDIR/qalora_wiki2_int3_g64_ep1.log"
    RC[qalora]=$?
    reclaim "${QALORA_OUT}-3bit-qalora-dequant-eval" "${QALORA_OUT}-3bit-qalora-merged-eval"
fi

# --- 4. QEFT: mixed precision (INT3 + k=128 fp16 weak columns), trained AND bare -------------
# This pipeline has no --output_dir / --eval_gpu of its own: Stage 0 takes BASE_GPU, Stage 2 takes
# EXPORT_GPU (added for exactly this reason — it used to be a hardcoded card 0), Stage 1's
# torchrun inherits CUDA_VISIBLE_DEVICES, and Stage 3 takes --eval_gpus.
if has_arm qeft; then
    echo -e "\n########## 4/5  QEFT INT3 k=128 (+ its untrained base) ##########"
    CUDA_VISIBLE_DEVICES="$WIKI2_GPU" BASE_GPU="$WIKI2_GPU" EXPORT_GPU="$WIKI2_GPU" \
    bash runs/qeft/run_qeft_wikitext2.sh \
        --config     baseline/QEFT/sqat/configs/qeft_wiki2_int3_g64_ep1_lr16e5.yaml \
        --bits 3 --group_size 64 --with_base --num_gpus "$LOCAL_NUM_GPUS" \
        --eval_gpus  "$LOCAL_EVAL_GPUS" \
        --note       "T2 Wiki2 INT3 g64 1-epoch cell: 3+fp16 mixed-precision artifact (k=128)" \
        2>&1 | tee "$LOGDIR/qeft_wiki2_int3_g64_ep1.log"
    RC[qeft]=$?
    reclaim "${QEFT_OUT}-3bit-qeft-dense-eval" "${QEFT_BASE}-3bit-qeftbase-eval"
fi

# --- 5. SALT-Q: the method ------------------------------------------------------------------
if has_arm saltq; then
    echo -e "\n########## 5/5  SALT-Q INT3 g64 k=128 (fresh permuted base + fresh codes) ##########"
    bash runs/saltq/run_saltq_wikitext2.sh \
        --config           configs/saltq_wiki2_int3_g64_ep1_bcal_sgptql.yaml \
        --bits 3 --num_gpus "$LOCAL_NUM_GPUS" \
        --output_root      "$SALTQ_OUT" \
        --train_layernorms false \
        --eval_gpu   "$LOCAL_EVAL_GPU" --eval_gpus "$LOCAL_EVAL_GPUS" \
        --note             "T2 Wiki2 INT3 g64 1-epoch cell: deployed artifact (identity merge)" \
        2>&1 | tee "$LOGDIR/saltq_wiki2_int3_g64_ep1.log"
    RC[saltq]=$?
    reclaim "${SALTQ_OUT}-3bit-saltq-deploy-eval"
fi

# --- the cell -------------------------------------------------------------------------------
echo -e "\n>>> INT3 g64 1-epoch cell, test-split PPL"
python - <<'PY'
import glob, json, os
seen = sorted(set(glob.glob("results/wikitext2_ppl/*int3*.json")))
if not seen:
    print("  (no INT3 perplexity summaries yet)")
for p in seen:
    r = json.load(open(p))
    print(f"{os.path.basename(p)[:-5]:<62} " +
          "  ".join(f"{k[8:]}: {v['ppl']:8.4f}" for k, v in sorted(r["results"].items())))
PY

echo "############################################################"
echo "# wiki2 INT3 g64 1-epoch cell done  $(date)"
for a in qlora floor qalora qeft saltq; do
    printf "#   %-8s : %s\n" "$a" "${RC[$a]-skipped}"
done
echo "# rows land in results_saltq.csv; summaries in $PPLDIR/"
echo "############################################################"
df -h / | tail -1

for a in qlora floor qalora qeft saltq; do
    v="${RC[$a]-skipped}"
    [ "$v" = skipped ] || [ "$v" = 0 ] || exit 1
done
exit 0
