#!/bin/bash
# =============================================================================
# jobs/local/mm_saltq_int3_g64_span_bcal_sgptql_zp2x.sh
#
# The zp x2 rerun of the MetaMath INT3 g64 SALT-Q cell. ONE key differs from the parent run
# (jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh, GSM8K 56.18 / MATH 10.90):
# qat.saltq.zp_lr_by_bits[3] 1.73e-3 -> 3.46e-3.
#
# WHY. The parent inherited the Commonsense INT3 recipe, where zp x2 was never an arm, so it was
# the only MetaMath cell that did NOT get the multiplier its INT2 sibling got — and it landed
# BELOW that sibling on MATH (10.90 vs 13.68), which is backwards for the easier bit width.
# scripts/measure_saltq_displacement.py on the parent's finished checkpoint says why, in the
# repo's own units: |Δz_N| p50 = 0.058 levels against a 0.1-0.3 target, i.e. the tier that owns
# 98% of the weights was driven 2-5x too weakly. 3.46e-3 x the measured c (~33 at T=4938)
# predicts ~0.115 levels — the band's low edge, and the drive the INT2 sibling actually got.
#
# BOTH OFFLINE BASES ARE REUSED FROM THE PARENT, not rebuilt, and that is the point rather than a
# shortcut: the permutation and the frozen codes depend on group_k / group_size / calibration /
# salient_init, none of which move here. Pointing at the parent's own files makes the two runs
# share the SAME codes bit-for-bit instead of merely identically-configured ones, so the only
# thing that differs between them is the zero-point learning rate. It also saves ~1 h of base
# building. (AGENTS.md "跑 bit 扫描时的基座纪律" — same discipline, applied to an lr sweep.)
#
# NOT CHANGED ON PURPOSE: salient_lr. It measures 0.134 grid steps against a MetaMath-derived
# ~0.5 target, but this recipe's 5e-5 came from Commonsense, where the same tool's note records
# the BEST score at 0.079 steps and the in-band 0.273 as the WORST. The target is dataset-
# dependent and unresolved here; moving it too would make this run two-variable. Sweep separately.
#
# Two cards (0,1) at 8 x 5 x 2 = effective batch 80, T = 4938 — identical to the parent, which
# matters because c ~ 0.47*sqrt(T) is what the predicted displacement above is computed with.
# Card 2 is left alone (the T7 segment ablation is on it).
#
# Run it:
#   mkdir -p logs/metamath
#   setsid nohup bash jobs/local/mm_saltq_int3_g64_span_bcal_sgptql_zp2x.sh \
#         > logs/metamath/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.job.log 2>&1 < /dev/null &
#
# Env knobs: PASS=1 trains only, PASS=2 re-exports + evaluates; default is end to end.
# =============================================================================

export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local2.yaml}"
LOCAL_NUM_GPUS="${LOCAL_NUM_GPUS:-2}"
LOCAL_EVAL_GPUS="${LOCAL_EVAL_GPUS:-0,1}"
LOCAL_EVAL_GPU="${LOCAL_EVAL_GPU:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

PARENT=outputs/saltq_mm_int3_g64_ep1_span_bcal_sgptql
CONFIG=configs/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.yaml
OUT=outputs/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x     # = training.output_dir in the yaml
CKPT="${OUT}-3bit-saltq/final"
LOGDIR=logs/metamath
PASS="${PASS:-both}"
mkdir -p "$LOGDIR"

for d in "$PARENT/permuted_fp16_base" "$PARENT/saltq_base_3bit_g64"; do
    [ -d "$d" ] || { echo "ERROR: the parent run's base is missing: $d"; exit 1; }
done

common_args=(
    --config            "$CONFIG"
    --bits              3
    --num_gpus          "$LOCAL_NUM_GPUS"
    --output_root       "$OUT"
    --permuted_base_dir "$PARENT/permuted_fp16_base"
    --saltq_base_dir    "$PARENT/saltq_base_3bit_g64"
    --train_layernorms  false
    --eval_gpu          "$LOCAL_EVAL_GPU"
    --eval_gpus         "$LOCAL_EVAL_GPUS"
    --note              "MetaMath INT3 g64 span bcal gptq_latent, zp_lr x2 (3.46e-3); bases shared with the 1x parent run"
)

echo "Start time: $(date)"
df -h / | tail -1

if [ "$PASS" = both ]; then
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.log"
elif [ "$PASS" = 1 ]; then
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" --skip_eval "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.train.log"
else
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }
    bash runs/saltq/run_saltq_math.sh "${common_args[@]}" --checkpoint_dir "$CKPT" "$@" \
        2>&1 | tee -a "$LOGDIR/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.export_eval.log"
fi

echo "End time: $(date)"
