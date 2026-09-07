#!/bin/bash
# =============================================================================
# jobs/local/cs_saltq_int2_g32_seg_ablation.sh
#
# Tracker T7, the segmentation / permutation ablation — INT2 g32 bcal column only.
# Commonsense-170k, Llama-2-7B, SALT-Q k=256, gptq_latent, zp x2. Everything except the number of
# residual permutations is held at the 74.76 "auto segmentation" arm's values.
#
#   arm        segments   runtime gathers   config                   order
#   ---------  ---------  ----------------  -----------------------  --------------------------
#   x1                 1                 0  ..._zp2x_seg1.yaml       FIRST — the headline arm
#   (default)          2                 1  ..._zp2x.yaml            ALREADY MEASURED: 74.76
#   x4                 4                 3  ..._zp2x_seg4.yaml       DEFERRED (not in ARMS)
#   per-layer         32                31  ..._zp2x_seg32.yaml      SECOND — the other extreme
#
# THE TWO ENDS FIRST, AND x4 NOT AT ALL YET. Forty GPU-hours on one card to fill a four-row table
# is a bad trade when the two extremes already answer the question the table exists for: x1 is
# the no-segmentation control (one global P, zero runtime gathers — structurally QEFT's shape),
# per-layer is the coverage oracle, and the default sits between them. If x1 ties the default,
# the segmentation machinery is not earning its complexity and x4 cannot rescue that; if
# per-layer beats the default by a lot, x4 is worth buying and can be added then. Middle points
# are for confirming a trend, not for discovering one. Add it back with ARMS="seg4" later.
#
# THE DEFAULT ARM IS THE x2 ROW, and that is a finding, not an accident of labelling: its log
# reports boundary_sizes=[2, 30], i.e. the DP given group_k=256 and max_segments=4 bought only
# TWO segments, because residual-outlier overflow stops improving after the second and the
# tie-break takes the cheapest nseg that reaches the minimum. So x2 needs no run — it IS 74.76 —
# and x4 is a genuinely different point (the optimal 4-way split the DP declined to buy), not a
# relabelling. force_segments=2 would reproduce [2, 30] exactly; that equality is asserted in the
# DP's unit check rather than bought with ten GPU-hours.
#
# ONE CARD, EFFECTIVE BATCH UNCHANGED. Cards 0 and 1 are held by the MetaMath INT3 campaign, so
# every arm runs on card 2 at 4 x 20 accum x 1 = 80 — the same 80 AND the same per-device 4 as
# the default's 4 x 5 x 4. T stays 1844, so the displacement constant c ~ 0.47*sqrt(T) and every
# lr derived against it transport untouched. Per-device 4 rather than 8 is deliberate: it makes
# the arms differ from the reference in the segmentation alone, and it cannot OOM ten hours in.
#
# EVERY ARM REBUILDS BOTH OFFLINE BASES, and must. force_segments changes the SPLIT, so the
# permutation, the salient slice and therefore the frozen codes are all different;
# scripts/train.py's reuse check now compares force_segments and refuses a mismatched base on its
# own (without that, every arm would silently inherit the default's [2, 30] base and this whole
# table would be four copies of one number).
#
# DISK. ~33 GB per arm live (permuted base 13 + codes 7 + deploy export 13). The deploy export is
# reclaimed once its MEAN(8) is on disk; the two bases are KEPT, because deleting them makes the
# trained checkpoint permanently un-exportable (AGENTS.md invariant 7). RECLAIM_BASES=1 overrides
# that if space runs out and you accept losing re-export. The job refuses to START an arm under
# MIN_FREE_GB, rather than dying halfway through one.
#
# Run it:
#   mkdir -p logs/commonsense_170k
#   setsid nohup bash jobs/local/cs_saltq_int2_g32_seg_ablation.sh \
#         > logs/commonsense_170k/cs_saltq_int2_g32_seg_ablation.job.log 2>&1 < /dev/null &
#
# Env knobs: ARMS="seg1 seg4 seg32" selects a subset (default: all, in that order);
# CS_GPU=2 moves the ablation to another card; MIN_FREE_GB (default 45); KEEP_EXPORTS=1;
# RECLAIM_BASES=1.
# =============================================================================

# ONE CARD BY DEFAULT because cards 0 and 1 are held by the MetaMath INT3 campaign. When that
# finishes, the remaining arms can move to two cards:
#     ACCEL_CONFIG=accelerate_config_local2.yaml CS_NUM_GPUS=2 CS_GPU=0 CS_EVAL_GPUS=0,1 \
#       ARMS=seg32 bash jobs/local/cs_saltq_int2_g32_seg_ablation.sh
# and the arm's config MUST then go gradient_accumulation_steps 20 -> 10, because 4 x 20 x 2 is
# 160, not 80. That is not a detail: 80 sets T = 1844, T sets the displacement constant
# c ~ 0.47*sqrt(T), and every lr in these configs is calibrated against it. The check below
# refuses to launch on a mismatch rather than discovering it in the results table.
CS_GPU="${CS_GPU:-2}"
CS_NUM_GPUS="${CS_NUM_GPUS:-1}"
EFF_BATCH_REQUIRED="${EFF_BATCH_REQUIRED:-80}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_config_local1.yaml}"   # gpu_ids: "2", 1 process
LOCAL_NUM_GPUS="$CS_NUM_GPUS"
LOCAL_EVAL_GPUS="${CS_EVAL_GPUS:-$CS_GPU}"
LOCAL_EVAL_GPU="$CS_GPU"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
set +e     # one arm's failure must not cancel the others

LOGDIR=logs/commonsense_170k
RESDIR=results/commonsense_vllm
ARMS="${ARMS:-seg1 seg32}"        # the two extremes; seg4 is deferred, see the header
MIN_FREE_GB="${MIN_FREE_GB:-45}"
mkdir -p "$LOGDIR" "$RESDIR"

declare -A RC
declare -A NSEG=( [seg1]=1 [seg4]=4 [seg32]=32 [int3_seg1]=1 [int3_seg32]=32 [randsel]=2 )
# int3_seg1 is the TRANSFER TEST, not part of the INT2 column: does the x1 tie survive in
# the cell where the salient budget actually binds (k=128, DP bought 4 segments, global
# union overflows 71% vs 42% at INT2)? It runs --bits 3 and its own config; everything
# else about the driver is unchanged. ARMS=int3_seg1 selects it.

free_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

run_arm() {                    # $1 = tag (seg1 | seg4 | seg32)
    local tag="$1" n="${NSEG[$1]}" bits=2
    local cfg="configs/saltq_cs170k_int2_g32_ep1_span_k256_bcal_sgptql_zp2x_${tag}.yaml"
    local out="outputs/saltq_cs170k_int2_g32_ep1_bcal_${tag}"
    case "$tag" in
      randsel)
        # NOT a segment arm: same [2, 30] segmentation as the default, but the salient sets are
        # random. It rides this driver only for the batch/disk/eval-batch guards.
        cfg="configs/saltq_cs170k_int2_g32_ep1_span_k256_bcal_sgptql_zp2x_randsel.yaml"
        out="outputs/saltq_cs170k_int2_g32_ep1_bcal_randsel" ;;
      int3_seg1|int3_seg32)
        bits=3
        cfg="configs/saltq_cs170k_int3_g64_ep1_span_bcal_sgptql_${tag#int3_}.yaml"
        out="outputs/saltq_cs170k_int3_g64_ep1_bcal_${tag#int3_}" ;;
    esac
    local evd="${out}-${bits}bit-saltq-deploy-eval"

    # The effective batch is the one thing that must not drift between arms, and the only way it
    # can is a GPU-count change that the config's accumulation depth was not adjusted for.
    local eff
    eff="$(python - "$cfg" "$LOCAL_NUM_GPUS" <<'PYEB'
import sys, yaml
t = yaml.safe_load(open(sys.argv[1]))["training"]
print(t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * int(sys.argv[2]))
PYEB
)"
    if [ "$eff" != "$EFF_BATCH_REQUIRED" ]; then
        echo "REFUSING to start $tag: effective batch is $eff, not $EFF_BATCH_REQUIRED."
        echo "  $cfg gives per_device x accum, times $LOCAL_NUM_GPUS GPU(s)."
        echo "  Fix gradient_accumulation_steps in that config (20 for 1 card, 10 for 2)."
        RC[$tag]=1; return 1
    fi

    local have; have="$(free_gb)"
    if [ "$have" -lt "$MIN_FREE_GB" ]; then
        echo "REFUSING to start $tag: ${have} GB free, need ${MIN_FREE_GB}."
        echo "  The biggest reclaimable things on this box right now:"
        du -sh outputs/* 2>/dev/null | sort -h | tail -5
        RC[$tag]=1; return 1
    fi

    echo "############################################################"
    echo "# T7 segment ablation — ${tag}: ${n} residual permutation(s), $((n - 1)) runtime gather(s)"
    echo "# $(date)   free: ${have} GB"
    echo "############################################################"
    bash runs/saltq/run_saltq_commonsense.sh \
        --config      "$cfg" \
        --bits        "$bits" \
        --num_gpus    "$LOCAL_NUM_GPUS" \
        --output_root "$out" \
        --train_layernorms false \
        --eval_gpu    "$LOCAL_EVAL_GPU" --eval_gpus "$LOCAL_EVAL_GPUS" \
        --note        "${NOTE_OVERRIDE:-T7 segment ablation INT${bits}: ${n} segment(s), $((n - 1)) runtime gather(s)}" \
        2>&1 | tee "$LOGDIR/$(basename "$out").log"
    RC[$tag]=$?

    if [ "${KEEP_EXPORTS:-0}" != 1 ] && [ -s "$RESDIR/$(basename "$evd").json" ]; then
        echo ">>> scored — reclaiming $evd"
        rm -rf "$evd" "${evd}-vllm"
    fi
    if [ "${RECLAIM_BASES:-0}" = 1 ] && [ -s "$RESDIR/$(basename "$evd").json" ]; then
        echo ">>> RECLAIM_BASES=1 — dropping the offline bases of $tag (checkpoint stays, but"
        echo "    it can no longer be exported without rebuilding them)"
        rm -rf "$out/permuted_fp16_base" "$out"/saltq_base_*
    fi
    df -h / | tail -1
}

echo "Start time: $(date)"
echo "arms: $ARMS   card: $CS_GPU"
df -h / | tail -1
for a in $ARMS; do run_arm "$a"; done

# --- the column ------------------------------------------------------------------------------
echo -e "\n>>> T7 INT2 g32 bcal column"
python - <<'PY'
import glob, json, os
TASKS = ["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Easy","ARC-Challenge","openbookqa"]
rows = [("auto segmentation (default, = x2)", 2, 1, None)]
for tag, n in (("seg1", 1), ("seg4", 4), ("seg32", 32)):
    p = f"results/commonsense_vllm/saltq_cs170k_int2_g32_ep1_bcal_{tag}-2bit-saltq-deploy-eval.json"
    rows.append((f"segment P x{n}" if n < 32 else "per-layer P (oracle)", n, n - 1,
                 p if os.path.exists(p) else None))
print(f"{'arm':<36} {'segs':>5} {'gathers':>8} {'MEAN(8)':>9}")
for name, n, g, p in rows:
    if p is None:
        avg = "74.76" if n == 2 else "—"
    else:
        r = json.load(open(p))
        res = r.get("results", r)
        vals = [v if isinstance(v, (int, float)) else v.get("acc", v.get("accuracy"))
                for k, v in res.items() if k != "mean"]
        vals = [v for v in vals if isinstance(v, (int, float))]
        avg = f"{100*sum(vals)/len(vals):.2f}" if vals and max(vals) <= 1 else \
              (f"{sum(vals)/len(vals):.2f}" if vals else "?")
    print(f"{name:<36} {n:>5} {g:>8} {avg:>9}")
print("\n(step time: read train_runtime / samples_per_second from each arm's log tail)")
PY

echo "############################################################"
echo "# segment ablation done  $(date)"
for a in $ARMS; do printf "#   %-6s : %s\n" "$a" "${RC[$a]-skipped}"; done
echo "############################################################"
for a in $ARMS; do
    v="${RC[$a]-skipped}"; [ "$v" = skipped ] || [ "$v" = 0 ] || exit 1
done
exit 0
