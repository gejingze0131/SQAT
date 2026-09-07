#!/bin/bash
# =============================================================================
# jobs/local/mm_int3_g64_campaign.sh
#
# The whole MetaMath INT3 g64 cell, back to back on the local 3-card box: SALT-Q first (the
# result), then its QA-LoRA control. Both are 2-GPU runs at effective batch 80 and cannot share
# the box, so they are serialized here rather than launched side by side.
#
#   1. jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh   ~16 h  (2 offline bases + train + eval)
#   2. jobs/local/mm_qalora_int3_g64_span_bcal.sh         ~14 h  (GPTQ base + train + eval)
#
# SALT-Q goes first because it is the row the cell exists for, and because its two offline base
# builds are the only genuinely new step — if the recipe is going to fail on this box it fails in
# the first two hours rather than after the control has spent half a day.
#
# A FAILURE IN THE FIRST ARM DOES NOT CANCEL THE SECOND. They share no artifact: SALT-Q builds
# its own permuted fp16 base and frozen INT3 codes under outputs/saltq_mm_int3_g64_*, QA-LoRA
# builds its own dense GPTQ INT3 base under outputs/qalora_mm_int3_*. The exit status of each arm
# is reported at the end and the script's own status is non-zero if either failed.
#
# Run it:
#   mkdir -p logs/metamath
#   setsid nohup bash jobs/local/mm_int3_g64_campaign.sh \
#         > logs/metamath/mm_int3_g64_campaign.log 2>&1 < /dev/null &
#
# Env knobs: ONLY=saltq or ONLY=qalora runs one arm; PASS is forwarded to both arms.
# Per-arm logs land in logs/metamath/ — this file only carries the driver's own narration.
# =============================================================================

set -uo pipefail          # NOT -e: arm 2 must run even when arm 1 fails.

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
mkdir -p logs/metamath

ONLY="${ONLY:-both}"
rc_saltq=skipped
rc_qalora=skipped

if [ "$ONLY" = both ] || [ "$ONLY" = saltq ]; then
    echo "############################################################"
    echo "# ARM 1/2  SALT-Q  MetaMath INT3 g64 k=128 span bcal sgptql"
    echo "# $(date)"
    echo "############################################################"
    bash jobs/local/mm_saltq_int3_g64_span_bcal_sgptql.sh
    rc_saltq=$?
    echo ">>> ARM 1 exit status: $rc_saltq   ($(date))"
fi

if [ "$ONLY" = both ] || [ "$ONLY" = qalora ]; then
    echo "############################################################"
    echo "# ARM 2/2  QA-LoRA control  MetaMath INT3 g64 span bcal"
    echo "# $(date)"
    echo "############################################################"
    bash jobs/local/mm_qalora_int3_g64_span_bcal.sh
    rc_qalora=$?
    echo ">>> ARM 2 exit status: $rc_qalora   ($(date))"
fi

echo "############################################################"
echo "# campaign done  $(date)"
echo "#   SALT-Q  : $rc_saltq"
echo "#   QA-LoRA : $rc_qalora"
echo "# rows land in results_saltq.csv; summaries in results/math_vllm/"
echo "############################################################"

[ "$rc_saltq"  = skipped ] || [ "$rc_saltq"  = 0 ] || exit 1
[ "$rc_qalora" = skipped ] || [ "$rc_qalora" = 0 ] || exit 1
exit 0
