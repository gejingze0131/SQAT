#!/bin/bash
# =============================================================================
# wait_and_run_saltq.sh — block until the GPUs are actually free, then run run_saltq.sh.
#
# Written because the box was busy with an unrelated training job when SALT-Q was ready to start.
# Rather than contend for compute (both jobs would roughly halve in throughput, with little OOM
# headroom left), this waits the other job out and then launches the full pipeline unattended.
#
# "Free" is deliberately conservative:
#   * nvidia-smi reports NO compute processes at all, and
#   * that stays true across two checks CONFIRM_GAP seconds apart.
# The second check exists because a multi-process job that is restarting (or between accelerate
# ranks) can show an empty process list for a few seconds, and starting into that window would
# just recreate the contention this script exists to avoid.
#
# Usage:
#   bash scripts/wait_and_run_saltq.sh [args passed through to run_saltq.sh]
# =============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/dong/anaconda3/envs/sqat/bin}"
export PATH="$CONDA_ENV_BIN:$PATH"

POLL_INTERVAL="${POLL_INTERVAL:-300}"     # seconds between GPU checks
CONFIRM_GAP="${CONFIRM_GAP:-60}"          # seconds between the two "free" confirmations
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/saltq_${STAMP}.log"
STATUS="logs/saltq_status.txt"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
status() { echo "$*" > "$STATUS"; }

gpu_busy_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$'
}

gpu_mem_summary() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
        | tr '\n' ';'
}

log "=========================================================="
log "SALT-Q unattended launcher"
log "  repo:        $REPO"
log "  python:      $(command -v python)"
log "  pass-through args: $*"
log "  poll every ${POLL_INTERVAL}s, confirm-free gap ${CONFIRM_GAP}s, max wait ${MAX_WAIT_HOURS}h"
log "=========================================================="

DEADLINE=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
WAITED=0

while true; do
    BUSY="$(gpu_busy_pids)"
    if [ -z "$BUSY" ]; then
        log "GPUs look free ($(gpu_mem_summary)) — confirming in ${CONFIRM_GAP}s ..."
        status "waiting: confirming GPUs are free"
        sleep "$CONFIRM_GAP"
        if [ -z "$(gpu_busy_pids)" ]; then
            log "Confirmed free. Starting the SALT-Q pipeline."
            break
        fi
        log "A process reappeared during the confirmation window — keep waiting."
    fi

    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "ERROR: GPUs still busy after ${MAX_WAIT_HOURS}h. Giving up without starting."
        status "aborted: GPUs busy for ${MAX_WAIT_HOURS}h"
        exit 2
    fi

    if [ $(( WAITED % 1800 )) -eq 0 ]; then
        N=$(echo "$BUSY" | wc -l)
        log "Still busy: ${N} compute process(es) — $(gpu_mem_summary)"
    fi
    status "waiting: $(echo "$BUSY" | wc -l) compute process(es) still on the GPUs (waited $((WAITED/60))m)"
    sleep "$POLL_INTERVAL"
    WAITED=$(( WAITED + POLL_INTERVAL ))
done

log "----------------------------------------------------------"
log "Launching: bash run_saltq.sh $*"
log "----------------------------------------------------------"
status "running: SALT-Q pipeline started $(date '+%F %T')"

set -o pipefail
bash run_saltq.sh "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

if [ "$RC" -eq 0 ]; then
    log "SALT-Q pipeline finished OK."
    status "done: pipeline finished OK $(date '+%F %T')"
else
    log "SALT-Q pipeline FAILED with exit code $RC."
    status "failed: exit $RC at $(date '+%F %T')"
fi
log "Log: $LOG"
exit "$RC"
