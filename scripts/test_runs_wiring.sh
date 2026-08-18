#!/bin/bash
# =============================================================================
# test_runs_wiring.sh — preflight for everything under runs/.
#
# Written after a real failure: runs/eval_vllm.sh sourced ../lib/common.sh, which is correct
# for the two-level runs/<method>/ scripts and wrong for a one-level one. `bash -n` does not
# catch it because the path is only resolved at runtime, so the break surfaced only in stage 3
# — after a full epoch of training had already been spent.
#
# Checks, all static and instant:
#   1. every `source .../common.sh` line resolves to a real file
#   2. every entry script's engine (`exec bash .../_x.sh`) resolves to a real file
#   3. every entry script names a --config that exists, whose data.train_dataset agrees with
#      its --dataset (the same rule runs/lib/common.sh enforces at launch)
#
# Run: bash scripts/test_runs_wiring.sh
# =============================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAIL=0
note() { printf "  %-56s %s\n" "$1" "$2"; }
fail() { note "$1" "FAIL: $2"; FAIL=1; }

echo "[1] source paths"
for f in $(find runs -name '*.sh' | sort); do
    rel=$(grep -oE 'source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[^"]*"' "$f" \
          | sed -E 's|.*\)/([^"]*)"|\1|') || true
    [ -z "$rel" ] && continue
    if [ -f "$(dirname "$f")/$rel" ]; then note "$f" "OK"; else fail "$f" "$rel"; fi
done

echo "[2] entry -> engine"
for f in $(find runs -name 'run_*_math.sh' -o -name 'run_*_commonsense.sh' | sort); do
    eng=$(grep -oE 'exec bash "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[^"]*"' "$f" \
          | sed -E 's|.*\)/([^"]*)"|\1|') || true
    if [ -z "$eng" ]; then fail "$f" "no engine line"; continue; fi
    if [ -f "$(dirname "$f")/$eng" ]; then note "$f" "-> $eng"; else fail "$f" "$eng"; fi
done

echo "[3] entry --config agrees with --dataset"
for f in $(find runs -name 'run_*_math.sh' -o -name 'run_*_commonsense.sh' | sort); do
    cfg=$(grep -oE '^\s+--config\s+\S+'  "$f" | awk '{print $2}')
    task=$(grep -oE '^\s+--dataset\s+\S+' "$f" | awk '{print $2}')
    if [ ! -f "$cfg" ]; then fail "$f" "missing config $cfg"; continue; fi
    want=$([ "$task" = math ] && echo datasets/metamath || echo datasets/commonsense)
    have=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['data']['train_dataset'])" 2>/dev/null)
    if [ "$have" = "$want" ]; then note "$f" "$task <- $cfg"
    else fail "$f" "$cfg trains on '$have', entry scores '$want'"; fi
done

echo
if [ "$FAIL" -eq 0 ]; then echo "PASS — runs/ is wired correctly."; else echo "FAILED"; fi
exit "$FAIL"
