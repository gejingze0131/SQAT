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

# Every per-task entry script under runs/<method>/: a run_*.sh that hands a --dataset to an
# engine. The pattern used to name the two tasks that existed when this file was written (math,
# commonsense), so run_*_wikitext2.sh and run_*_alpaca.sh were silently unchecked -- exactly the
# class of break section [3] exists to catch. Selecting on `--dataset` instead of on the task
# name keeps the next task covered for free, and leaves runs/analysis/ (which drives no
# training task and names no config) out, rather than failing it for lacking one.
entry_scripts() { grep -lF 'exec bash "$(dirname' $(find runs -mindepth 2 -name 'run_*.sh') | sort; }

# The same mapping runs/lib/common.sh:dataset_dir_for enforces at launch.
dataset_dir_for_test() {
    case "$1" in
        math)        echo datasets/metamath ;;
        commonsense) echo datasets/commonsense ;;
        wikitext2)   echo datasets/wikitext2 ;;
        alpaca)      echo datasets/alpaca ;;
        *)           echo "" ;;
    esac
}

echo "[1] source paths"
for f in $(find runs -name '*.sh' | sort); do
    rel=$(grep -oE 'source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[^"]*"' "$f" \
          | sed -E 's|.*\)/([^"]*)"|\1|') || true
    [ -z "$rel" ] && continue
    if [ -f "$(dirname "$f")/$rel" ]; then note "$f" "OK"; else fail "$f" "$rel"; fi
done

echo "[2] entry -> engine"
for f in $(entry_scripts); do
    eng=$(grep -oE 'exec bash "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[^"]*"' "$f" \
          | sed -E 's|.*\)/([^"]*)"|\1|') || true
    if [ -z "$eng" ]; then fail "$f" "no engine line"; continue; fi
    if [ -f "$(dirname "$f")/$eng" ]; then note "$f" "-> $eng"; else fail "$f" "$eng"; fi
done

echo "[3] entry --config agrees with --dataset"
for f in $(entry_scripts); do
    cfg=$(grep -oE '^\s+--config\s+\S+'  "$f" | awk '{print $2}')
    task=$(grep -oE '^\s+--dataset\s+\S+' "$f" | awk '{print $2}')
    if [ ! -f "$cfg" ]; then fail "$f" "missing config $cfg"; continue; fi
    want=$(dataset_dir_for_test "$task")
    if [ -z "$want" ]; then fail "$f" "unknown --dataset '$task'"; continue; fi
    have=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['data']['train_dataset'])" 2>/dev/null)
    if [ "$have" = "$want" ]; then note "$f" "$task <- $cfg"
    else fail "$f" "$cfg trains on '$have', entry scores '$want'"; fi
done

# Every flag an entry script SHOWS IN ITS OWN USAGE COMMENTS has to be one its engine parses.
# The engines each carry a private `case` and they had drifted: --bits was documented on every
# entry and in the README, and three of the five engines answered "Unknown argument: --bits" and
# exited 1. That costs a queue slot and a job submission to discover, and `bash -n` cannot see it
# because the string is only ever compared at runtime.
echo "[4] flags documented in an entry are accepted by its engine"
for f in $(entry_scripts); do
    eng=$(grep -oE 'exec bash "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[^"]*"' "$f" \
          | sed -E 's|.*\)/([^"]*)"|\1|') || true
    [ -n "$eng" ] && [ -f "$(dirname "$f")/$eng" ] || continue
    engine="$(dirname "$f")/$eng"
    # Flags appearing in a commented example invocation of THIS script.
    doc=$(grep -E '^#.*bash runs/' "$f" | grep -oE ' --[a-z_]+' | tr -d ' ' | sort -u)
    missing=""
    for flag in $doc; do
        grep -qE "^\s+$flag\)" "$engine" || missing="$missing $flag"
    done
    if [ -z "$missing" ]; then note "$f" "$(echo $doc | wc -w) documented flag(s) all parsed"
    else fail "$f" "$eng does not accept:$missing"; fi
done

echo
if [ "$FAIL" -eq 0 ]; then echo "PASS — runs/ is wired correctly."; else echo "FAILED"; fi
exit "$FAIL"
