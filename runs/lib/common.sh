#!/bin/bash
# =============================================================================
# runs/lib/common.sh — sourced by every pipeline under runs/.
#
# Two jobs, both about making a whole class of silent mistakes impossible:
#
#   cd_repo_root                    every pipeline addresses configs/, scripts/, outputs/ and
#                                   datasets/ relative to the repo root, so the caller's cwd
#                                   must not matter. `bash runs/saltq/run_saltq_math.sh` and
#                                   `cd runs/saltq && bash run_saltq_math.sh` now behave the same.
#
#   assert_config_matches_dataset   the config decides what the model TRAINS on; --dataset
#                                   decides what it is SCORED on. Nothing used to tie the two
#                                   together, so `--dataset commonsense` against the default
#                                   metamath config trained on math, evaluated on commonsense,
#                                   and reported a number without a single error message. The
#                                   per-task entry scripts make that hard to do by accident;
#                                   this makes it impossible to do at all.
# =============================================================================

# Resolved at SOURCE time, while the cwd is still whatever the caller started in. Resolving it
# lazily inside the function would break as soon as anything cd'd first, because BASH_SOURCE
# holds the path as written on the `source` line, which is usually relative.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd_repo_root() { cd "$REPO_ROOT"; }

# Map a --dataset name to the local data dir the configs point at.
dataset_dir_for() {
    case "$1" in
        math)        echo "datasets/metamath" ;;
        commonsense) echo "datasets/commonsense" ;;
        # wikitext2 is a RAW-TEXT causal-LM task (data.task_type: lm) scored by perplexity, not
        # by generation + exact match. runs/eval_vllm.sh dispatches it to scripts/eval_ppl.py;
        # everything upstream of eval -- the config/dataset agreement check included -- is the
        # same as for the other two.
        wikitext2)   echo "datasets/wikitext2" ;;
        # alpaca is an instruction-tuning task with NO test split of its own: it is scored on
        # MMLU (lm-eval, 5-shot), an external benchmark the model never trains on. So
        # runs/eval_vllm.sh dispatches it to scripts/eval_mmlu.py, and this mapping exists only
        # to keep the config/dataset agreement check honest about what was TRAINED on.
        alpaca)      echo "datasets/alpaca" ;;
        *) echo "ERROR: unknown --dataset '$1' (expected math, commonsense, wikitext2 or alpaca)" >&2; return 1 ;;
    esac
}

assert_config_matches_dataset() {
    local cfg="$1" task="$2" want have
    want="$(dataset_dir_for "$task")" || exit 1

    if [ ! -f "$cfg" ]; then
        echo "ERROR: config not found: $cfg" >&2
        exit 1
    fi

    have="$(python - "$cfg" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["data"]["train_dataset"])
PY
)"

    if [ "$have" != "$want" ]; then
        echo "ERROR: task mismatch between training and evaluation." >&2
        echo "  --dataset $task  scores  $want/test.json" >&2
        echo "  $cfg  trains on  '$have'" >&2
        echo >&2
        echo "  A model trained on one task and scored on another produces a number that means" >&2
        echo "  nothing, and neither stage would have complained. Point --config at a config for" >&2
        echo "  this task, or pass --skip_eval if you only meant to train." >&2
        exit 1
    fi
}

# Read training.output_dir out of a config. Every artefact a run produces is named from it --
# src/export.py builds "<output_dir>-<bits>bit-<qat_mode>-dequant-eval" and the Trainer writes
# "<output_dir>-<bits>bit-<qat_mode>/final" -- so a pipeline that GUESSES this path instead of
# reading it breaks the moment a config uses a different output_dir.
#
# That guess was written three times under runs/permute_sqat/ as a hardcoded
# `outputs/qlora-sqat-permute*` glob. Two of the three fail LOUDLY (empty match -> "could not
# locate training output dir" after a completed epoch) and the third failed SILENTLY, under
# `shopt -s nullglob`, printing "(no exported eval dirs found)" and exiting 0.
config_output_dir() {
    local cfg="$1" out
    out="$(python - "$cfg" <<'PYCFG'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["training"]["output_dir"])
PYCFG
)" || return 1
    [ -n "$out" ] || { echo "ERROR: no training.output_dir in $cfg" >&2; return 1; }
    echo "$out"
}
