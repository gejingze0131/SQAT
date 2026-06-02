#!/usr/bin/env bash
# =============================================================================
# run_validation.sh — Stage 1 permutation equivalence validation
#
# Tests that both permutation passes (residual-stream + block-internal) produce
# a numerically equivalent model on plain fp16 weights (no NF4, no LoRA).
#
# What is verified:
#   1. Residual-stream permutation (P_k) + boundary gathers are closed
#   2. Block-internal permutations (P2_l for o_proj, P4_l for down_proj) are closed
#   3. Final logits max-abs error < 0.1 across >=4 prompts
#   4. q_proj output invariance (P must not leak into output rows — RoPE safety)
#   5. num_runtime_permutes == num_segments - 1  (hard assert in code)
#
# Usage:
#   bash run_validation.sh                          # default: Llama-2-7b, 2-segment
#   bash run_validation.sh --boundary_sizes 8 24    # custom 2-segment split
#   bash run_validation.sh --num_boundaries 4        # 4 equal segments (8 layers each)
#   bash run_validation.sh --model_name meta-llama/Llama-2-13b-hf --boundary_sizes 20 20
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI args passed through to sqat_permute.py)
# ---------------------------------------------------------------------------
MODEL_NAME="meta-llama/Llama-2-7b-hf"
N_SAMPLES=512           # small for fast validation; increase to 128 for thorough run
SEQ_LEN=2048            # shorter seq for fast validation
GROUP_K=128
GROUP_SIZE=128
TOP_K_RATIO=0.01
DATASET="wikitext"
BOUNDARY_ARG="--boundary_sizes 2 30"   # Llama-2-7b: 32 layers → [2, 30]
TOL=0.05               # generous tol for validation; hard fail threshold is 0.1 in code
EXTRA_ARGS=""

LOG_DIR="outputs/validation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/stage1_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Parse pass-through arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)       MODEL_NAME="$2";   shift 2 ;;
        --n_samples)        N_SAMPLES="$2";    shift 2 ;;
        --seq_len)          SEQ_LEN="$2";      shift 2 ;;
        --group_k)          GROUP_K="$2";      shift 2 ;;
        --group_size)       GROUP_SIZE="$2";   shift 2 ;;
        --top_k_ratio)      TOP_K_RATIO="$2";  shift 2 ;;
        --dataset)          DATASET="$2";      shift 2 ;;
        --boundary_sizes)
            shift
            SIZES=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SIZES="$SIZES $1"; shift
            done
            BOUNDARY_ARG="--boundary_sizes${SIZES}"
            ;;
        --num_boundaries)
            BOUNDARY_ARG="--num_boundaries $2"; shift 2 ;;
        --tol)              TOL="$2";          shift 2 ;;
        *)                  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " SQAT-Permute Stage 1 Validation"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Boundary:       $BOUNDARY_ARG"
echo " group_k:        $GROUP_K  group_size: $GROUP_SIZE"
echo " top_k_ratio:    $TOP_K_RATIO"
echo " n_samples:      $N_SAMPLES  seq_len: $SEQ_LEN  dataset: $DATASET"
echo " tol (advisory): $TOL   (hard fail in code: 0.1)"
echo " Log:            $LOG_FILE"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run Stage 1 test
# ---------------------------------------------------------------------------
CMD="python sqat_permute.py \
    --test_permute \
    --model_name   $MODEL_NAME \
    --n_samples    $N_SAMPLES \
    --seq_len      $SEQ_LEN \
    --group_k      $GROUP_K \
    --group_size   $GROUP_SIZE \
    --top_k_ratio  $TOP_K_RATIO \
    --dataset      $DATASET \
    --tol          $TOL \
    $BOUNDARY_ARG \
    $EXTRA_ARGS"

echo "Running: $CMD"
echo ""

# Tee to log file and stdout
set +e
$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------------------
# Parse results from log
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Parsing results from $LOG_FILE"
echo "============================================================"

# Extract key metrics
MAX_ERR=$(grep -oP "max_abs_logit_err=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")
NUM_RT=$(grep -oP "num_runtime_permutes=\K[0-9]+" "$LOG_FILE" | tail -1 || echo "N/A")
NUM_INT=$(grep -oP "num_P4_perms=\K[0-9]+" "$LOG_FILE" | tail -1 || \
          grep -oP "Applied P4.*permutations: \K[0-9]+" "$LOG_FILE" | tail -1 || \
          echo "N/A")
NUM_H=$(grep -oP "num_H_layers=\K[0-9]+" "$LOG_FILE" | tail -1 || \
        grep -oP "Hadamard rotation.*\K[0-9]+(?=/)" "$LOG_FILE" | tail -1 || \
        echo "N/A")
QPROJ_ERR=$(grep -oP "q_proj output max_abs_err=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")

PASS_LINE=$(grep "Equivalence PASSED" "$LOG_FILE" || echo "")
FAIL_LINE=$(grep "FAILED\|RuntimeError\|AssertionError\|Traceback" "$LOG_FILE" || echo "")

echo ""
echo " num_runtime_permutes : $NUM_RT   (expected: num_segments - 1)"
echo " num_P4_perms (down_proj): $NUM_INT  (expected: num_layers)"
echo " num_H_layers (Hadamard) : $NUM_H   (expected: num_layers for non-GQA)"
echo " max_abs_logit_err    : $MAX_ERR  (hard threshold: 0.1)"
echo " q_proj output err    : $QPROJ_ERR  (tol: 1e-4)"
echo ""

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌  STAGE 1 FAILED (process exit code $EXIT_CODE)"
    if [[ -n "$FAIL_LINE" ]]; then
        echo ""
        echo "Failure lines:"
        grep -n "FAILED\|RuntimeError\|AssertionError\|Error" "$LOG_FILE" | head -20
    fi
    echo ""
    echo "Full log: $LOG_FILE"
    exit 1
fi

if [[ -n "$PASS_LINE" ]]; then
    echo "✓  STAGE 1 PASSED"
    echo ""
    echo "Full log: $LOG_FILE"
    exit 0
else
    echo "⚠   Stage 1 completed without explicit PASS/FAIL marker — check log manually."
    echo "Full log: $LOG_FILE"
    exit 1
fi
