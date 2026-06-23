#!/usr/bin/env bash
# =============================================================================
# run_validation.sh — Stage 1 permutation equivalence + Stage 1b AWQ-S fusion
#
# Tests, on plain fp32 weights (no NF4, no LoRA, no training):
#   Stage 1  — permutation equivalence
#     1. Residual-stream permutation (P_k) + boundary gathers are closed
#     2. Block-internal permutation (P4_l for down_proj, salient-first) is closed
#     3. Final logits max-abs error < 0.1 across >=4 prompts
#     4. q_proj output invariance (P must not leak into output rows — RoPE safety)
#     5. num_runtime_permutes == num_segments - 1
#   Stage 1b — AWQ-S amplify/bake-back fusion (when AWQ_SCALE=true; the default)
#     6. S is [num_layers, group_k] per source, S in [1, max], per-row min == 1
#     7. amplified-space TRAIN fakequant grid == EXPORT quantize->dequant->/S grid
#        on every salient slice (so the /S bake-back deploys bit-identically to training)
#     8. S genuinely changes the quant grid vs no scaling (not a silent no-op)
#
# Usage:
#   bash run_validation.sh                              # Llama-2-7b, 2-segment, AWQ-S on
#   bash run_validation.sh --boundary_sizes 8 24        # custom 2-segment split
#   bash run_validation.sh --group_size 64              # match the training config (gs=64)
#   bash run_validation.sh --no_awq_scale               # permutation equivalence only
#   bash run_validation.sh --awq_alpha 0.5 --awq_max 2.0
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI args passed through to scripts/verify_permute.py)
# ---------------------------------------------------------------------------
MODEL_NAME="meta-llama/Llama-2-7b-hf"
N_SAMPLES=512           # small for fast validation; increase for a thorough run
SEQ_LEN=2048
GROUP_K=128
GROUP_SIZE=128          # pass --group_size 64 to match the training config exactly
TOP_K_RATIO=0.01
DATASET="wikitext"
BOUNDARY_ARG="--boundary_sizes 2 30"   # Llama-2-7b: 32 layers → [2, 30]
TOL=0.05               # generous tol for validation; hard fail threshold is 0.1 in code

# AWQ-style salient scaling (Stage 1b fusion check) — matches the config defaults.
AWQ_SCALE=true         # true → run the amplify/bake-back fusion check; false → skip
AWQ_ALPHA=0.5
AWQ_MAX=2.0
Q_BITS=4
SYMMETRIC=false        # false → asymmetric (the sqat_permute config default)

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
        --awq_scale)        AWQ_SCALE="$2";    shift 2 ;;
        --awq_alpha)        AWQ_ALPHA="$2";    shift 2 ;;
        --awq_max)          AWQ_MAX="$2";      shift 2 ;;
        --q_bits)           Q_BITS="$2";       shift 2 ;;
        --symmetric)        SYMMETRIC="$2";    shift 2 ;;
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

# Map toggles to verify_permute.py flags.
if [ "$AWQ_SCALE" = "true" ]; then AWQ_FLAG="--awq_scale"; else AWQ_FLAG="--no_awq_scale"; fi
if [ "$SYMMETRIC" = "true" ]; then SYM_FLAG="--symmetric"; else SYM_FLAG="--asymmetric"; fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " SQAT-Permute Stage 1 + 1b Validation"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Boundary:       $BOUNDARY_ARG"
echo " group_k:        $GROUP_K  group_size: $GROUP_SIZE"
echo " top_k_ratio:    $TOP_K_RATIO"
echo " n_samples:      $N_SAMPLES  seq_len: $SEQ_LEN  dataset: $DATASET"
echo " tol (advisory): $TOL   (hard fail in code: 0.1)"
echo " AWQ-S fusion:   $AWQ_SCALE  (alpha=$AWQ_ALPHA, max=$AWQ_MAX, q_bits=$Q_BITS, symmetric=$SYMMETRIC)"
echo " Log:            $LOG_FILE"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run Stage 1 (+ 1b) test
# ---------------------------------------------------------------------------
CMD="python scripts/verify_permute.py \
    --model_name   $MODEL_NAME \
    --n_samples    $N_SAMPLES \
    --seq_len      $SEQ_LEN \
    --group_k      $GROUP_K \
    --group_size   $GROUP_SIZE \
    --top_k_ratio  $TOP_K_RATIO \
    --dataset      $DATASET \
    --tol          $TOL \
    $AWQ_FLAG \
    --awq_alpha    $AWQ_ALPHA \
    --awq_max      $AWQ_MAX \
    --q_bits       $Q_BITS \
    $SYM_FLAG \
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

# Stage 1 metrics
MAX_ERR=$(grep -oP "max_abs_logit_err=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")
NUM_RT=$(grep -oP "num_runtime_permutes=\K[0-9]+" "$LOG_FILE" | tail -1 || echo "N/A")
NUM_INT=$(grep -oP "num_P4_perms=\K[0-9]+" "$LOG_FILE" | tail -1 || echo "N/A")
QPROJ_ERR=$(grep -oP "q_proj output max_abs_err=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")

# Stage 1b (AWQ-S) metrics — parsed from the ASCII-only "[AWQ] METRICS ..." summary line
AWQ_GRID=$(grep -oP "\[AWQ\] METRICS grid=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")
AWQ_EFFECT=$(grep -oP "\[AWQ\] METRICS grid=[0-9.e+-]+ effect=\K[0-9.e+-]+" "$LOG_FILE" | tail -1 || echo "N/A")
AWQ_OK=$(grep "FUSION OK" "$LOG_FILE" || echo "")
AWQ_FAIL=$(grep "FUSION FAILED" "$LOG_FILE" || echo "")

PASS_LINE=$(grep "Equivalence PASSED" "$LOG_FILE" || echo "")
FAIL_LINE=$(grep "FAILED\|RuntimeError\|AssertionError\|Traceback" "$LOG_FILE" || echo "")

echo ""
echo " num_runtime_permutes : $NUM_RT   (expected: num_segments - 1)"
echo " num_P4_perms (down_proj): $NUM_INT  (expected: num_layers)"
echo " max_abs_logit_err    : $MAX_ERR  (hard threshold: 0.1)"
echo " q_proj output err    : $QPROJ_ERR  (tol: 1e-4)"
if [ "$AWQ_SCALE" = "true" ]; then
    echo " AWQ train/export grid: $AWQ_GRID  (must be < 1e-4: /S bake-back == training)"
    echo " AWQ S-vs-noS effect  : $AWQ_EFFECT  (should be > 0: S actually changes the grid)"
fi
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

if [ "$AWQ_SCALE" = "true" ] && [[ -n "$AWQ_FAIL" ]]; then
    echo "❌  STAGE 1b (AWQ-S fusion) FAILED — train/export grid mismatch."
    echo "Full log: $LOG_FILE"
    exit 1
fi

if [[ -n "$PASS_LINE" ]]; then
    echo "✓  STAGE 1 PASSED"
    [ "$AWQ_SCALE" = "true" ] && [[ -n "$AWQ_OK" ]] && echo "✓  STAGE 1b (AWQ-S fusion) PASSED"
    echo ""
    echo "Full log: $LOG_FILE"
    exit 0
else
    echo "⚠   Stage 1 completed without explicit PASS/FAIL marker — check log manually."
    echo "Full log: $LOG_FILE"
    exit 1
fi
