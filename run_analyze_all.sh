#!/usr/bin/env bash
set -euo pipefail

# Residual-stream salient-channel analysis for SQAT segment permutation.
#
# The analysis is architecture-agnostic w.r.t. GQA vs MHA: it only measures
# E[x^2] on the residual-stream inputs to q_proj ('attn') and gate_proj ('mlp'),
# both of which are hidden_size-dimensional for every model below. GQA only
# shrinks the k/v *output* heads; the residual input q/k/v share is unchanged,
# and the segment permutation acts on input columns, so no GQA-specific code is
# needed here. Deeper models just get more segment headroom (--max_segments).
#
# Second moments are cached per model so re-running with a different
# --outlier_log_sigma / --max_segments / --group_k_candidates skips the model
# forward pass entirely (--load_second_moments).

DATASET="${DATASET:-metamath}"
N_SAMPLES="${N_SAMPLES:-512}"
SEQ_LEN="${SEQ_LEN:-2048}"
SIGMA="${SIGMA:-3.0}"
GROUP_K="${GROUP_K:-64 128 256 512}"
OUT_ROOT="${OUT_ROOT:-salient_analysis_out}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"

# model_name | output tag | max_segments | extra flags
MODELS=(
#   "meta-llama/Llama-2-7b-hf|llama2-7b|4|"
#   "meta-llama/Llama-3.1-8B|llama31-8b|4|"
#   "Qwen/Qwen2.5-14B|qwen25-14b|6|"
  "Qwen/Qwen2.5-32B|qwen25-32b|6|--trust_remote_code"
)

# Optionally restrict to a subset, e.g.:  ./run_analyze.sh qwen25-14b llama31-8b
WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model tag max_seg extra <<< "$entry"
  want "$tag" || continue

  out_dir="${OUT_ROOT}/${tag}"
  sm_path="${out_dir}/second_moments.pt"
  mkdir -p "$out_dir"

  echo "==================================================================="
  echo "Model: $model  (tag=$tag, max_segments=$max_seg)"
  echo "==================================================================="

  load_flag=()
  save_flag=(--save_second_moments "$sm_path")
  if [ -f "$sm_path" ]; then
    echo "Reusing cached second moments: $sm_path"
    load_flag=(--load_second_moments "$sm_path")
    save_flag=()
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" python analyze_boundary_salient_channels.py \
    --model_name "$model" \
    --dataset "$DATASET" \
    --n_samples "$N_SAMPLES" \
    --seq_len "$SEQ_LEN" \
    --outlier_log_sigma "$SIGMA" \
    --group_k_candidates $GROUP_K \
    --max_segments "$max_seg" \
    --output_dir "$out_dir" \
    "${load_flag[@]}" "${save_flag[@]}" $extra
done
