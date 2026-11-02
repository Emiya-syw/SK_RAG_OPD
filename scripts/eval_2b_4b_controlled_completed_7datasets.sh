#!/usr/bin/env bash
# Evaluate 6 completed runs: four controlled-only stage1 models, followed by
# the 2B/4B base-8B-teacher stage2 models. No training is launched.
# Activate the inference environment and export DASHSCOPE_API_KEY first.
# Defaults: 8x4090, mixed retrieval v2 top3, no thinking, MSSBench judge protocol.
# STAGE=generate: GPU inference only; STAGE=judge: score saved answers only.
# STAGE=all: stream scoring while generating. Re-run to resume existing outputs.
# DRY_RUN=true: check inputs without inference/API calls/checkpoint exports.
# START_FROM=<experiment> resumes the model list at that experiment.
# LIMIT=16 RESULTS_ROOT=/tmp/controlled-smoke-results SCORES_ROOT=/tmp/controlled-smoke-scores ...
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
models=(
  opsd_s2b_stage1_controlled_entropy_jsd
  opd_s2b_t8b_stage1_controlled_entropy_jsd
  opsd_s4b_stage1_controlled_entropy_jsd
  opd_s4b_t8b_stage1_controlled_entropy_jsd
  opd_s2b_t8b_base_stage2_controlled_entropy_jsd
  opd_s4b_t8b_base_stage2_controlled_entropy_jsd
)
case "${1:-}" in
  --list) printf '%s\n' "${models[@]}"; exit 0 ;;
  --help|-h) sed -n '2,11p' "$0"; exit 0 ;;
  '') ;;
  *) echo 'Usage: bash scripts/eval_2b_4b_controlled_completed_7datasets.sh [--list|--help]' >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo 'Too many arguments' >&2; exit 2; }
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export DATASETS='MM-SafetyBench VLGuard MSSBench SPA-VL SIUO-Gen VLSU VLSBench'
export DATA_ROOT="${DATA_ROOT:-${repo_root}/rag_eval/data}"
export TOP_K="${TOP_K:-3}"
export STAGE="${STAGE:-all}"
export DRY_RUN="${DRY_RUN:-false}"
export LIMIT="${LIMIT:-0}"
export ENABLE_THINKING=false
export MULTI_GPU=true
export MERGE_LORA_FOR_INFERENCE=true
export DTYPE=bfloat16
export EVAL_PROFILE=auto
export JUDGE_MODEL="${JUDGE_MODEL:-qwen3.6-flash}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export JUDGE_ADAPTIVE="${JUDGE_ADAPTIVE:-true}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
export JUDGE_MAX_IMAGE_PIXELS="${JUDGE_MAX_IMAGE_PIXELS:-786432}"
# Keep standalone judging consistent with the streaming judge's image limit.
if [[ "$STAGE" == judge ]]; then export MAX_IMAGE_PIXELS="$JUDGE_MAX_IMAGE_PIXELS"; fi
results_root="${RESULTS_ROOT:-${repo_root}/rag_eval/results/controlled_completed_7datasets}"
scores_root="${SCORES_ROOT:-${repo_root}/rag_eval/evaluation/controlled_completed_7datasets}"
log_root="${LOG_ROOT:-${repo_root}/rag_eval/logs/controlled_completed_7datasets}"
case "$STAGE" in all|generate|judge) ;; *) echo 'STAGE must be all, generate, or judge' >&2; exit 2 ;; esac
case "$DRY_RUN" in true|false) ;; *) echo 'DRY_RUN must be true or false' >&2; exit 2 ;; esac
if [[ "$STAGE" != generate && "$DRY_RUN" != true && "${RECOMPUTE_ONLY:-false}" != true ]]; then
  : "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY, or set STAGE=generate}"
fi
start_index=0
if [[ -n "${START_FROM:-}" ]]; then
  start_index=-1
  for i in "${!models[@]}"; do
    if [[ "${models[$i]}" == "$START_FROM" ]]; then start_index="$i"; break; fi
  done
  (( start_index >= 0 )) || { echo "Unknown START_FROM: $START_FROM" >&2; exit 2; }
fi
# Resolve the base/adapter independently for each run; never reuse a caller's
# single-model settings. The existing resolver selects the latest numeric step
# and exports the LoRA adapter if needed (except in DRY_RUN or judge-only mode).
unset MODELS_FILE BASE_MODEL ADAPTER_PATH RUN_NAME MODEL_NAME OUTPUT_DIR EVAL_OUTPUT_ROOT
printf 'Evaluating %s models x 7 datasets; stage=%s; top_k=%s; GPUs=%s\n' \
  "$(( ${#models[@]} - start_index ))" "$STAGE" "$TOP_K" "$GPU_IDS"
printf 'Answers: %s\nScores: %s\n' "$results_root" "$scores_root"
for ((i=start_index; i<${#models[@]}; i++)); do
  model="${models[$i]}"
  echo "[$(date -Is)] START $model"
  if [[ "$DRY_RUN" == true ]]; then
    RUN_NAME="$model" MODEL_NAME="$model" \
    OUTPUT_DIR="$results_root/$model" EVAL_OUTPUT_ROOT="$scores_root/$model/mssbench_protocol" \
      bash "$repo_root/scripts/eval_mixed_top10_v2.sh"
  else
    mkdir -p "$log_root"
    RUN_NAME="$model" MODEL_NAME="$model" \
    OUTPUT_DIR="$results_root/$model" EVAL_OUTPUT_ROOT="$scores_root/$model/mssbench_protocol" \
      bash "$repo_root/scripts/eval_mixed_top10_v2.sh" 2>&1 | tee -a "$log_root/${model}.${STAGE}.log"
  fi
  echo "[$(date -Is)] DONE $model"
done
echo 'All selected runs finished. Inspect summary.json errors before using scores.'
