#!/usr/bin/env bash
# Activate the inference/training environment and export DASHSCOPE_API_KEY first.
# Optional: DRY_RUN=true for input preflight; START_FROM=<experiment> to resume
# from a specific experiment. Existing matching answers/judgments are reused.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiments=(
  opsd_s7b_stage1_entropy_jsd
  opsd_s2b_stage1_entropy_jsd_v4
  opsd_s4b_stage1_entropy_jsd
  opsd_s8b_stage1_entropy_jsd
  opsd_s7b_stage2_entropy_jsd
  opsd_s2b_stage2_entropy_jsd
  opsd_s4b_stage2_entropy_jsd
  opsd_s8b_stage2_entropy_jsd
  opd_s2b_t8b_stage1_entropy_jsd
  opd_s4b_t8b_stage1_entropy_jsd
  opd_s2b_t8b_stage2_entropy_jsd
  opd_s4b_t8b_stage2_entropy_jsd
  opsd_s8b_stage1_entropy_jsd_6k_v2
)

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export STAGE=all MULTI_GPU=true MAX_NEW_TOKENS=1536 TOP_K=3 PROMPT_MODE=guided
export EVAL_PROFILE=auto
# Each experiment resolves its own paths and batch profile. Prevent inherited
# single-model settings from silently selecting the same model for every run.
unset BASE_MODEL ADAPTER_PATH MODELS_FILE GENERATION_BATCH_SIZE
if [[ "${DRY_RUN:-false}" != true ]]; then
  : "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY before starting online scoring}"
fi

start_from="${START_FROM:-${experiments[0]}}"
found=false
for experiment in "${experiments[@]}"; do
  [[ "$experiment" != "$start_from" ]] || found=true
done
[[ "$found" == true ]] || { echo "Unknown START_FROM: $start_from" >&2; exit 2; }

results_root="${RESULTS_ROOT:-$repo_root/rag_eval/results}"
scores_root="${SCORES_ROOT:-$repo_root/rag_eval/evaluation}"
logs_root="${LOGS_ROOT:-$repo_root/rag_eval/logs/mixed_v2_7datasets}"
mkdir -p "$logs_root"
started=false
for experiment in "${experiments[@]}"; do
  [[ "$experiment" != "$start_from" ]] || started=true
  [[ "$started" == true ]] || continue
  echo "[$(date -Is)] Starting $experiment"
  OUTPUT_DIR="$results_root/${experiment}_mixed_v2_7datasets" \
  EVAL_OUTPUT_ROOT="$scores_root/${experiment}_mixed_v2_7datasets/mssbench_protocol" \
    bash scripts/eval_mixed_top10_v2.sh "$experiment" \
    2>&1 | tee -a "$logs_root/${experiment}.log"
  echo "[$(date -Is)] Finished $experiment"
done
