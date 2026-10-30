#!/usr/bin/env bash
# Six stage-2 checkpoints x top-{1,2,4,5,6} x MSSBench/VLGuard/VLSU.
# Activate the inference environment and export DASHSCOPE_API_KEY first.
# GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_stage2_topk_sweep.sh
# DRY_RUN=true checks all combinations without inference, API calls or export.
# STAGE=generate runs inference only; STAGE=judge scores saved answers only.
# Default STAGE=all overlaps generation and judging within each combination.
# Override RESULTS_ROOT/SCORES_ROOT for an independent run (e.g. a LIMIT=16 smoke).
# Uses precomputed test_mixed_top10_v2.jsonl; scoring is the MSSBench protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  sed -n '2,9p' "$0"
  exit 0
fi
(( $# == 0 )) || { echo 'Usage: run_stage2_topk_sweep.sh [--help]' >&2; exit 2; }

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export DATASETS='MSSBench VLGuard VLSU'
export DATA_ROOT="${DATA_ROOT:-${repo_root}/rag_eval/data}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen3.6-flash}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
export ENABLE_THINKING=false
export LIMIT="${LIMIT:-0}"
export DRY_RUN="${DRY_RUN:-false}"
export STAGE="${STAGE:-all}"
results_root="${RESULTS_ROOT:-${repo_root}/rag_eval/results/topk_sweep}"
scores_root="${SCORES_ROOT:-${repo_root}/rag_eval/evaluation/topk_sweep}"
case "$STAGE" in all|generate|judge) ;; *) echo 'STAGE must be all, generate, or judge' >&2; exit 2 ;; esac
case "$DRY_RUN" in true|false) ;; *) echo 'DRY_RUN must be true or false' >&2; exit 2 ;; esac
if [[ "$STAGE" != generate && "$DRY_RUN" != true ]]; then
  : "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY, or use STAGE=generate}"
fi
# Resolve each experiment independently; inherited single-model settings must
# not accidentally select the same base/adapter for all six experiments.
unset MODELS_FILE BASE_MODEL ADAPTER_PATH OUTPUT_DIR EVAL_OUTPUT_ROOT MODEL_NAME
unset EVAL_PROFILE

models=(
  opsd_s7b_stage2_entropy_jsd
  opsd_s2b_stage2_entropy_jsd
  opsd_s4b_stage2_entropy_jsd
  opsd_s8b_stage2_entropy_jsd
  opd_s2b_t8b_stage2_entropy_jsd
  opd_s4b_t8b_stage2_entropy_jsd
)
top_ks=(1 2 4 5 6)
declare -A bases adapters

# Resolve once so all K values use the same checkpoint during this invocation.
if [[ "$STAGE" != judge ]]; then
  for model in "${models[@]}"; do
    resolved="$(RUN_NAME="$model" "$PYTHON_BIN" -m scripts.resolve_eval_checkpoint)"
    mapfile -t paths <<< "$resolved"
    [[ ${#paths[@]} == 2 && -n "${paths[0]}" && -n "${paths[1]}" ]] || {
      echo "Invalid checkpoint resolution for $model" >&2; exit 1;
    }
    bases["$model"]="${paths[0]}"
    adapters["$model"]="${paths[1]}"
  done
fi

run_one() {
  local model="$1" k="$2" dry="$3"
  local name="${model}_top${k}_3datasets"
  RUN_NAME="$name" MODEL_NAME="$name" \
  BASE_MODEL="${bases[$model]:-}" ADAPTER_PATH="${adapters[$model]:-}" \
  OUTPUT_DIR="$results_root/$name" \
  EVAL_OUTPUT_ROOT="$scores_root/$name/mssbench_protocol" \
  TOP_K="$k" DRY_RUN="$dry" \
    bash "$repo_root/scripts/eval_mixed_top10_v2.sh"
}

echo "Sweep: 6 models x 5 K values x 3 datasets = 90 evaluations; stage=$STAGE"
echo "Answers: $results_root; scores: $scores_root; GPUs: $GPU_IDS"
for model in "${models[@]}"; do
  for k in "${top_ks[@]}"; do
    run_one "$model" "$k" true
  done
done
[[ "$DRY_RUN" != true ]] || { echo 'All preflight checks passed.'; exit 0; }

for model in "${models[@]}"; do
  for k in "${top_ks[@]}"; do
    echo "[$(date -Is)] START $model TOP_K=$k"
    run_one "$model" "$k" false
    echo "[$(date -Is)] DONE $model TOP_K=$k"
  done
done
echo "[$(date -Is)] ALL DONE"
