#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# ============================ Configuration ============================
# This script only recalculates metrics from existing answers/judgments.
# It does not load the generation model, run retrieval, or generate answers.
python_bin="${PYTHON_BIN:-python}"
input_root="${INPUT_ROOT:-${repo_root}/rag_eval/results/test_V1}"
output_root="${EVAL_OUTPUT_ROOT:-${repo_root}/rag_eval/evaluation/test_V1}"
datasets="${DATASETS:-all}"
judge_model="${JUDGE_MODEL:-qwen3.6-flash}"
judge_base_url="${JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
judge_workers="${JUDGE_WORKERS:-4}"
limit="${LIMIT:-0}"

args=(
  --input-root "${input_root}"
  --output-root "${output_root}"
  --datasets ${datasets}
  --judge-model "${judge_model}"
  --base-url "${judge_base_url}"
  --workers "${judge_workers}"
  --limit "${limit}"
)

# Keep the existing judgments and only reuse valid entries. Failed or missing
# entries may still trigger API calls, so provide the key when needed.
[[ -n "${DASHSCOPE_API_KEY:-}" ]] && args+=(--api-key "${DASHSCOPE_API_KEY}")
[[ "${ONLINE_VLGUARD:-true}" == "true" ]] && args+=(--online-vlguard)
[[ "${DRY_RUN:-false}" == "true" ]] && args+=(--dry-run)

echo "Recalculating metrics from: ${input_root}"
echo "Writing evaluation summaries to: ${output_root}"
"${python_bin}" -m rag_eval.evaluate_online "${args[@]}"
