#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
input_root="${INPUT_ROOT:-${repo_root}/rag_eval/results/test_V1}"
output_root="${EVAL_OUTPUT_ROOT:-${repo_root}/rag_eval/evaluation/test_V1}"
api_key="${DASHSCOPE_API_KEY:-}"

args=(
  --input-root "${input_root}" --output-root "${output_root}"
  --datasets ${DATASETS:-all}
  --judge-model "${JUDGE_MODEL:-qwen3.6-flash}"
  --base-url "${JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
  --workers "${JUDGE_WORKERS:-4}" --limit "${LIMIT:-0}"
)
[[ -n "${api_key}" ]] && args+=(--api-key "${api_key}")
[[ "${HARMLESSNESS_SCORE:-false}" == true ]] && args+=(--harmlessness-score)
[[ "${HARMFUL_REQUEST_EVAL:-false}" == true ]] && args+=(--harmful-request-eval)
[[ "${ONLINE_VLGUARD:-false}" == true ]] && args+=(--online-vlguard)
[[ "${DRY_RUN:-false}" == true ]] && args+=(--dry-run)
"${python_bin}" -m rag_eval.evaluate_online "${args[@]}"
