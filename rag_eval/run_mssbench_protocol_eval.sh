#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Override any of these settings by exporting the corresponding variable.
model_name="${MODEL_NAME:-qwen3vl-2b-ideal}"
python_bin="${PYTHON_BIN:-python3}"
input_root="${INPUT_ROOT:-${repo_root}/rag_eval/results/${model_name}}"
output_root="${EVAL_OUTPUT_ROOT:-${repo_root}/rag_eval/evaluation/${model_name}/mssbench_protocol}"
judge_model="${JUDGE_MODEL:-qwen3.6-flash}"
judge_base_url="${JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
judge_workers="${JUDGE_WORKERS:-4}"
judge_batch_size="${JUDGE_BATCH_SIZE:-128}"
eval_passes="${EVAL_PASSES:-2}"
limit="${LIMIT:-0}"
datasets_text="${DATASETS:-MSSBench VLGuard MM-SafetyBench}"
read -r -a datasets <<< "${datasets_text}"

if [[ "${python_bin}" == */* ]]; then
  [[ -x "${python_bin}" ]] || {
    echo "Python is not executable: ${python_bin}" >&2
    exit 1
  }
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python command not found: ${python_bin}" >&2
  exit 1
fi

if [[ ${#datasets[@]} -eq 0 ]]; then
  echo "DATASETS must contain at least one dataset name." >&2
  exit 1
fi

if ! [[ "${eval_passes}" =~ ^[0-9]+$ ]] || (( eval_passes < 1 )); then
  echo "EVAL_PASSES must be a positive integer." >&2
  exit 1
fi

for dataset in "${datasets[@]}"; do
  answers_path="${input_root}/${dataset}/answers.jsonl"
  [[ -f "${answers_path}" ]] || {
    echo "Missing input file: ${answers_path}" >&2
    exit 1
  }
done

if [[ "${DRY_RUN:-false}" != "true" && "${RECOMPUTE_ONLY:-false}" != "true" ]]; then
  : "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running online evaluation}"
fi

args=(
  --input-root "${input_root}"
  --output-root "${output_root}"
  --datasets "${datasets[@]}"
  --judge-model "${judge_model}"
  --base-url "${judge_base_url}"
  --workers "${judge_workers}"
  --batch-size "${judge_batch_size}"
  --limit "${limit}"
  --flush-every "${FLUSH_EVERY:-20}"
  --max-retries "${MAX_RETRIES:-3}"
  --retry-sleep "${RETRY_SLEEP:-2}"
  --max-image-pixels "${MAX_IMAGE_PIXELS:-786432}"
)

[[ "${DRY_RUN:-false}" == "true" ]] && args+=(--dry-run)
[[ "${RECOMPUTE_ONLY:-false}" == "true" ]] && args+=(--recompute-only)

echo "MSSBench protocol evaluation"
echo "  model:    ${model_name}"
echo "  input:    ${input_root}"
echo "  output:   ${output_root}"
echo "  datasets: ${datasets[*]}"
echo "  judge:    ${judge_model}"
echo "  workers:  ${judge_workers}"
echo "  batch:    ${judge_batch_size}"
echo "  passes:   ${eval_passes}"

for (( pass = 1; pass <= eval_passes; pass++ )); do
  echo "Pass ${pass}/${eval_passes} started."
  "${python_bin}" -m rag_eval.evaluate_mssbench_protocol "${args[@]}"
  echo "Pass ${pass}/${eval_passes} completed."
done

echo "Completed. Summary: ${output_root}/summary.json"
