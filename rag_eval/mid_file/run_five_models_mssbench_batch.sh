#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before starting the evaluation}"

models=(
  qwen2.5-vl-7b-spavl
  qwen2.5-vl-7b-mm-rlhf
  safeguard-vl-rl
  safeqwen2.5-vl-7b
  safework-r1-qwen2.5vl-7b
)
datasets=(MSSBench VLGuard MM-SafetyBench)
workers="${JUDGE_WORKERS:-8}"
batch_size="${JUDGE_BATCH_SIZE:-128}"
model_parallelism="${MODEL_PARALLELISM:-2}"
run_root="${repo_root}/rag_eval/mid_file/mssbench_batch_eval"
log_root="${run_root}/logs"
status_file="${run_root}/status.tsv"

mkdir -p "${log_root}"
printf 'model\tstatus\n' > "${status_file}"

run_model() {
  local model="$1"
  local pass
  for pass in 1 2; do
    echo "[$(date -u +%FT%TZ)] ${model}: pass ${pass} started"
    python3 -m rag_eval.evaluate_mssbench_protocol \
      --input-root "${repo_root}/rag_eval/results/${model}" \
      --output-root "${repo_root}/rag_eval/evaluation/${model}/mssbench_protocol" \
      --datasets "${datasets[@]}" \
      --judge-model qwen3.6-flash \
      --workers "${workers}" \
      --batch-size "${batch_size}" \
      --flush-every 20 \
      --max-retries 3 \
      --retry-sleep 2
    echo "[$(date -u +%FT%TZ)] ${model}: pass ${pass} completed"
  done
}

pids=()
pid_models=()
failures=0

wait_for_first() {
  local pid="${pids[0]}"
  local model="${pid_models[0]}"
  if wait "${pid}"; then
    printf '%s\tcomplete\n' "${model}" >> "${status_file}"
  else
    printf '%s\tfailed\n' "${model}" >> "${status_file}"
    failures=$((failures + 1))
  fi
  pids=("${pids[@]:1}")
  pid_models=("${pid_models[@]:1}")
}

for model in "${models[@]}"; do
  run_model "${model}" > "${log_root}/${model}.out" 2> "${log_root}/${model}.err" &
  pids+=("$!")
  pid_models+=("${model}")
  if (( ${#pids[@]} >= model_parallelism )); then
    wait_for_first
  fi
done

while (( ${#pids[@]} )); do
  wait_for_first
done

if (( failures )); then
  echo "${failures} model evaluation job(s) failed; inspect ${log_root}" >&2
  exit 1
fi

echo "All model evaluations completed."
