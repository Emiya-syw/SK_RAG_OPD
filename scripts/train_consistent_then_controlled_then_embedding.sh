#!/usr/bin/env bash
set -euo pipefail

# Transformers 5.x emits a non-actionable processor kwargs warning repeatedly.
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 六个位置参数依次覆盖三份数据文件和三个阶段的输出目录。
# Edit this block for a repeat run. Command-line arguments, when provided,
# temporarily override these values without changing the script.
consistent_file="data/category_consistent.jsonl"
controlled_file="data/category_controlled.jsonl"
embedding_file="data/category_embedding.jsonl"
consistent_output="outputs/consistent"
controlled_output="outputs/controlled_after_consistent"
embedding_output="outputs/embedding_after_controlled"

consistent_file="${1:-${consistent_file}}"
controlled_file="${2:-${controlled_file}}"
embedding_file="${3:-${embedding_file}}"
consistent_output="${4:-${consistent_output}}"
controlled_output="${5:-${controlled_output}}"
embedding_output="${6:-${embedding_output}}"
script_dir="$(dirname "${BASH_SOURCE[0]}")"

# 三阶段共享 veRL distillation 配置，并通过导出的 PEFT adapter 接续训练。
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k3}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-False}"

echo "[1/3] Training on category-consistent data"
if [[ -n "${CONSISTENT_INIT_PATH:-}" ]]; then
  [[ -d "${CONSISTENT_INIT_PATH}" ]] || { echo "CONSISTENT_INIT_PATH does not exist: ${CONSISTENT_INIT_PATH}" >&2; exit 1; }
  consistent_adapter="${CONSISTENT_INIT_PATH}"
  echo "Using existing consistent LoRA adapter: ${consistent_adapter}"
else
  "${script_dir}/train_consistent.sh" \
    "${consistent_file}" "${consistent_output}"
  consistent_adapter="$("${script_dir}/latest_adapter.sh" "${consistent_output}")"
fi

echo "[2/3] Continuing from the consistent LoRA adapter on controlled data"
LORA_INIT_PATH="${consistent_adapter}" \
  "${script_dir}/train_controlled.sh" \
  "${controlled_file}" "${controlled_output}"
controlled_adapter="$("${script_dir}/latest_adapter.sh" "${controlled_output}")"

echo "[3/3] Continuing from the controlled LoRA adapter on embedding-retrieval data"
LORA_INIT_PATH="${controlled_adapter}" \
  "${script_dir}/train_embedding.sh" \
  "${embedding_file}" "${embedding_output}"
