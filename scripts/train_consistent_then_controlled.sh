#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 四个位置参数依次覆盖两份数据文件和两个阶段的输出目录。
# Edit this block for a repeat run. Command-line arguments, when provided,
# temporarily override these values without changing the script.
consistent_file="data/category_consistent.jsonl"
controlled_file="data/category_controlled.jsonl"
consistent_output="outputs/consistent"
controlled_output="outputs/controlled_after_consistent"

consistent_file="${1:-${consistent_file}}"
controlled_file="${2:-${controlled_file}}"
consistent_output="${3:-${consistent_output}}"
controlled_output="${4:-${controlled_output}}"

# 两阶段共享 veRL distillation 配置，并通过导出的 PEFT adapter 接续训练。
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k3}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-False}"

echo "[1/2] Training on category-consistent data"
if [[ -n "${CONSISTENT_INIT_PATH:-}" ]]; then
  [[ -d "${CONSISTENT_INIT_PATH}" ]] || { echo "CONSISTENT_INIT_PATH does not exist: ${CONSISTENT_INIT_PATH}" >&2; exit 1; }
  consistent_adapter="${CONSISTENT_INIT_PATH}"
  echo "Using existing consistent LoRA adapter: ${consistent_adapter}"
else
  "$(dirname "${BASH_SOURCE[0]}")/train_consistent.sh" \
    "${consistent_file}" "${consistent_output}"
  consistent_adapter="$(scripts/latest_adapter.sh "${consistent_output}")"
fi

echo "[2/2] Continuing from the consistent LoRA adapter on controlled data"
LORA_INIT_PATH="${consistent_adapter}" \
  "$(dirname "${BASH_SOURCE[0]}")/train_controlled.sh" \
  "${controlled_file}" "${controlled_output}"
