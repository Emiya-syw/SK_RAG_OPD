#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

consistent_file="${1:-data/category_consistent.jsonl}"
controlled_file="${2:-data/category_controlled.jsonl}"
consistent_output="${3:-outputs/consistent}"
controlled_output="${4:-outputs/controlled_after_consistent}"

echo "[1/2] Training on category-consistent data"
if [[ -n "${CONSISTENT_INIT_PATH:-}" ]]; then
  [[ -d "${CONSISTENT_INIT_PATH}" ]] || { echo "CONSISTENT_INIT_PATH does not exist: ${CONSISTENT_INIT_PATH}" >&2; exit 1; }
  consistent_output="${CONSISTENT_INIT_PATH}"
  echo "Using existing consistent LoRA weights: ${consistent_output}"
else
  "$(dirname "${BASH_SOURCE[0]}")/train_consistent.sh" \
    "${consistent_file}" "${consistent_output}"
fi

echo "[2/2] Continuing from the consistent LoRA adapter on controlled data"
LORA_INIT_PATH="${consistent_output}" \
  "$(dirname "${BASH_SOURCE[0]}")/train_controlled.sh" \
  "${controlled_file}" "${controlled_output}"
