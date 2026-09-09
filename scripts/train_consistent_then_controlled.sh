#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

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

# Shared training configuration for both phases.
export LOSS_TYPE="${LOSS_TYPE:-reverse_kl}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
export VALIDATION_SIZE="${VALIDATION_SIZE:-256}"
export VALIDATION_SEED="${VALIDATION_SEED:-42}"
export VALIDATION_MAX_NEW_TOKENS="${VALIDATION_MAX_NEW_TOKENS:-512}"
export VALIDATION_CUDA_VISIBLE_DEVICES="${VALIDATION_CUDA_VISIBLE_DEVICES:-0,1}"

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
