#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

consistent_file="${1:-data/category_consistent.jsonl}"
controlled_file="${2:-data/category_controlled.jsonl}"
embedding_file="${3:-data/category_embedding.jsonl}"
consistent_output="${4:-outputs/consistent}"
controlled_output="${5:-outputs/controlled_after_consistent}"
embedding_output="${6:-outputs/embedding_after_controlled}"
script_dir="$(dirname "${BASH_SOURCE[0]}")"

echo "[1/3] Training on category-consistent data"
if [[ -n "${CONSISTENT_INIT_PATH:-}" ]]; then
  [[ -d "${CONSISTENT_INIT_PATH}" ]] || { echo "CONSISTENT_INIT_PATH does not exist: ${CONSISTENT_INIT_PATH}" >&2; exit 1; }
  consistent_output="${CONSISTENT_INIT_PATH}"
  echo "Using existing consistent LoRA weights: ${consistent_output}"
else
  "${script_dir}/train_consistent.sh" \
    "${consistent_file}" "${consistent_output}"
fi

echo "[2/3] Continuing from the consistent LoRA adapter on controlled data"
LORA_INIT_PATH="${consistent_output}" \
  "${script_dir}/train_controlled.sh" \
  "${controlled_file}" "${controlled_output}"

echo "[3/3] Continuing from the controlled LoRA adapter on embedding-retrieval data"
LORA_INIT_PATH="${controlled_output}" \
  "${script_dir}/train_embedding.sh" \
  "${embedding_file}" "${embedding_output}"
