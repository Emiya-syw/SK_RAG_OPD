#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

consistent_file="${1:-data/category_consistent.jsonl}"
controlled_file="${2:-data/category_controlled.jsonl}"
consistent_output="${3:-outputs/consistent}"
controlled_output="${4:-outputs/controlled_after_consistent}"

echo "[1/2] Training on category-consistent data"
"$(dirname "${BASH_SOURCE[0]}")/train_consistent.sh" \
  "${consistent_file}" "${consistent_output}"

echo "[2/2] Continuing from the consistent LoRA adapter on controlled data"
LORA_INIT_PATH="${consistent_output}" \
  "$(dirname "${BASH_SOURCE[0]}")/train_controlled.sh" \
  "${controlled_file}" "${controlled_output}"
