#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
train_file="${1:?usage: scripts/run_opd.sh TRAIN_JSONL [OUTPUT_DIR]}"
output_dir="${2:-outputs/opd}"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" train_opd.py \
  --train_file "$train_file" \
  --output_dir "$output_dir" \
  --use_lora true \
  --remove_unused_columns false \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-5e-6}" \
  --num_train_epochs "${EPOCHS:-1}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}" \
  --loss_type "${LOSS_TYPE:-sampled_pg}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}"
