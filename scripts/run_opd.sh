#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# $1 为训练 JSONL，$2 为输出目录。
train_file="${1:?usage: scripts/run_opd.sh TRAIN_JSONL [OUTPUT_DIR]}"
output_dir="${2:-outputs/opd}"
# python_bin 指定解释器；validation_* 控制 checkpoint 验证。
python_bin="${PYTHON_BIN:-python3}"
validation_enabled="${VALIDATION_ENABLED:-false}"
validation_size="${VALIDATION_SIZE:-256}"
validation_seed="${VALIDATION_SEED:-42}"
validation_max_new_tokens="${VALIDATION_MAX_NEW_TOKENS:-512}"
validation_gpus="${VALIDATION_CUDA_VISIBLE_DEVICES:-0,1}"
# 训练/显存参数：batch、梯度累积、序列长度和 rollout 长度。
# OPD 参数：loss_type、repetition_penalty；teacher_prompt_mode 控制 teacher 输入模式。
"$python_bin" train_opd.py \
  --train_file "$train_file" \
  --output_dir "$output_dir" \
  --use_lora true \
  --remove_unused_columns false \
  --gradient_checkpointing true \
  --validation_enabled "${validation_enabled}" \
  --teacher_prompt_mode "${TEACHER_PROMPT_MODE:-student}" \
  --validation_test_file "${VALIDATION_TEST_FILE:-rag_eval/data/VLGuard/test_qwen3vl_embedding_top3.jsonl}" \
  --validation_sample_size "${validation_size}" \
  --validation_seed "${validation_seed}" \
  --validation_max_new_tokens "${validation_max_new_tokens}" \
  --validation_cuda_visible_devices "${validation_gpus}" \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-5e-6}" \
  --num_train_epochs "${EPOCHS:-1}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}" \
  --loss_type "${LOSS_TYPE:-reverse_kl}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}"
