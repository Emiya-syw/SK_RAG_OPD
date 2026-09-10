#!/usr/bin/env bash
set -euo pipefail

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# train_file/output_dir 指定 controlled 数据和输出目录；LORA_INIT_PATH 用于接续 adapter。
train_file="${1:-data/category_controlled.jsonl}"
output_dir="${2:-outputs/controlled}"
model_path="${MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
teacher_model_path="${TEACHER_MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking}"
python_bin="${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd/bin/python}"
num_processes="${NUM_PROCESSES:-2}"
port="${MAIN_PROCESS_PORT:-29647}"
validation_enabled="${VALIDATION_ENABLED:-false}"
validation_size="${VALIDATION_SIZE:-256}"
validation_seed="${VALIDATION_SEED:-42}"
validation_max_new_tokens="${VALIDATION_MAX_NEW_TOKENS:-512}"
validation_gpus="${VALIDATION_CUDA_VISIBLE_DEVICES:-0,1}"
enable_thinking="${ENABLE_THINKING:-true}"

# 训练、长度、teacher prompt、验证和 LoRA 参数含义与 train_consistent.sh 相同。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

model_args=()
if [[ -n "${LORA_INIT_PATH:-}" ]]; then
  model_args+=(--lora_init_path "${LORA_INIT_PATH}")
fi

"${python_bin}" -m accelerate.commands.launch \
  --num_processes "${num_processes}" \
  --main_process_port "${port}" \
  train_opd.py \
  --train_file "${train_file}" \
  --model_name_or_path "${model_path}" \
  --teacher_model_name_or_path "${teacher_model_path}" \
  "${model_args[@]}" \
  --output_dir "${output_dir}" \
  --remove_unused_columns false \
  --bf16 true \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-5e-6}" \
  --max_grad_norm 0.1 \
  --num_train_epochs "${EPOCHS:-1}" \
  --logging_steps 1 \
  --save_steps 50 \
  --save_total_limit 2 \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}" \
  --max_image_pixels "${MAX_IMAGE_PIXELS:-262144}" \
  --include_reference_answer true \
  --teacher_prompt_mode "${TEACHER_PROMPT_MODE:-student}" \
  --enable_thinking "${enable_thinking}" \
  --validation_enabled "${validation_enabled}" \
  --validation_test_file "${VALIDATION_TEST_FILE:-rag_eval/data/VLGuard/test_qwen3vl_embedding_top3.jsonl}" \
  --validation_sample_size "${validation_size}" \
  --validation_seed "${validation_seed}" \
  --validation_max_new_tokens "${validation_max_new_tokens}" \
  --validation_cuda_visible_devices "${validation_gpus}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --generation_temperature 1.0 \
  --generation_top_p 0.95 \
  --generation_top_k 20 \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}" \
  --loss_type "${LOSS_TYPE:-reverse_kl}" \
  --beta 0.5 \
  --top_k_loss "${TOP_K_LOSS:-128}" \
  --jsd_token_clip 0.05 \
  --use_lora true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --fixed_teacher true
