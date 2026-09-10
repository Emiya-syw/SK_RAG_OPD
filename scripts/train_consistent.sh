#!/usr/bin/env bash
set -euo pipefail

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# train_file 和 output_dir 分别是训练数据与输出目录。
train_file="${1:-data/category_consistent.jsonl}"
output_dir="${2:-outputs/consistent}"
model_path="${MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
teacher_model_path="${TEACHER_MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking}"
python_bin="${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd/bin/python}"
num_processes="${NUM_PROCESSES:-2}"
port="${MAIN_PROCESS_PORT:-29643}"
validation_enabled="${VALIDATION_ENABLED:-false}"
validation_size="${VALIDATION_SIZE:-256}"
validation_seed="${VALIDATION_SEED:-42}"
validation_max_new_tokens="${VALIDATION_MAX_NEW_TOKENS:-512}"
validation_gpus="${VALIDATION_CUDA_VISIBLE_DEVICES:-0,1}"
enable_thinking="${ENABLE_THINKING:-true}"

# model_path/teacher_model_path 指定学生/教师模型；num_processes 指定 GPU 进程数。
# BATCH_SIZE、GRAD_ACCUM、LR、EPOCHS 控制训练；MAX_PROMPT_LENGTH/MAX_NEW_TOKENS 控制长度。
# TEACHER_PROMPT_MODE 控制 teacher 使用 student 或 privileged prompt。
# VALIDATION_* 控制 checkpoint 保存后的 VLGuard 验证；LoRA 参数控制 adapter。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

"${python_bin}" -m accelerate.commands.launch \
  --num_processes "${num_processes}" \
  --main_process_port "${port}" \
  train_opd.py \
  --train_file "${train_file}" \
  --model_name_or_path "${model_path}" \
  --teacher_model_name_or_path "${teacher_model_path}" \
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
