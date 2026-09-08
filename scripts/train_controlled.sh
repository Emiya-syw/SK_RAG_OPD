#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

train_file="${1:-data/category_controlled.jsonl}"
output_dir="${2:-outputs/controlled}"
model_path="${MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
teacher_model_path="${TEACHER_MODEL_PATH:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking}"
python_bin="${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd/bin/python}"
num_processes="${NUM_PROCESSES:-4}"
port="${MAIN_PROCESS_PORT:-29647}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-5e-6}" \
  --max_grad_norm 0.1 \
  --num_train_epochs "${EPOCHS:-1}" \
  --logging_steps 1 \
  --save_steps 50 \
  --save_total_limit 2 \
  --max_prompt_length 4096 \
  --max_image_pixels "${MAX_IMAGE_PIXELS:-262144}" \
  --include_reference_answer false \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --generation_temperature 1.0 \
  --generation_top_p 0.95 \
  --generation_top_k 20 \
  --loss_type "${LOSS_TYPE:-jsd}" \
  --beta 0.5 \
  --top_k_loss "${TOP_K_LOSS:-128}" \
  --jsd_token_clip 0.05 \
  --use_lora true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --fixed_teacher true
