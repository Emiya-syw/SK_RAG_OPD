#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CHECKPOINT_PATH 指定 checkpoint；BASE_MODEL 指定学生 base model。
# VALIDATION_SIZE/SEED 控制抽样；VALIDATION_MAX_NEW_TOKENS 控制生成长度。
# CUDA_VISIBLE_DEVICES 指定验证 GPU；VALIDATION_OUTPUT 指定 JSON 输出路径。
# VALIDATION_ENABLED=false 可跳过验证。
if [[ "${VALIDATION_ENABLED:-true}" != "true" ]]; then
  echo "OPD validation disabled"
  exit 0
fi

python_bin="${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd/bin/python}"
checkpoint="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a saved student checkpoint}"
base_model="${BASE_MODEL:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
output="${VALIDATION_OUTPUT:-${checkpoint}/vlguard_validation.json}"
thinking_flag="--enable-thinking"
[[ "${ENABLE_THINKING:-true}" == true ]] || thinking_flag="--no-enable-thinking"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${python_bin}" -m rag_eval.validate_opd \
  --enabled \
  --asr-only \
  --checkpoint "${checkpoint}" \
  --base-model "${base_model}" \
  --output "${output}" \
  --sample-size "${VALIDATION_SIZE:-256}" \
  --seed "${VALIDATION_SEED:-42}" \
  --max-new-tokens "${VALIDATION_MAX_NEW_TOKENS:-512}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS:-262144}" \
  --dtype "${DTYPE:-bfloat16}"
  "${thinking_flag}"
