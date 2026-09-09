#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Set VALIDATION_ENABLED=false to make this a no-op in checkpoint pipelines.
if [[ "${VALIDATION_ENABLED:-true}" != "true" ]]; then
  echo "OPD validation disabled"
  exit 0
fi

python_bin="${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd/bin/python}"
checkpoint="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a saved student checkpoint}"
base_model="${BASE_MODEL:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
output="${VALIDATION_OUTPUT:-${checkpoint}/vlguard_validation.json}"

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
