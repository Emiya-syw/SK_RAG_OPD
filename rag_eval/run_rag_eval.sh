#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
base_model="${BASE_MODEL:-/home/sunyw/SK_RAG/models/Qwen3-VL-2B-Instruct}"
adapter="${ADAPTER_PATH:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/rag_eval/results/default}"
read -r -a dataset_args <<< "${DATASETS:-all}"

args=(
  --stage "${STAGE:-generate}"
  --datasets "${dataset_args[@]}"
  --data-root "${DATA_ROOT:-${repo_root}/rag_eval/data}"
  --output-dir "${output_dir}"
  --cache-dir "${CACHE_DIR:-${repo_root}/rag_eval/cache}"
  --embedding-repo "${EMBEDDING_REPO:-/home/sunyw/Qwen3-VL-Embedding}"
  --embedding-model "${EMBEDDING_MODEL:-/home/sunyw/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B}"
  --base-model "${base_model}"
  --top-k "${TOP_K:-3}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-4}"
  --max-image-pixels "${MAX_IMAGE_PIXELS:-262144}"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
  --dtype "${DTYPE:-bfloat16}"
  --limit "${LIMIT:-0}"
)

if [[ -n "${adapter}" ]]; then
  args+=(--adapter "${adapter}")
fi
if [[ "${REBUILD_CACHE:-false}" == "true" ]]; then
  args+=(--rebuild-cache)
fi

"${python_bin}" -m rag_eval.run_rag_eval "${args[@]}"
