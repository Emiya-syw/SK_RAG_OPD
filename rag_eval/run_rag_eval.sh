#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
base_model="${BASE_MODEL:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
adapter="${ADAPTER_PATH:-/home/sunyw/SK_RAG_OPD/outputs/consistent/}"
output_dir="${OUTPUT_DIR:-${repo_root}/rag_eval/results/test_V1}"
read -r -a dataset_args <<< "${DATASETS:-VLGuard}"

# Set MULTI_GPU=true to launch one generation process per GPU. Each process
# receives every N-th test row and writes a shard which is merged afterwards.
if [[ "${MULTI_GPU:-true}" == "true" ]]; then
  if [[ "${STAGE:-generate}" != "generate" ]]; then
    echo "MULTI_GPU mode currently supports STAGE=generate only; run retrieval once first." >&2
    exit 2
  fi
  IFS=',' read -r -a gpu_ids <<< "${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
  num_shards="${#gpu_ids[@]}"
  pids=()
  for shard_index in "${!gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard_index]}" MULTI_GPU=false \
      "${BASH_SOURCE[0]}" \
      --internal-multi-gpu-shard "${shard_index}" "${num_shards}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
  "${python_bin}" -m rag_eval.merge_answer_shards \
    --output-dir "${output_dir}" --datasets "${dataset_args[@]}" --num-shards "${num_shards}"
  exit 0
fi

if [[ "${1:-}" == "--internal-multi-gpu-shard" ]]; then
  shard_index="$2"
  num_shards="$3"
else
  shard_index=0
  num_shards=1
fi

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
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
  --generation-batch-size "${GENERATION_BATCH_SIZE:-1}"
  --dtype "${DTYPE:-bfloat16}"
  --limit "${LIMIT:-0}"
  --shard-index "${shard_index}" \
  --num-shards "${num_shards}"
)

if [[ "${ENABLE_THINKING:-true}" == "true" ]]; then
  args+=(--enable-thinking)
else
  args+=(--no-enable-thinking)
fi

if [[ "${num_shards}" -gt 1 ]]; then
  args+=(--output-suffix ".rank${shard_index}of${num_shards}")
fi

if [[ -n "${adapter}" ]]; then
  args+=(--adapter "${adapter}")
fi
if [[ "${REBUILD_CACHE:-false}" == "true" ]]; then
  args+=(--rebuild-cache)
fi

"${python_bin}" -m rag_eval.run_rag_eval "${args[@]}"
