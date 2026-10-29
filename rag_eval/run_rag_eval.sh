#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
base_model="${BASE_MODEL:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
adapter="${ADAPTER_PATH-/home/sunyw/SK_RAG_OPD/outputs/consistent/}"
output_dir="${OUTPUT_DIR:-${repo_root}/rag_eval/results/test_V1}"
read -r -a dataset_args <<< "${DATASETS:-MSSBench VLGuard}"

# Set MULTI_GPU=true to launch one generation process per GPU. Each process
# receives every N-th test row and writes a shard which is merged afterwards.
if [[ "${MULTI_GPU:-true}" == "true" ]]; then
  if [[ "${STAGE:-generate}" != "generate" ]]; then
    echo "MULTI_GPU mode currently supports STAGE=generate only; run retrieval once first." >&2
    exit 2
  fi
  selected_gpu_ids="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
  if [[ "${AUTO_GPU_POLICY:-true}" == true && -f "${repo_root}/rag_eval/auto_gpu_policy.json" ]]; then
    selected_gpu_ids="$("${python_bin}" -m rag_eval.gpu_capacity \
      --policy "${repo_root}/rag_eval/auto_gpu_policy.json" --output-root "${output_dir}" \
      --datasets "${dataset_args[@]}" --requested "$selected_gpu_ids")"
  fi
  exec "${python_bin}" -m rag_eval.multi_gpu_generate \
    --script "${repo_root}/rag_eval/run_rag_eval.sh" \
    --output-dir "${output_dir}" --datasets "${dataset_args[@]}" \
    --gpu-ids "$selected_gpu_ids"
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
  --prompt-mode "${PROMPT_MODE:-guided}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-4}"
  --max-image-pixels "${MAX_IMAGE_PIXELS:-262144}"
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
  --generation-batch-size "${GENERATION_BATCH_SIZE:-1}"
  --dtype "${DTYPE:-bfloat16}"
  --attn-implementation "${EVAL_ATTN_IMPLEMENTATION:-${ATTN_IMPLEMENTATION:-sdpa}}"
  --limit "${LIMIT:-0}"
  --shard-index "${shard_index}" \
  --num-shards "${num_shards}"
)

if [[ -n "${GENERATION_COMPLETION_DIR:-}" ]]; then
  args+=(--completion-dir "$GENERATION_COMPLETION_DIR")
fi

if [[ "${MERGE_LORA_FOR_INFERENCE:-false}" == "true" ]]; then
  args+=(--merge-lora)
fi

if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  args+=(--enable-thinking)
else
  args+=(--no-enable-thinking)
fi
if [[ "${DIRECT_GENERATION:-false}" == "true" ]]; then
  args+=(--direct-generation)
fi
if [[ -n "${RETRIEVAL_ROOT:-}" ]]; then
  args+=(--retrieval-root "${RETRIEVAL_ROOT}")
fi

if [[ "${num_shards}" -gt 1 || -n "${GENERATION_COMPLETION_DIR:-}" ]]; then
  args+=(--output-suffix ".rank${shard_index}of${num_shards}")
fi

if [[ -n "${adapter}" ]]; then
  args+=(--adapter "${adapter}")
fi
if [[ "${REBUILD_CACHE:-false}" == "true" ]]; then
  args+=(--rebuild-cache)
fi

exec "${python_bin}" -m rag_eval.run_rag_eval "${args[@]}"
