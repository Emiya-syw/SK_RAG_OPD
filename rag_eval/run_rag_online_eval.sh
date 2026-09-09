#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"

# ============================ Configuration ============================
# Edit these values for a one-command run. Environment variables override
# them when explicitly provided in the shell.
export DATASETS="${DATASETS:-VLGuard}"
export BASE_MODEL="${BASE_MODEL:-/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking}"
export ADAPTER_PATH="${ADAPTER_PATH:-/home/sunyw/SK_RAG_OPD/outputs/consistent}"
export GPU_IDS="${GPU_IDS:-0,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export MULTI_GPU="${MULTI_GPU:-true}"
export STAGE="generate"

# Retrieval/generation settings. Generation reads the precomputed retrieval
# file when available and does not run the embedding model in this pipeline.
export TOP_K="${TOP_K:-3}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
export DTYPE="${DTYPE:-bfloat16}"
export LIMIT="${LIMIT:-0}"

# Online judge settings. VLGuard is online-evaluated by default.
export ONLINE_VLGUARD="${ONLINE_VLGUARD:-true}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen3.6-flash}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export HARMLESSNESS_SCORE="${HARMLESSNESS_SCORE:-false}"
export HARMFUL_REQUEST_EVAL="${HARMFUL_REQUEST_EVAL:-false}"
export DRY_RUN="${DRY_RUN:-false}"
# Keep the key out of source control. Set it in the environment before
# launching, or uncomment and fill the following line in a private copy:
# export DASHSCOPE_API_KEY="sk-..."

rag_output_root="${RAG_OUTPUT_DIR:-${OUTPUT_DIR:-${repo_root}/rag_eval/results/test_V1}}"
eval_output_root="${EVAL_OUTPUT_ROOT:-${repo_root}/rag_eval/evaluation/test_V1}"

echo "[1/2] Generating RAG answers"
OUTPUT_DIR="${rag_output_root}" \
  STAGE=generate \
  bash "${repo_root}/rag_eval/run_rag_eval.sh"

echo "[2/2] Running online evaluation"
INPUT_ROOT="${rag_output_root}" \
EVAL_OUTPUT_ROOT="${eval_output_root}" \
PYTHON_BIN="${python_bin}" \
  bash "${repo_root}/rag_eval/run_online_eval.sh"

echo "Completed. Answers: ${rag_output_root}"
echo "Evaluation: ${eval_output_root}"
