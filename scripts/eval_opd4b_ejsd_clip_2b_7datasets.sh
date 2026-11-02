#!/usr/bin/env bash
# 4B OPD EJSD Stage2：依次评测 CLIP、2B 检索文件。
# 两组独立输出，每组七基准；默认 top3、8x4090，边生成边评分。
# STAGE=generate 只生成；STAGE=judge 只评分；DRY_RUN=true 只检查。
# 自动读取 outputs/<RUN_NAME> 的最新 checkpoint，必要时导出 adapter。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export DATASETS='MM-SafetyBench VLGuard MSSBench SPA-VL SIUO-Gen VLSU VLSBench'
export DATA_ROOT="${DATA_ROOT:-$PWD/rag_eval/data}"
export TOP_K="${TOP_K:-3}"
export STAGE="${STAGE:-all}"
export DRY_RUN="${DRY_RUN:-false}"
export LIMIT="${LIMIT:-0}"
export ENABLE_THINKING=false MULTI_GPU=true MERGE_LORA_FOR_INFERENCE=true
export DTYPE=bfloat16 EVAL_PROFILE=auto
export JUDGE_MODEL="${JUDGE_MODEL:-qwen3.6-flash}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export JUDGE_ADAPTIVE="${JUDGE_ADAPTIVE:-true}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
export JUDGE_MAX_IMAGE_PIXELS="${JUDGE_MAX_IMAGE_PIXELS:-786432}"
if [[ "$STAGE" == judge ]]; then export MAX_IMAGE_PIXELS="$JUDGE_MAX_IMAGE_PIXELS"; fi
results_root="${RESULTS_ROOT:-$PWD/rag_eval/results/opd4b_loss_retrieval_7datasets}"
scores_root="${SCORES_ROOT:-$PWD/rag_eval/evaluation/opd4b_loss_retrieval_7datasets}"
unset MODELS_FILE BASE_MODEL ADAPTER_PATH RUN_NAME MODEL_NAME OUTPUT_DIR EVAL_OUTPUT_ROOT

# ===== 4B OPD / EJSD + CLIP / Stage2 =====
RUN_NAME=opd_s4b_t8b_stage2_entropy_jsd \
MODEL_NAME=opd_s4b_t8b_stage2_entropy_jsd \
RETRIEVAL_FILE=test_mixed_top10_clip.jsonl \
OUTPUT_DIR="$results_root/opd_s4b_t8b_stage2_entropy_jsd_retrieval_clip" \
EVAL_OUTPUT_ROOT="$scores_root/opd_s4b_t8b_stage2_entropy_jsd_retrieval_clip/mssbench_protocol" \
bash scripts/eval_mixed_top10_v2.sh

# ===== 4B OPD / EJSD + Emb.-2B / Stage2 =====
RUN_NAME=opd_s4b_t8b_stage2_entropy_jsd \
MODEL_NAME=opd_s4b_t8b_stage2_entropy_jsd \
RETRIEVAL_FILE=test_mixed_top10_2b.jsonl \
OUTPUT_DIR="$results_root/opd_s4b_t8b_stage2_entropy_jsd_retrieval_2b" \
EVAL_OUTPUT_ROOT="$scores_root/opd_s4b_t8b_stage2_entropy_jsd_retrieval_2b/mssbench_protocol" \
bash scripts/eval_mixed_top10_v2.sh
