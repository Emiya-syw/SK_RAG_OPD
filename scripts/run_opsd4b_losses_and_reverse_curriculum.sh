#!/usr/bin/env bash
# 前八段：4B OPSD 的 RKL/FKL/EKL/JSD 两阶段训练。
# 接着四段：2B/4B OPSD/OPD 从已有 controlled 第一阶段继续训练 consistent。
# 最后两段：4B OPSD/OPD 使用 consistent+controlled 混合数据，从基础学生开始训练。
# 所有命令依次训练，不生成、不评测；第二阶段自动导出并加载对应第一阶段 adapter。
# RKL=k3（单样本估计器），EOPD/EKL=entropy_kl。出错立即停止。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 使用当前环境，也可显式指定 PYTHON_BIN。
export PYTHON_BIN="${PYTHON_BIN:-python}"
# 本脚本按下列命令顺序执行，不再接受 losses/reverse/mixed 分组参数。
if (( $# > 0 )); then echo "Usage: bash $0 (no arguments)" >&2; exit 2; fi

# ===== 4B OPSD / RKL / 第1阶段：consistent，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_stage1_consistent_k3 \
DISTILLATION_LOSS_MODE=k3 \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / RKL / 第2阶段：controlled，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="opsd_s4b_stage1_consistent_k3" \
EXPERIMENT_NAME=opsd_s4b_stage2_controlled_k3 \
DISTILLATION_LOSS_MODE=k3 \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / FKL / 第1阶段：consistent，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_stage1_consistent_forward_kl \
DISTILLATION_LOSS_MODE=forward_kl \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / FKL / 第2阶段：controlled，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="opsd_s4b_stage1_consistent_forward_kl" \
EXPERIMENT_NAME=opsd_s4b_stage2_controlled_forward_kl \
DISTILLATION_LOSS_MODE=forward_kl \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / EKL (EOPD) / 第1阶段：consistent，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_stage1_consistent_entropy_kl \
DISTILLATION_LOSS_MODE=entropy_kl \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / EKL (EOPD) / 第2阶段：controlled，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="opsd_s4b_stage1_consistent_entropy_kl" \
EXPERIMENT_NAME=opsd_s4b_stage2_controlled_entropy_kl \
DISTILLATION_LOSS_MODE=entropy_kl \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / JSD / 第1阶段：consistent，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_stage1_consistent_jsd \
DISTILLATION_LOSS_MODE=jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / JSD / 第2阶段：controlled，基础 4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="opsd_s4b_stage1_consistent_jsd" \
EXPERIMENT_NAME=opsd_s4b_stage2_controlled_jsd \
DISTILLATION_LOSS_MODE=jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / 反向课程第2阶段：已有 controlled → consistent =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="opsd_s4b_stage1_controlled_entropy_jsd" \
EXPERIMENT_NAME=opsd_s4b_stage2_consistent_from_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / 反向课程第2阶段：已有 controlled → consistent =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="opd_s4b_t8b_stage1_controlled_entropy_jsd" \
EXPERIMENT_NAME=opd_s4b_t8b_stage2_consistent_from_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD / 混合数据：consistent + controlled =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_mixed_consistent_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / 混合数据：consistent + controlled =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_mixed_consistent_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
START_STAGE=train \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh
