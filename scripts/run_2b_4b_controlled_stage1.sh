#!/usr/bin/env bash
# 前四段：controlled 第一阶段；接着四段：基础 8B 教师两阶段 OPD；最后六段：4B 三种损失两阶段 OPD。
# 所有命令依次训练，不生成、不评测；第二阶段启动时自动导出并加载对应第一阶段 adapter。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ===== 2B OPSD：2B 学生，2B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s2b_stage1_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 2B OPD：2B 学生，8B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s2b_t8b_stage1_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPSD：4B 学生，4B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opsd_s4b_stage1_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD：4B 学生，8B 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_stage1_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 2B OPD 第1阶段：consistent 数据，基础 8B Thinking 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s2b_t8b_base_stage1_consistent_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 2B OPD 第2阶段：controlled 数据，基础 8B Thinking 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-2B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking \
STAGE1_EXPERIMENT_NAME="opd_s2b_t8b_base_stage1_consistent_entropy_jsd" \
EXPERIMENT_NAME=opd_s2b_t8b_base_stage2_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD 第1阶段：consistent 数据，基础 8B Thinking 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_base_stage1_consistent_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD 第2阶段：controlled 数据，基础 8B Thinking 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-8B-Thinking \
STAGE1_EXPERIMENT_NAME="opd_s4b_t8b_base_stage1_consistent_entropy_jsd" \
EXPERIMENT_NAME=opd_s4b_t8b_base_stage2_controlled_entropy_jsd \
DISTILLATION_LOSS_MODE=entropy_jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / FKL / 第1阶段：consistent，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage1_consistent_forward_kl \
DISTILLATION_LOSS_MODE=forward_kl \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / FKL / 第2阶段：controlled，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="opd_s4b_t8b_merged_stage1_consistent_forward_kl" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage2_controlled_forward_kl \
DISTILLATION_LOSS_MODE=forward_kl \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / JSD / 第1阶段：consistent，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage1_consistent_jsd \
DISTILLATION_LOSS_MODE=jsd \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / JSD / 第2阶段：controlled，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="opd_s4b_t8b_merged_stage1_consistent_jsd" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage2_controlled_jsd \
DISTILLATION_LOSS_MODE=jsd \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / EKL / 第1阶段：consistent，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage1_consistent_entropy_kl \
DISTILLATION_LOSS_MODE=entropy_kl \
TRAIN_DATA_FILE=data/category_consistent_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=5e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh

# ===== 4B OPD / EKL / 第2阶段：controlled，8B stage1 merged 教师 =====
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/Qwen3-VL-4B-Thinking \
TEACHER_MODEL_PATH=/home/sunyw/SK_RAG_OPD/models/opsd_s8b_stage1_entropy_jsd_merged \
STAGE1_EXPERIMENT_NAME="opd_s4b_t8b_merged_stage1_consistent_entropy_kl" \
EXPERIMENT_NAME=opd_s4b_t8b_merged_stage2_controlled_entropy_kl \
DISTILLATION_LOSS_MODE=entropy_kl \
TRAIN_DATA_FILE=data/category_controlled_abstract.jsonl \
PROMPT_MODE=guided \
USE_POLICY_GRADIENT=false \
LR=1e-6 \
SAVE_FREQ=100 \
TEACHER_USE_PRIVILEGED_INFO=True \
TRAIN_ONLY=true \
bash scripts/train_eval_pipeline.sh
