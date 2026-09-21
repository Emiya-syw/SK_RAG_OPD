# SK-RAG-OPD on veRL

本项目使用 veRL 原生 On-Policy Distillation（OPD）训练链路训练检索增强视觉语言模型。student 通过 vLLM 对当前图像、问题和检索样例做 on-policy rollout；独立 teacher server 对相同 rollout token 评分；veRL 的 Ray trainer 调度 rollout、teacher 与 FSDP actor，并只在 response mask 上计算蒸馏损失。

## 架构

```text
SK-RAG JSONL + images
        │  scripts/prepare_verl_data.py
        ▼
veRL RLHFDataset (prompt / images / data_source / extra_info)
        │
        ├── vLLM student rollout ──► sampled response
        ├── vLLM teacher ─────────► teacher token log-probs
        └── FSDP actor ───────────► OPD loss / optimizer / checkpoint
```

训练入口不再创建 Hugging Face `Trainer` 或在每个训练 step 内调用 `generate()`。模型分片、Ray worker、rollout 权重同步、teacher 推理和 checkpoint 均由 veRL 管理。

## 依赖

代码针对 veRL commit `10db40d0da4d59150bb389960b77585f81a89b8d` 的原生多模态 OPD API。建议在 veRL 官方支持的 CUDA 环境安装：

```bash
python -m pip install -r requirements.txt
```

student rollout/FSDP 和 teacher 使用独立 Ray resource pool。默认配置需要 2 张 GPU：1 张用于 student actor/rollout，1 张用于 teacher。更多 GPU 可设置 `TRAINER_GPUS_PER_NODE`、`TEACHER_GPUS_PER_NODE`、`ROLLOUT_TP` 和 `TEACHER_TP`。

## 数据准备

源 JSONL 每行至少包含 `image_path`、`question` 和 `reference_answer`（也兼容 `response`/`answer`）。`retrieved_examples` 中的问题、回答和图片会依次加入 student prompt。转换器保证 `<image>` 占位符数量与 `images` 数量严格一致：

```bash
python scripts/prepare_verl_data.py \
  data/category_consistent.jsonl \
  data/verl/category_consistent.parquet
```

也可以输出 `.jsonl`；当前 veRL 的 `RLHFDataset` 同时支持 JSONL 和 Parquet。默认会检查全部图片是否存在。只验证数据结构时可以加 `--skip-image-check`。

转换后的关键字段如下：

```json
{
  "data_source": "sk_rag_opd",
  "prompt": [{"role": "user", "content": "Current Image\n\n<image>..."}],
  "images": ["/absolute/current.jpg", "/absolute/retrieved.jpg"],
  "ability": "vision_safety",
  "reward_model": {"style": "model", "ground_truth": "..."},
  "extra_info": {"index": 0, "id": "..."}
}
```

## 训练

所有当前配方使用的训练参数都集中在 `scripts/run_opd.sh` 顶部的“训练参数配置区”。
每个参数旁边标明了默认值、可选值或范围以及对训练的作用。可以直接修改脚本默认值，
也可以用同名环境变量临时覆盖；命令行末尾仍可追加任意高级 veRL/Hydra override。
启动 veRL 前会打印最终生效配置，并默认保存为当前阶段输出目录中的
`training_config.txt`。可通过 `PRINT_TRAIN_CONFIG=false` 关闭终端打印、
`SAVE_TRAIN_CONFIG=false` 关闭保存，或用 `TRAIN_CONFIG_FILE` 更改保存路径。

单阶段训练会先转换原始数据，然后启动 veRL：

```bash
bash scripts/train_consistent.sh
bash scripts/train_controlled.sh
bash scripts/train_embedding.sh
```

`train_consistent.sh` 在交互式终端中启动时会询问权重保存目录名，例如输入
`consistent_full` 后，checkpoint 和 `training_config.txt` 会保存到
`outputs/consistent_full/`。直接回车则使用 `YYYY-MM-DD_HH-MM` 作为目录名。
脚本参数 `$2` 仍可用于非交互式运行时显式指定输出目录：

```bash
bash scripts/train_consistent.sh data/category_consistent.jsonl outputs/consistent_full
```

也可直接训练已转换的数据：

```bash
bash scripts/run_opd.sh data/verl/train.parquet outputs/opd
```

默认 student/teacher 分别为：

```text
Qwen/Qwen3-VL-2B-Thinking
Qwen/Qwen3-VL-8B-Thinking
```

训练和测试默认加载 Thinking checkpoint，同时使用
`opd_rag/chat_templates/qwen3_vl_thinking_off.jinja` 在生成前写入空的
`<think></think>` 块，使模型直接生成最终回答。Qwen3-VL-Thinking 自带模板会
忽略 `enable_thinking=false`，因此代码不能只依赖该参数。若要临时恢复原生
Thinking 模式，可在命令前设置 `ENABLE_THINKING=true`。

常用覆盖参数：

```bash
MODEL_PATH=/path/to/student \
TEACHER_MODEL_PATH=/path/to/teacher \
ENABLE_THINKING=false \
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
MAX_PROMPT_LENGTH=8192 \
MAX_RESPONSE_LENGTH=2048 \
MAX_IMAGE_PIXELS=100352 \
TEACHER_ENABLE_PREFIX_CACHING=false \
bash scripts/train_consistent.sh
```

默认情况下学生和教师共享同一条指令提示词。若要将源 JSONL 中的
`teacher_demonstration` 作为教师专享的特权信息追加到教师评分提示词，可开启：

```bash
TEACHER_USE_PRIVILEGED_INFO=true \
  bash scripts/train_consistent.sh
```

关闭（默认）或样本没有 `teacher_demonstration` 时，行为与原实现一致；学生提示词
始终不会包含该字段。教师侧追加的 token 会在返回蒸馏张量前移除，以保持与学生
response 的位置对齐。追加内容会明确标注为仅供参考的思考指导，要求教师独立分析、
不要照抄其中的思考或答案，并继续直接完整地回答当前用户问题。

OPD teacher 依赖 vLLM 为完整的 student 序列返回逐 token
`prompt_logprobs`，因此脚本默认关闭 teacher 的 prefix cache。多图训练默认把
单图限制为 100352 像素，防止所有图片累积后的视觉 token 超过 prompt 上限。
固定的 veRL 版本还需要应用 Qwen3-VL teacher 响应后缀对齐补丁；完整安装脚本
会自动应用。已有环境可直接执行 `bash scripts/patch_verl_runtime.sh`，无需重新
安装 Python/CUDA 依赖。

额外的 veRL/Hydra override 可附加到 `run_opd.sh`：

```bash
bash scripts/run_opd.sh data/verl/train.parquet outputs/opd \
  trainer.total_training_steps=10 actor_rollout_ref.actor.optim.lr=1e-6
```

默认蒸馏为 `loss_mode=k3`、`use_policy_gradient=False`、`use_task_rewards=False`，即直接反向传播 veRL 的单样本 reverse-KL estimator。可切换为 PG-OPD：

由于 veRL 仍会为 rollout 生成 `rm_scores`，纯蒸馏配置使用 `opd_rag/zero_reward.py` 返回零 task reward；该值不会进入最终 loss。

```bash
DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True \
  bash scripts/train_consistent.sh
```

或使用 teacher top-k forward KL：

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk DISTILLATION_TOPK=128 \
  bash scripts/train_consistent.sh
```

Rollout-mixture distillation 可按比例把 on-policy rollout 替换为数据中的 golden/off-policy response，降低训练分布被退化长 rollout 主导的风险。`prepare_verl_data.py` 会把 `teacher_demonstration` 包进 `<think>...</think>`，并紧接 `reference_answer/response` 写入 `extra_info.golden_response`。该模式会强制开启 thinking，并关闭 teacher-only privileged prompt，保证 teacher/student 评分提示词一致：

```bash
DISTILLATION_LOSS_MODE=rollout_mixture_k3 ROLLOUT_MIXTURE_OFF_POLICY_RATIO=0.3 \
USE_POLICY_GRADIENT=false \
  bash scripts/train_consistent.sh data/category_consistent.jsonl outputs/consistent_mix30
```

建议先配合默认的 direct distillation 使用；`USE_POLICY_GRADIENT=true` 时 golden response 没有真实 rollout logprob，不建议作为首选配置。

## 多阶段训练

连续训练会用 `verl.model_merger` 从上一阶段的最新 FSDP checkpoint 导出 PEFT adapter，再作为下一阶段的 `lora_adapter_path`：

```bash
bash scripts/train_consistent_then_controlled.sh
bash scripts/train_consistent_then_controlled_then_embedding.sh
```

已有 adapter 可跳过 consistent 阶段：

```bash
CONSISTENT_INIT_PATH=/path/to/adapter \
  bash scripts/train_consistent_then_controlled_then_embedding.sh
```

该目录必须包含 `adapter_config.json` 和 adapter 权重。

## 验证

CPU 环境可验证数据转换和现有评测工具：

```bash
python -m pytest -q
bash -n scripts/*.sh
```

完整训练 smoke test 需要安装 `requirements-verl.txt`、可访问 student/teacher 权重，并提供至少两个可用 GPU。

## 目录

```text
opd_rag/verl_data.py             veRL 多模态数据适配
scripts/prepare_verl_data.py     JSONL -> veRL JSONL/Parquet
scripts/run_opd.sh               veRL OPD/Hydra 启动参数
scripts/train_*.sh               单阶段与连续训练
rag_eval/                        RAG 和安全对齐评测
tests/                           CPU 测试
```
