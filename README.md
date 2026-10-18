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

三个训练入口（`train_consistent.sh`、`train_controlled.sh`、`train_embedding.sh`）
均支持通过 `PROMPT_MODE=guided|basic` 选择提示词，默认 `guided`。
`guided` 使用详细的案例对比与回答指导；`basic` 与 RAG 测试共用以下指令：

```text
You are a helpful assistant. You may refer to the retrieved examples when answering the current question.
```

例如，使用合并后的数据和 basic 提示词训练：

```bash
PROMPT_MODE=basic bash scripts/train_consistent.sh \
  data/category_consistent_abstract_merged.jsonl outputs/consistent_basic
```

两个模式都保留当前问题、图片和检索案例，选择记录在转换后数据的
`extra_info.prompt_mode` 中。提示词在数据转换时写入；单独转换可使用
`python scripts/prepare_verl_data.py INPUT OUTPUT --prompt-mode basic`。
已有 veRL 数据不会因设置环境变量自动改变提示词，需要重新转换。

也可直接训练已转换的数据（使用数据中已保存的提示词）：

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
DISTILLATION_LOSS_MODE=forward_kl_topk DISTILLATION_TOPK=16 \
  bash scripts/train_consistent.sh
```

新增四种分布损失（默认仍为 `k3`）：

| `DISTILLATION_LOSS_MODE` | 逐 token 损失 |
| --- | --- |
| `forward_kl` | EOPD 式教师 top-k 前向 KL 近似：教师 top-k 内重归一化，学生保留完整词表概率 |
| `jsd` | `β KL(T‖M) + (1−β) KL(S‖M)`，默认 `β=0.5`，即标准对称 JSD |
| `entropy_kl` | 保留系数为 1 的采样 PPO 式反向 KL 项，另加 `α` 乘上述前向 KL 近似；`α` 随教师熵、逐词覆盖缺口和教师 top-k 质量在 0–1 之间变化 |
| `entropy_jsd` | 同上广义 JSD 公式，但逐 token 设置 `β=1−α` |

这里 `M=βT+(1−β)S`。`entropy_kl` 和 `entropy_jsd` 使用师生各自前 `ENTROPY_TOPK` 个 token 的原始概率与一个尾部概率桶近似熵，熵采用自然对数，单位为 nat。两种模式共用基础信号：在教师损失 top-k 集合 `K` 上计算 `C=Σmin(T(x),S(x))/ΣT(x)`，`D=clamp((1−C)/ENTROPY_KL_COVERAGE_DEFICIT_SCALE,0,1)`；教师 top-k 质量置信度为 `q=clamp((ΣT(K)−ENTROPY_KL_TEACHER_MASS_FLOOR)/(1−ENTROPY_KL_TEACHER_MASS_FLOOR),0,1)`；`a=sigmoid((H(T)−ENTROPY_KL_TEACHER_ENTROPY_THRESHOLD)/τ)·D`。`entropy_kl` 的附加前向 KL 系数为 `q·a`，反向 PPO 项系数固定为 1；`entropy_jsd` 的学生混合系数为 `α=0.5+q·(a−0.5)`，教师系数 `β=1−α`。教师 top-k 质量低至 `q=0` 时，`entropy_jsd` 退回 `β=0.5`，仍保留对称 JSD 信号。教师分布与门控权重停止梯度，学生在 M 中的贡献保留梯度。旧熵差门控公式见 [2026-10-18 快照](reports/entropy_jsd_coefficient_snapshot_20261018.md)。

```bash
# 将模式替换为 forward_kl / jsd / entropy_kl / entropy_jsd
DISTILLATION_LOSS_MODE=entropy_jsd DISTILLATION_TOPK=16 ENTROPY_TOPK=64 \
  USE_POLICY_GRADIENT=false ENTROPY_GATE_TEMPERATURE=1.0 \
  ENTROPY_KL_TEACHER_ENTROPY_THRESHOLD=0.8 \
  ENTROPY_KL_COVERAGE_DEFICIT_SCALE=0.3 ENTROPY_KL_TEACHER_MASS_FLOOR=0.95 \
  ENTROPY_JSD_NORMALIZE=false \
  bash scripts/train_consistent.sh

# 标准 JSD；改变 JSD_BETA 可使用固定权重的广义 JSD
DISTILLATION_LOSS_MODE=jsd JSD_BETA=0.5 USE_POLICY_GRADIENT=false \
  bash scripts/run_opd.sh data/verl/train.parquet outputs/opd_jsd
```

`forward_kl` 和 `entropy_kl` 的前向项按 EOPD 论文的 top-k 公式计算，默认 `DISTILLATION_TOPK=16`：教师概率仅在 top-k 内重归一化，学生使用完整 softmax 对这些 token 的原始概率，因此仍惩罚学生分配给 top-k 外的概率。熵估计独立使用 `ENTROPY_TOPK=64`：教师一次返回两种用途所需的较大 top-k，损失仅截取前 `DISTILLATION_TOPK` 项；学生从完整 logits 提取自己的前 `ENTROPY_TOPK` 项。它是带尾部桶的近似熵，不是精确全词表熵。`jsd` 和 `entropy_jsd` 在损失 top-k 集合上使用双方的原始概率，并将集合外各自的剩余质量合为一个尾部桶；它能惩罚概率质量整体流出集合，但无法区分尾部各 token。熵估计的 top-k 尾桶与 JSD 损失的 top-k 尾桶使用不同的 k。两种熵模式的逐词覆盖率使用损失 top-k 上双方完整 softmax 的原始概率，不在 top-k 内重新归一化；`entropy_kl` 反向项使用学生实际采样词在师生模型下的 logprob，以 `log T−log S_old` 为 PPO 式优势。即使采样词不在教师 top-k 中，也会单独取得其教师概率。`student_mass` 和 `teacher_mass` 记录损失 top-k 内的原始概率质量。日志包含师生熵、尾部质量、逐词覆盖率、门控置信度以及 `forward_weight` 或 `jsd_beta`，只统计 response mask 内位置。师生须使用一致的词表及 token ID。

默认的 `k3` 反向 KL 路径只使用 rollout 实际采样 token 的师生 logprob，是单样本估计，并未请求完整词表概率。四种分布损失若要改成精确的完整词表版本，需要 teacher 返回每个回答位置的全量 logits；当前 vLLM teacher 接口只提供 prompt logprob 的 top-k 输出。

新模式使用本仓库 `verl-runtime` 的 FSDP eager logits 路径，要求 `USE_POLICY_GRADIENT=false`，不支持 Megatron 或 fused loss kernels。对 `entropy_kl`，此开关关闭通用蒸馏策略梯度路径，但其反向分量会在内部单独使用 EOPD 式标准 PPO clipping，不使用通用 `vanilla` PPO 的 dual-clip。`JSD_BETA` 默认 0.5 且须在 `(0,1)` 内；`ENTROPY_GATE_TEMPERATURE` 默认 1.0 且须大于 0。两种熵模式共用教师熵阈值、覆盖缺口尺度和教师 top-k 质量下限，默认分别为 0.8、0.3、0.95。`ENTROPY_GATE_EPSILON` 与 `ENTROPY_GATE_TAIL_CUTOFF` 仅保留旧配置兼容性，不再参与系数计算。`ENTROPY_JSD_NORMALIZE=true` 仅对 `entropy_jsd` 额外除以 `β(1−β)`，会改变损失尺度，默认关闭；`β=1` 时损失为 0，归一化分母以 `1e-12` 保底。新模式不使用旧的 `LOG_PROB_MIN_CLAMP`；JSD 对概率取对数时采用 `1e-12` 下限，前向 KL 的学生原始概率不采用该下限。`LOSS_MAX_CLAMP` 仍限制直接反传的分布项（默认 10，设为 `null` 可关闭）；`entropy_kl` 的反向项使用 PPO 概率比 clipping，不应用这个损失截断。对应 Hydra 配置均位于 `distillation.distillation_loss` 下，名称为 `topk`、`entropy_topk`、`jsd_beta`、`entropy_gate_temperature`、`entropy_kl_teacher_entropy_threshold`、`entropy_kl_coverage_deficit_scale`、`entropy_kl_teacher_mass_floor`、`entropy_jsd_normalize`。

`train_eval_pipeline.sh` 的 2B 自动预设在四种分布损失下将学生反传 token 预算从 65536 降至 24576，并默认启用 `USE_CHUNKED_TOPK=true`、`CHUNKED_TOPK_CHUNK_SIZE=512`，以降低词表 logits 相关的峰值显存；显式环境变量仍优先。遇到 OOM 可进一步减小 `MAX_PROMPT_LENGTH` 与 `MAX_TOKEN_LEN_PER_GPU`，同时确保预算至少覆盖最大 prompt＋response 长度。`TRAIN_BATCH_SIZE` 控制全局样本数，不直接替代每卡的反传 token 预算。

训练默认使用 `TRAINER_LOGGER=['console','file']`，将每步所有 veRL 指标保存到 `outputs/<EXPERIMENT_NAME>/training_metrics.jsonl`；训练结束（包括报错退出）时自动生成 `training_curves.svg` 和对应的 `training_curves.csv`。曲线包括蒸馏损失、师生熵、混合权重、top-k 概率质量、回答长度、梯度范数、rollout 一致性和耗时；JSONL 保留全部指标。可在训练中读取 JSONL，或随时运行 `python3 scripts/plot_train_metrics.py outputs/<EXPERIMENT_NAME>/training_metrics.jsonl` 重画。恢复训练时指标文件追加写入，并以最新的同一步记录绘图；全新训练则覆盖旧指标。用 `TRAIN_METRICS_FILE`、`TRAIN_METRICS_PLOT_FILE` 更改路径，用 `PLOT_TRAIN_METRICS=false` 关闭自动绘图。显式覆盖 `TRAINER_LOGGER` 时若不包含 `file`，则不会保存本地指标。

Rollout-mixture distillation 可按比例把 on-policy rollout 替换为数据中的 golden/off-policy response，降低训练分布被退化长 rollout 主导的风险。`prepare_verl_data.py` 会按 Instruct 输出格式把 `teacher_demonstration` 和 `reference_answer/response` 以普通 assistant 文本拼接后写入 `extra_info.golden_response`，不会加入 `<think>` 标签。该模式保持默认的 thinking-off 配置，并关闭 teacher-only privileged prompt，保证 teacher/student 评分提示词一致：

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
