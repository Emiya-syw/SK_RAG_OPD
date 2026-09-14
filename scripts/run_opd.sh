#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
project_root="$(pwd)"

train_file="${1:?usage: scripts/run_opd.sh VERL_DATASET [OUTPUT_DIR] [HYDRA_OVERRIDES...]}"
output_dir="${2:-outputs/opd}"
if (( $# >= 2 )); then shift 2; else shift 1; fi

# =============================================================================
# 训练参数配置区
# 直接修改每行最后的默认值即可；也可用同名环境变量临时覆盖。
# 例：LR=1e-6 EPOCHS=3 bash scripts/train_consistent.sh
# =============================================================================

# --- 运行环境与模型 -----------------------------------------------------------
# Python 可执行文件。可选：python3、conda 环境中的绝对路径。
PYTHON_BIN="${PYTHON_BIN:-python3}"
# student 权重。可选：Hugging Face model ID 或本地模型目录。
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Thinking}"
# teacher 权重。可选：与 student 同 tokenizer 词表的 HF ID 或本地目录；越强通常蒸馏信号越好、显存越高。
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-VL-8B-Thinking}"
# 已导出的 PEFT adapter，用于接续阶段训练。可选：空字符串或含 adapter_config.json 的目录。
LORA_INIT_PATH="${LORA_INIT_PATH:-}"
# 是否启用 Qwen Thinking。可选：false（空 think 后直接回答）、true（使用模型原生思考模板）。
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# Thinking 关闭模板。仅 ENABLE_THINKING=false 时使用；可换成兼容的 Jinja 模板文件。
THINKING_OFF_TEMPLATE="${THINKING_OFF_TEMPLATE:-${project_root}/opd_rag/chat_templates/qwen3_vl_thinking_off.jinja}"
# 是否允许模型仓库自定义代码。可选：true/false；Qwen 自定义实现通常设 true。
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
# student FSDP 注意力实现。可选：sdpa（无需额外依赖，推荐）、flash_attention_2（需安装 flash-attn）、eager（最慢）。
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# --- 数据与长度 ---------------------------------------------------------------
# 验证集。可选：空字符串（复用训练集）或 veRL JSONL/Parquet 路径。
VAL_FILE="${VAL_FILE:-}"
# 一次 rollout 的全局 prompt 数。正整数；增大会提高吞吐并增加显存/等待时间。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# prompt 最大 token 数。超过后由 FILTER_OVERLONG_PROMPTS/TRUNCATION 处理。
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
# response 最大 token 数。正整数；越大越耗 rollout 显存和 teacher 评分时间。MAX_NEW_TOKENS 是兼容别名。
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${MAX_NEW_TOKENS:-2048}}"
# vLLM 总上下文长度。默认 prompt + response + 1；可设更大，但显存占用会上升。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}"
# 单张图像最大像素数。正整数；降低可减少视觉 token 和显存，可能损失细节。
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
# 是否过滤过长样本。可选：false（推荐，避免在 Ray worker 内嵌套多进程）、true（训练前逐图计算并过滤）。
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-false}"
# 过滤 prompt 长度时使用的 CPU 进程数。正整数；多模态数据建议 4-8，过大会增加内存占用。
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}"
# 超长后的截断策略。可选：error、left、right、middle；推荐配合过滤使用 error，避免静默截断图像上下文。
TRUNCATION="${TRUNCATION:-error}"
# 数据中的图像字段名。应与 prepare_verl_data.py 输出一致。
IMAGE_KEY="${IMAGE_KEY:-images}"
# 是否返回多模态 processor 输入。视觉模型必须为 true。
RETURN_MULTI_MODAL_INPUTS="${RETURN_MULTI_MODAL_INPUTS:-true}"
# 是否打乱训练数据。可选：true/false。
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
# 数据打乱与 rollout 随机种子。任意非负整数；固定值便于复现。
SEED="${SEED:-42}"

# --- LoRA 与 student 模型 -----------------------------------------------------
# LoRA rank。可选：正整数，0 表示全参数训练；越大容量和显存占用越高。
LORA_RANK="${LORA_RANK:-64}"
# LoRA alpha。正数；与 rank 的比值影响 LoRA 更新缩放。
LORA_ALPHA="${LORA_ALPHA:-128}"
# LoRA 目标层。常用：all-linear，或 veRL/PEFT 支持的模块列表表达式。
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
# 是否移除 padding 后计算。可选：true/false；true 通常节省长短不齐 batch 的算力和显存。
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
# 梯度检查点。可选：true/false；true 节省显存但增加反向计算时间。
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
# 是否冻结视觉编码器。可选：false（训练视觉层）、true（只训练语言侧/LoRA，显存更低）。
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"

# --- 优化器与 actor 更新 ------------------------------------------------------
# AdamW 学习率。正浮点数；LoRA 常用约 1e-6 到 1e-4，本项目默认 5e-6。
LR="${LR:-5e-6}"
# 权重衰减。非负浮点数；0 关闭，增大可加强正则化。
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
# warmup 占总训练步数比例。范围 0~1。
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"
# 学习率调度器。常用：constant、linear、cosine。
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
# 梯度范数裁剪阈值。正数；缓解梯度爆炸。
CLIP_GRAD="${CLIP_GRAD:-1.0}"
# 每次 actor 更新使用的全局 mini-batch。正整数，通常不大于 TRAIN_BATCH_SIZE。
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
# 每张 GPU 的 micro-batch。正整数；OOM 时优先减小。
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
# 动态按 token 组 batch。可选：true/false；true 更充分利用 MAX_TOKEN_LEN_PER_GPU。
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
# 每张 GPU 每个 actor micro-batch 的最大 token 总数；减小可缓解 OOM。
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
# 同一批 rollout 重复 actor 更新次数。正整数；增大数据利用率，也增加过拟合/离策略程度。
PPO_EPOCHS="${PPO_EPOCHS:-1}"
# loss 聚合。常用：token-mean、seq-mean-token-sum、seq-mean-token-mean；影响长短回答权重。
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"

# --- FSDP 与显存 --------------------------------------------------------------
# 参数 CPU offload。可选：true/false；true 降显存但增加 CPU 内存和 PCIe 开销。
PARAM_OFFLOAD="${PARAM_OFFLOAD:-true}"
# 优化器 CPU offload。可选：true/false；true 显著降显存但训练更慢。
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"
# FSDP 前向后重新分片。可选：true/false；true 更省显存，false 可能更快。
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-true}"
# 是否使用 torch.compile。可选：true/false；true 可能提速，遇到编译兼容问题可关闭。
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"

# --- student rollout / vLLM --------------------------------------------------
# rollout 后端。当前配方可选：vllm；安装并适配后也可尝试 sglang。
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
# vLLM 注意力后端。A100/4090 推荐 FLASH_ATTN；可选：FLASH_ATTN、FLASHINFER、TRITON_ATTN、FLEX_ATTENTION。
# 该值通过 rollout/teacher 的 engine_kwargs 传给 vLLM，避免自动加载未完整安装的可选 FlashInfer cubin 包。
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
# 权重加载格式。常用：safetensors、auto；safetensors 与当前 Qwen 权重匹配。
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
# rollout tensor parallel 数。正整数且不能超过 student GPU 数；模型放不下时增大。
ROLLOUT_TP="${ROLLOUT_TP:-1}"
# vLLM 可使用的单卡显存比例。范围 0~1；OOM 时降低，KV cache 不足时提高。
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
# 采样温度。0 为近似贪心，>0 启用随机性；OPD 需要 on-policy 多样性时可提高。
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
# nucleus sampling。范围 0~1；越小输出越保守。
GENERATION_TOP_P="${GENERATION_TOP_P:-0.95}"
# top-k sampling。-1 关闭；正整数只从概率最高的 k 个 token 采样。
GENERATION_TOP_K="${GENERATION_TOP_K:--1}"
# 每个 prompt 生成的 rollout 数。正整数；增大样本数会线性增加生成与 teacher 评分开销。
ROLLOUT_N="${ROLLOUT_N:-1}"
# 是否忽略 EOS 并生成到最大长度。可选：false/true；通常保持 false。
ROLLOUT_IGNORE_EOS="${ROLLOUT_IGNORE_EOS:-false}"
# 强制 eager 执行。可选：false/true；true 更易调试，通常更慢。
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-false}"
# vLLM 单次调度最大 token 数。正整数；提高可能增吞吐，也会增显存。
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
# vLLM 同时处理的最大序列数。正整数；提高并发也会增加 KV cache 压力。
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
# chunked prefill。可选：true/false；长 prompt 通常开启更稳。
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}"
# prefix cache。可选：true/false；重复前缀多时提速，但占用缓存。
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-true}"

# --- OPD teacher 与蒸馏损失 --------------------------------------------------
# 是否启用蒸馏。当前脚本必须为 true；false 会退化为无有效 reward 的训练。
DISTILLATION_ENABLED="${DISTILLATION_ENABLED:-true}"
# loss 类型。可选：k3（推荐直接反传）、forward_kl_topk（更完整但更耗算力）、k1（仅配 PG），以及 kl/abs/mse/k2/low_var_kl。
DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k3}"
# teacher/student top-k 数。仅 forward_kl_topk 使用；越大越接近完整分布、越耗显存和带宽。
DISTILLATION_TOPK="${DISTILLATION_TOPK:-128}"
# 蒸馏 loss 权重。非负浮点数；与 task reward 联用时控制蒸馏占比。
DISTILLATION_LOSS_COEF="${DISTILLATION_LOSS_COEF:-1.0}"
# 是否叠加任务 reward。可选：false/true；本项目默认 zero_reward，开启前应替换 reward 函数。
USE_TASK_REWARDS="${USE_TASK_REWARDS:-false}"
# 是否把蒸馏差异当作 policy-gradient reward。false 为直接反传（配 k3/top-k）；true 应配 k1。
USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-false}"
# 蒸馏 loss 上界裁剪。正数或 null；降低可抑制异常大 loss。
LOSS_MAX_CLAMP="${LOSS_MAX_CLAMP:-10.0}"
# log-prob 下界裁剪。负数或 null；用于数值稳定。
LOG_PROB_MIN_CLAMP="${LOG_PROB_MIN_CLAMP:--10.0}"
# top-k 分块计算。可选：false/true；仅长上下文 forward_kl_topk OOM 时开启，速度会下降。
USE_CHUNKED_TOPK="${USE_CHUNKED_TOPK:-false}"
# top-k 的 token 分块大小。正整数；越小峰值显存越低、kernel 调用越多。
CHUNKED_TOPK_CHUNK_SIZE="${CHUNKED_TOPK_CHUNK_SIZE:-4096}"
# PG policy loss。目前 veRL OPD 仅支持 vanilla。
DISTILLATION_POLICY_LOSS_MODE="${DISTILLATION_POLICY_LOSS_MODE:-vanilla}"
# PG PPO clip。仅 USE_POLICY_GRADIENT=true 时生效；常用 0.1~0.3。
DISTILLATION_CLIP_RATIO="${DISTILLATION_CLIP_RATIO:-0.2}"
# teacher tensor parallel 数。正整数，且必须整除 TEACHER_GPUS_PER_NODE。
TEACHER_TP="${TEACHER_TP:-1}"
# teacher vLLM 显存比例。范围 0~1。
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.8}"
# teacher 最大并发序列数。正整数；降低可减少显存峰值。
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-1024}"
# teacher 权重加载格式。常用：auto、safetensors。
TEACHER_LOAD_FORMAT="${TEACHER_LOAD_FORMAT:-auto}"
# teacher vLLM eager 模式。可选：true/false；true 更稳但可能更慢。
TEACHER_ENFORCE_EAGER="${TEACHER_ENFORCE_EAGER:-true}"
# teacher 副本数。空字符串表示 TEACHER_GPUS_PER_NODE / TEACHER_TP；可手动设正整数。
TEACHER_NUM_REPLICAS="${TEACHER_NUM_REPLICAS:-}"

# --- 分布式、日志、保存与恢复 -------------------------------------------------
# 每节点 student actor/rollout GPU 数。正整数。
TRAINER_GPUS_PER_NODE="${TRAINER_GPUS_PER_NODE:-1}"
# 每节点 teacher GPU 数。正整数；默认总计需要 student 1 卡 + teacher 1 卡。
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-1}"
# student 节点数。正整数。
NNODES="${NNODES:-1}"
# teacher 节点数。正整数；一般与 NNODES 一致。
TEACHER_NNODES="${TEACHER_NNODES:-${NNODES}}"
# 按 token 平衡 data-parallel batch。可选：true/false；变长序列推荐 true。
BALANCE_BATCH="${BALANCE_BATCH:-true}"
# 总 epoch。正整数；若 TOTAL_TRAINING_STEPS 非 null，则后者可限制总步数。
EPOCHS="${EPOCHS:-1}"
# 总训练步数。可选：null 或正整数；正整数便于 smoke test/固定预算。
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
# 每多少个 step 保存。-1 不保存，正整数按间隔保存。
SAVE_FREQ="${SAVE_FREQ:-50}"
# 最多保留多少个 actor checkpoint。null 全保留，正整数自动清理旧 checkpoint。
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}"
# 每多少个 step 验证。-1 关闭；正整数定期验证。
TEST_FREQ="${TEST_FREQ:--1}"
# 训练前是否先验证。可选：false/true。
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
# 恢复模式。可选：disable（不恢复）、auto（自动找最新 checkpoint）。
RESUME_MODE="${RESUME_MODE:-disable}"
# 指定恢复 checkpoint。可选：空字符串或路径；使用时通常设 RESUME_MODE=resume_path。
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
# 日志后端的 Hydra 列表。常用：['console']、['console','wandb']。
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
# 每次验证记录多少条 prompt/output/score。0 关闭；正整数配合 wandb/swanlab 和 TEST_FREQ 使用。
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
# 训练 rollout 文本落盘目录。空字符串关闭；可设为 outputs/.../rollouts，便于定性检查但会占磁盘。
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
# 验证生成文本落盘目录。空字符串关闭；必须同时令 TEST_FREQ>0 才会产生文件。
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-}"
# 启动 veRL 前是否在终端打印最终训练配置。可选：true/false。
PRINT_TRAIN_CONFIG="${PRINT_TRAIN_CONFIG:-true}"
# 是否把最终训练配置保存到文件。可选：true/false。
SAVE_TRAIN_CONFIG="${SAVE_TRAIN_CONFIG:-true}"
# 配置保存路径。默认写入当前训练输出目录；可指定任意文本文件路径。
TRAIN_CONFIG_FILE="${TRAIN_CONFIG_FILE:-${output_dir}/training_config.txt}"
# 日志项目名。
PROJECT_NAME="${PROJECT_NAME:-sk_rag_opd}"
# 实验名；阶段脚本会分别设为 consistent/controlled/embedding。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd}"

# --- 算法与 reward ------------------------------------------------------------
# advantage estimator。直接反传 OPD 时保持 grpo；PG/task reward 模式可按 veRL 配置选择 gae、grpo 等。
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
# 是否把 reference KL 加入 reward。可选：false/true；纯 OPD 默认 false。
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-false}"
# reward Python 文件与函数名。纯 OPD 使用零 reward；启用 USE_TASK_REWARDS 时可替换。
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${project_root}/opd_rag/zero_reward.py}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"

if (( TEACHER_GPUS_PER_NODE % TEACHER_TP != 0 )); then
  echo "TEACHER_GPUS_PER_NODE must be divisible by TEACHER_TP" >&2
  exit 2
fi
teacher_replicas="${TEACHER_NUM_REPLICAS:-$((TEACHER_GPUS_PER_NODE / TEACHER_TP))}"
val_file="${VAL_FILE:-${train_file}}"
rollout_data_dir="${ROLLOUT_DATA_DIR:-null}"
validation_data_dir="${VALIDATION_DATA_DIR:-null}"

render_config_group() {
  local title="$1"
  shift
  printf '\n[%s]\n' "${title}"
  local config_name
  for config_name in "$@"; do
    printf '%-38s = %q\n' "${config_name}" "${!config_name}"
  done
}

render_training_config() {
  printf 'SK-RAG-OPD resolved training configuration\n'
  printf 'generated_at                           = %q\n' "$(date --iso-8601=seconds)"
  printf 'git_commit                             = %q\n' "$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"
  printf 'TRAIN_FILE                             = %q\n' "${train_file}"
  printf 'OUTPUT_DIR                             = %q\n' "${output_dir}"
  printf 'EFFECTIVE_VAL_FILE                     = %q\n' "${val_file}"
  printf 'EFFECTIVE_TEACHER_NUM_REPLICAS         = %q\n' "${teacher_replicas}"

  render_config_group "runtime_and_models" \
    PYTHON_BIN MODEL_PATH TEACHER_MODEL_PATH LORA_INIT_PATH ENABLE_THINKING \
    THINKING_OFF_TEMPLATE TRUST_REMOTE_CODE ATTN_IMPLEMENTATION
  render_config_group "data_and_lengths" \
    VAL_FILE TRAIN_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN \
    MAX_IMAGE_PIXELS FILTER_OVERLONG_PROMPTS FILTER_OVERLONG_PROMPTS_WORKERS TRUNCATION IMAGE_KEY \
    RETURN_MULTI_MODAL_INPUTS DATA_SHUFFLE SEED
  render_config_group "lora_and_student" \
    LORA_RANK LORA_ALPHA LORA_TARGET_MODULES USE_REMOVE_PADDING \
    ENABLE_GRADIENT_CHECKPOINTING FREEZE_VISION_TOWER
  render_config_group "optimizer_and_actor" \
    LR WEIGHT_DECAY LR_WARMUP_STEPS_RATIO LR_SCHEDULER_TYPE CLIP_GRAD \
    PPO_MINI_BATCH_SIZE MICRO_BATCH_SIZE USE_DYNAMIC_BSZ MAX_TOKEN_LEN_PER_GPU \
    PPO_EPOCHS LOSS_AGG_MODE
  render_config_group "fsdp_and_memory" \
    PARAM_OFFLOAD OPTIMIZER_OFFLOAD RESHARD_AFTER_FORWARD USE_TORCH_COMPILE
  render_config_group "student_rollout" \
    ROLLOUT_BACKEND VLLM_ATTENTION_BACKEND ROLLOUT_LOAD_FORMAT ROLLOUT_TP ROLLOUT_GPU_MEMORY_UTILIZATION \
    GENERATION_TEMPERATURE GENERATION_TOP_P GENERATION_TOP_K ROLLOUT_N \
    ROLLOUT_IGNORE_EOS ROLLOUT_ENFORCE_EAGER ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    ROLLOUT_MAX_NUM_SEQS ROLLOUT_ENABLE_CHUNKED_PREFILL ROLLOUT_ENABLE_PREFIX_CACHING
  render_config_group "teacher_and_distillation" \
    DISTILLATION_ENABLED DISTILLATION_LOSS_MODE DISTILLATION_TOPK \
    DISTILLATION_LOSS_COEF USE_TASK_REWARDS USE_POLICY_GRADIENT LOSS_MAX_CLAMP \
    LOG_PROB_MIN_CLAMP USE_CHUNKED_TOPK CHUNKED_TOPK_CHUNK_SIZE \
    DISTILLATION_POLICY_LOSS_MODE DISTILLATION_CLIP_RATIO TEACHER_TP \
    TEACHER_GPU_MEMORY_UTILIZATION TEACHER_MAX_NUM_SEQS TEACHER_LOAD_FORMAT \
    TEACHER_ENFORCE_EAGER TEACHER_NUM_REPLICAS
  render_config_group "distributed_logging_and_checkpoints" \
    TRAINER_GPUS_PER_NODE TEACHER_GPUS_PER_NODE NNODES TEACHER_NNODES \
    BALANCE_BATCH EPOCHS TOTAL_TRAINING_STEPS SAVE_FREQ MAX_ACTOR_CKPT_TO_KEEP \
    TEST_FREQ VAL_BEFORE_TRAIN RESUME_MODE RESUME_FROM_PATH TRAINER_LOGGER \
    LOG_VAL_GENERATIONS ROLLOUT_DATA_DIR VALIDATION_DATA_DIR PRINT_TRAIN_CONFIG \
    SAVE_TRAIN_CONFIG TRAIN_CONFIG_FILE PROJECT_NAME EXPERIMENT_NAME
  render_config_group "algorithm_and_reward" \
    ADV_ESTIMATOR USE_KL_IN_REWARD REWARD_FUNCTION_PATH REWARD_FUNCTION_NAME

  printf '\n[extra_hydra_overrides]\n'
  if (( $# == 0 )); then
    printf '(none)\n'
  else
    local override_index=0
    local override
    for override in "$@"; do
      printf 'override_%03d                           = %q\n' "${override_index}" "${override}"
      override_index=$((override_index + 1))
    done
  fi
}

[[ -f "${train_file}" ]] || { echo "veRL dataset does not exist: ${train_file}" >&2; exit 2; }
if ! runtime_import_error="$("${PYTHON_BIN}" -c \
  'import pandas, datasets, ray, hydra, vllm, transfer_queue; import verl.trainer.main_ppo' 2>&1)"; then
  printf 'veRL runtime import failed in %s:\n%s\n\n' "${PYTHON_BIN}" "${runtime_import_error}" >&2
  printf 'Repair the pinned runtime with:\n  %q -m pip install --no-cache-dir -r %q\n' \
    "${PYTHON_BIN}" "${project_root}/requirements-verl.txt" >&2
  printf 'If pandas itself is corrupted, reinstall it first with:\n  %q -m pip install --no-cache-dir --force-reinstall --no-deps pandas==2.2.3\n' \
    "${PYTHON_BIN}" >&2
  printf 'Then verify it with:\n  %q -c %q\n' "${PYTHON_BIN}" \
    'import pandas, datasets, vllm, transfer_queue; import verl.trainer.main_ppo; print(pandas.__version__)' >&2
  exit 2
fi

model_args=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}"
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
  "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}"
  "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
)
chat_template_args=()
if [[ "${ENABLE_THINKING}" != "true" ]]; then
  [[ -f "${THINKING_OFF_TEMPLATE}" ]] || {
    echo "thinking-off chat template does not exist: ${THINKING_OFF_TEMPLATE}" >&2
    exit 2
  }
  export OPD_CHAT_TEMPLATE
  OPD_CHAT_TEMPLATE="$(<"${THINKING_OFF_TEMPLATE}")"
  model_args+=('actor_rollout_ref.model.custom_chat_template=${oc.env:OPD_CHAT_TEMPLATE}')
else
  # The native Qwen template consumes this variable. The thinking-off custom
  # template already fixes the empty think block and must not receive it:
  # transformers 5 warns once per sample for unused processor kwargs.
  chat_template_args+=("+data.apply_chat_template_kwargs.enable_thinking=true")
fi
if [[ -n "${LORA_INIT_PATH}" ]]; then
  [[ -f "${LORA_INIT_PATH}/adapter_config.json" ]] || {
    echo "LORA_INIT_PATH is not an exported PEFT adapter: ${LORA_INIT_PATH}" >&2
    exit 2
  }
  model_args+=("actor_rollout_ref.model.lora_adapter_path=${LORA_INIT_PATH}")
fi

trainer_args=(
  "trainer.resume_mode=${RESUME_MODE}"
)
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  trainer_args+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

resolved_training_config="$(render_training_config "$@")"
if [[ "${SAVE_TRAIN_CONFIG}" == "true" ]]; then
  mkdir -p "$(dirname "${TRAIN_CONFIG_FILE}")"
  printf '%s\n' "${resolved_training_config}" > "${TRAIN_CONFIG_FILE}"
fi
if [[ "${PRINT_TRAIN_CONFIG}" == "true" ]]; then
  printf '\n%s\n' "${resolved_training_config}"
  if [[ "${SAVE_TRAIN_CONFIG}" == "true" ]]; then
    printf '\nTraining configuration saved to %s\n\n' "${TRAIN_CONFIG_FILE}"
  fi
fi

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  "algorithm.adv_estimator=${ADV_ESTIMATOR}" \
  "algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}" \
  "reward.custom_reward_function.path=${REWARD_FUNCTION_PATH}" \
  "reward.custom_reward_function.name=${REWARD_FUNCTION_NAME}" \
  "data.train_files=['${train_file}']" \
  "data.val_files=['${val_file}']" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
  "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
  "data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS}" \
  "data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS}" \
  "data.truncation=${TRUNCATION}" \
  "data.image_key=${IMAGE_KEY}" \
  "data.return_multi_modal_inputs=${RETURN_MULTI_MODAL_INPUTS}" \
  "data.shuffle=${DATA_SHUFFLE}" \
  "data.seed=${SEED}" \
  "${chat_template_args[@]}" \
  "+data.mm_processor_kwargs.max_pixels=${MAX_IMAGE_PIXELS}" \
  "${model_args[@]}" \
  "actor_rollout_ref.actor.optim.lr=${LR}" \
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}" \
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}" \
  "actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
  "actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD}" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}" \
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}" \
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}" \
  "actor_rollout_ref.actor.freeze_vision_tower=${FREEZE_VISION_TOWER}" \
  "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}" \
  "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${RESHARD_AFTER_FORWARD}" \
  "actor_rollout_ref.actor.fsdp_config.use_torch_compile=${USE_TORCH_COMPILE}" \
  "actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND}" \
  "actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.temperature=${GENERATION_TEMPERATURE}" \
  "actor_rollout_ref.rollout.top_p=${GENERATION_TOP_P}" \
  "actor_rollout_ref.rollout.top_k=${GENERATION_TOP_K}" \
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.seed=${SEED}" \
  "actor_rollout_ref.rollout.ignore_eos=${ROLLOUT_IGNORE_EOS}" \
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" \
  "actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  "actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}" \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=${VLLM_ATTENTION_BACKEND}" \
  "trainer.balance_batch=${BALANCE_BATCH}" \
  "trainer.logger=${TRAINER_LOGGER}" \
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}" \
  "trainer.rollout_data_dir=${rollout_data_dir}" \
  "trainer.validation_data_dir=${validation_data_dir}" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE}" \
  "trainer.nnodes=${NNODES}" \
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}" \
  "trainer.test_freq=${TEST_FREQ}" \
  "trainer.total_epochs=${EPOCHS}" \
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
  "trainer.default_local_dir=${output_dir}" \
  "${trainer_args[@]}" \
  "distillation.enabled=${DISTILLATION_ENABLED}" \
  "distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}" \
  "distillation.nnodes=${TEACHER_NNODES}" \
  "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL_PATH}" \
  "distillation.teacher_models.teacher_model.num_replicas=${teacher_replicas}" \
  "distillation.teacher_models.teacher_model.inference.name=${ROLLOUT_BACKEND}" \
  "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}" \
  "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}" \
  "distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}" \
  "distillation.teacher_models.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}" \
  "distillation.teacher_models.teacher_model.inference.load_format=${TEACHER_LOAD_FORMAT}" \
  "distillation.teacher_models.teacher_model.inference.enforce_eager=${TEACHER_ENFORCE_EAGER}" \
  "+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.attention_backend=${VLLM_ATTENTION_BACKEND}" \
  "distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}" \
  "distillation.distillation_loss.topk=${DISTILLATION_TOPK}" \
  "distillation.distillation_loss.distillation_loss_coef=${DISTILLATION_LOSS_COEF}" \
  "distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}" \
  "distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}" \
  "distillation.distillation_loss.loss_max_clamp=${LOSS_MAX_CLAMP}" \
  "distillation.distillation_loss.log_prob_min_clamp=${LOG_PROB_MIN_CLAMP}" \
  "+distillation.distillation_loss.use_chunked_topk=${USE_CHUNKED_TOPK}" \
  "+distillation.distillation_loss.chunked_topk_chunk_size=${CHUNKED_TOPK_CHUNK_SIZE}" \
  "distillation.distillation_loss.policy_loss_mode=${DISTILLATION_POLICY_LOSS_MODE}" \
  "distillation.distillation_loss.clip_ratio=${DISTILLATION_CLIP_RATIO}" \
  "distillation.distillation_loss.clip_ratio_low=${DISTILLATION_CLIP_RATIO}" \
  "distillation.distillation_loss.clip_ratio_high=${DISTILLATION_CLIP_RATIO}" \
  "$@"
