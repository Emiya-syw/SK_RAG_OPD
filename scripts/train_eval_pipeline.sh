#!/usr/bin/env bash
# Train -> export latest LoRA -> link retrieval data -> generate -> judge.
# Usage: bash scripts/train_eval_pipeline.sh [SOURCE_JSONL] [TRAIN_OUTPUT_DIR]
# Override defaults with environment variables. START_STAGE=train|export|generate|judge.
# DRY_RUN=true prints the plan without running stages or requiring an API key.
# Set DASHSCOPE_API_KEY in the environment; it is never stored by this script.
# Separate image limits: TRAIN_MAX_IMAGE_PIXELS, EVAL_MAX_IMAGE_PIXELS,
# JUDGE_MAX_IMAGE_PIXELS. Other training overrides pass through to run_opd.sh.
# AUTO_CONFIG=true detects 4/8 A100 80GB and Qwen3-VL 2B/4B/8B configs.
# Explicit environment overrides win. AUTO_CONFIG=false uses manual defaults.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if [[ "${1:-}" == --help ]]; then
  sed -n '2,8p' "${BASH_SOURCE[0]}"
  exit 0
fi
(( $# <= 2 )) || { echo 'Expected at most SOURCE_JSONL and TRAIN_OUTPUT_DIR' >&2; exit 2; }

# Only assign an unset variable, preserving explicit caller overrides.
default_env() {
  local name="$1" value="$2"
  if [[ ! -v "$name" ]]; then printf -v "$name" '%s' "$value"; fi
  export "$name"
}
absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$repo_root" "$1"; fi
}

default_env PYTHON_BIN /opt/conda/envs/sk_rag_opd_verl/bin/python
default_env MODEL_PATH "${repo_root}/models/Qwen3-VL-4B-Thinking"
default_env TEACHER_MODEL_PATH "${repo_root}/models/Qwen3-VL-8B-Teacher-S2-Merged"
default_env PROMPT_MODE guided
default_env ENABLE_THINKING false
default_env ATTN_IMPLEMENTATION flash_attention_2
# Stable names follow the actual model paths and prompt mode, including custom teachers.
model_tag="$(basename "$MODEL_PATH")"
teacher_tag="$(basename "$TEACHER_MODEL_PATH")"
default_env EXPERIMENT_NAME "${model_tag}_teacher_${teacher_tag}_${PROMPT_MODE}"
# Resolve presets before filling manual defaults. Judge-only needs no CUDA or model files.
if [[ "${AUTO_CONFIG:-true}" == true && "${START_STAGE:-train}" != judge ]]; then
  auto_settings="$("$PYTHON_BIN" scripts/pipeline_auto_config.py)"
  while IFS=$'\t' read -r name value; do
    default_env "$name" "$value"
  done <<< "$auto_settings"
fi
default_env CUDA_VISIBLE_DEVICES 0,1,2,3
default_env TRAIN_BATCH_SIZE 4
default_env PPO_MINI_BATCH_SIZE 4
default_env MICRO_BATCH_SIZE 1
default_env MAX_PROMPT_LENGTH 20480
default_env MAX_RESPONSE_LENGTH 1024
default_env FILTER_OVERLONG_PROMPTS true
default_env MAX_TOKEN_LEN_PER_GPU 24576
default_env PARAM_OFFLOAD false
default_env OPTIMIZER_OFFLOAD false
default_env ROLLOUT_GPU_MEMORY_UTILIZATION 0.60
default_env ROLLOUT_MAX_NUM_BATCHED_TOKENS 8192
default_env ROLLOUT_MAX_NUM_SEQS 24
default_env TEACHER_GPU_MEMORY_UTILIZATION 0.25
default_env TEACHER_MAX_NUM_SEQS 4
default_env DISTILLATION_LOSS_MODE k3
default_env TRAINER_GPUS_PER_NODE 2
default_env TEACHER_GPUS_PER_NODE 2
default_env TEACHER_TP 1
default_env TEACHER_NUM_REPLICAS 2
default_env TEACHER_USE_PRIVILEGED_INFO true
default_env SAVE_FREQ 200
default_env DATASETS 'MSSBench VLGuard MM-SafetyBench'
export PYTHONPATH="${repo_root}/verl-runtime${PYTHONPATH:+:${PYTHONPATH}}"

source_file="$(absolute_path "${1:-data/category_consistent_abstract.jsonl}")"
train_dir="$(absolute_path "${2:-outputs/${EXPERIMENT_NAME}}")"
result_dir="$(absolute_path "${OUTPUT_DIR:-rag_eval/results/${EXPERIMENT_NAME}}")"
judge_dir="$(absolute_path "${EVAL_OUTPUT_ROOT:-rag_eval/evaluation/${EXPERIMENT_NAME}/mssbench_protocol}")"
data_root="$(absolute_path "${DATA_ROOT:-rag_eval/data}")"
train_script="${TRAIN_SCRIPT:-scripts/train_consistent.sh}"
start_stage="${START_STAGE:-train}"
case "$start_stage" in
  train) start=1 ;;
  export) start=2 ;;
  generate) start=3 ;;
  judge) start=4 ;;
  *) echo 'START_STAGE must be train, export, generate, or judge' >&2; exit 2 ;;
esac
read -r -a datasets <<< "$DATASETS"
(( ${#datasets[@]} > 0 )) || { echo 'DATASETS must not be empty' >&2; exit 2; }
for dataset in "${datasets[@]}"; do
  case "$dataset" in MSSBench|VLGuard|MM-SafetyBench) ;; *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;; esac
done

printf 'Start stage: %s\nStudent: %s\nTeacher: %s\nTraining data: %s\nCheckpoints: %s\nAnswers: %s\nScores: %s\n' \
  "$start_stage" "$MODEL_PATH" "$TEACHER_MODEL_PATH" "$source_file" "$train_dir" "$result_dir" "$judge_dir"
printf 'Prompt mode: %s\n' "$PROMPT_MODE"
for name in CUDA_VISIBLE_DEVICES TRAINER_GPUS_PER_NODE TEACHER_GPUS_PER_NODE TEACHER_TP TEACHER_NUM_REPLICAS TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_TOKEN_LEN_PER_GPU ROLLOUT_GPU_MEMORY_UTILIZATION TEACHER_GPU_MEMORY_UTILIZATION TEACHER_MAX_NUM_SEQS; do
  printf '%s=%s\n' "$name" "${!name}"
done
printf 'GENERATION_BATCH_SIZE=%s\n' "${GENERATION_BATCH_SIZE:-64}"
printf 'Training LR: %s (run_opd.sh default if unset)\nImage pixels: train=%s generate=%s judge=%s\n' \
  "${LR:-5e-6}" "${TRAIN_MAX_IMAGE_PIXELS:-50176}" "${EVAL_MAX_IMAGE_PIXELS:-262144}" "${JUDGE_MAX_IMAGE_PIXELS:-786432}"
if [[ "${DRY_RUN:-false}" == true ]]; then exit 0; fi
: "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY before starting the pipeline}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
if (( start <= 3 )); then
  [[ -d "$MODEL_PATH" ]] || { echo "Missing student model: $MODEL_PATH" >&2; exit 2; }
  for dataset in "${datasets[@]}"; do
    [[ -f "$data_root/$dataset/test_mixed_top10.jsonl" ]] || {
      echo "Missing retrieval input: $data_root/$dataset/test_mixed_top10.jsonl" >&2; exit 2;
    }
  done
fi
if (( start == 1 )); then
  [[ -f "$source_file" && -f "$train_script" && -d "$TEACHER_MODEL_PATH" ]] || {
    echo 'Missing training source, training script, or teacher model' >&2; exit 2;
  }
  # Generation resumes existing answers by default. Never silently reuse them
  # after a new training run changes the adapter.
  for dataset in "${datasets[@]}"; do
    if compgen -G "$result_dir/$dataset/answers*.jsonl" >/dev/null; then
      echo "Existing answers in $result_dir/$dataset; choose a fresh OUTPUT_DIR for new training." >&2
      exit 2
    fi
  done
  mkdir -p "$train_dir"
  # Save only the selected preset, never the full environment/API credentials.
  if [[ -n "${auto_settings:-}" ]]; then
    printf '%s\n' "$auto_settings" > "$train_dir/pipeline_auto_config.tsv"
  fi
  echo '[1/4] Training'
  MAX_IMAGE_PIXELS="${TRAIN_MAX_IMAGE_PIXELS:-50176}" bash "$train_script" "$source_file" "$train_dir"
fi

adapter="${ADAPTER_PATH:-}"
if (( start <= 2 )); then
  echo '[2/4] Exporting latest checkpoint adapter'
  mkdir -p "$train_dir"
  # The merger emits logs before the final adapter path. Capture the final
  # line only, while keeping the full output visible and on disk.
  bash scripts/latest_adapter.sh "$train_dir" | tee "$train_dir/pipeline_adapter_export.log"
  adapter="$(tail -n 1 "$train_dir/pipeline_adapter_export.log")"
fi
if (( start <= 3 )); then
  : "${adapter:?START_STAGE=generate requires ADAPTER_PATH (or use START_STAGE=export)}"
  adapter="$(absolute_path "$adapter")"
  [[ -f "$adapter/adapter_config.json" ]] || { echo "Missing adapter config: $adapter" >&2; exit 2; }
  [[ -f "$adapter/adapter_model.safetensors" || -f "$adapter/adapter_model.bin" ]] || {
    echo "Missing adapter weights: $adapter" >&2; exit 2;
  }
  echo "[3/4] Generating with adapter: $adapter"
  for dataset in "${datasets[@]}"; do
    mkdir -p "$result_dir/$dataset"
    link="$result_dir/$dataset/retrieval.jsonl"
    if [[ -e "$link" && ! -L "$link" ]]; then
      echo "Refusing to replace existing non-symlink retrieval file: $link" >&2; exit 2
    fi
    ln -sfnT "$data_root/$dataset/test_mixed_top10.jsonl" "$link"
  done
  BASE_MODEL="$MODEL_PATH" ADAPTER_PATH="$adapter" OUTPUT_DIR="$result_dir" DATA_ROOT="$data_root" \
    RETRIEVAL_ROOT="$result_dir" STAGE=generate DIRECT_GENERATION=false \
    TOP_K="${TOP_K:-3}" MULTI_GPU=true GPU_IDS="${GPU_IDS:-$CUDA_VISIBLE_DEVICES}" \
    GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-64}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}" \
    MAX_IMAGE_PIXELS="${EVAL_MAX_IMAGE_PIXELS:-262144}" DTYPE="${DTYPE:-bfloat16}" LIMIT="${LIMIT:-0}" \
    bash rag_eval/run_rag_eval.sh
fi

echo '[4/4] Scoring generated answers'
INPUT_ROOT="$result_dir" EVAL_OUTPUT_ROOT="$judge_dir" MODEL_NAME="$EXPERIMENT_NAME" \
  MAX_IMAGE_PIXELS="${JUDGE_MAX_IMAGE_PIXELS:-786432}" LIMIT="${LIMIT:-0}" \
  bash rag_eval/run_mssbench_protocol_eval.sh
printf 'Pipeline completed. Summary: %s/summary.json\n' "$judge_dir"
