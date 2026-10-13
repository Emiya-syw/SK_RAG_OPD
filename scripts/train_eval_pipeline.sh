#!/usr/bin/env bash
# Train -> export latest LoRA -> link retrieval data -> generate -> judge.
# Usage: bash scripts/train_eval_pipeline.sh [SOURCE_JSONL] [TRAIN_OUTPUT_DIR]
# Data precedence: SOURCE_JSONL argument > TRAIN_DATA_FILE > stage default.
# Override defaults with environment variables. START_STAGE=train|export|generate|judge.
# DRY_RUN=true prints the plan without running stages or requiring an API key.
# Existing artifacts are checked per stage/dataset; complete stages print [SKIP].
# DASHSCOPE_API_KEY is required only for unfinished scoring; it is never stored.
# Separate image limits: TRAIN_MAX_IMAGE_PIXELS, EVAL_MAX_IMAGE_PIXELS,
# JUDGE_MAX_IMAGE_PIXELS. Other training overrides pass through to run_opd.sh.
# AUTO_CONFIG=true detects 4/8 A100 80GB and Qwen3-VL 2B/4B/8B configs.
# OVERLAP_GENERATION_JUDGING=true (default): generate and judge in parallel, one judge dataset at a time.
# Explicit environment overrides win. AUTO_CONFIG=false uses manual defaults.
# STAGE1_EXPERIMENT_NAME resolves outputs/<name>'s latest adapter for stage 2.
# Empty/unset STAGE1_EXPERIMENT_NAME: consistent stage 1; otherwise controlled stage 2.
# The stage also selects the default source JSONL and clears stale LORA_INIT_PATH for stage 1.
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
stage_tag=stage1
if [[ -n "${STAGE1_EXPERIMENT_NAME:-}" ]]; then stage_tag=stage2; fi
default_env EXPERIMENT_NAME "${model_tag}_teacher_${teacher_tag}_${PROMPT_MODE}_${stage_tag}"
# Delay GPU detection until a stage actually needs the GPU.
configure_runtime() {
  if [[ "${runtime_configured:-false}" == true ]]; then return; fi
  # Resolve presets before filling manual defaults. Judge-only needs no CUDA or model files.
  if [[ "${AUTO_CONFIG:-true}" == true ]]; then
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
  runtime_configured=true
  printf 'GPU config: visible=%s student=%s teacher=%s batch=%s prompt=%s\n' \
    "$CUDA_VISIBLE_DEVICES" "$TRAINER_GPUS_PER_NODE" "$TEACHER_GPUS_PER_NODE" "$TRAIN_BATCH_SIZE" "$PROMPT_MODE"
}
default_env DATASETS 'MSSBench VLGuard'
export PYTHONPATH="${repo_root}/verl-runtime${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${STAGE1_EXPERIMENT_NAME:-}" ]]; then
  train_script="scripts/train_controlled.sh"
  default_source="data/category_controlled_merged.jsonl"
else
  train_script="scripts/train_consistent.sh"
  default_source="data/category_consistent_abstract.jsonl"
fi
source_file="$(absolute_path "${1:-${TRAIN_DATA_FILE:-$default_source}}")"
train_dir="$(absolute_path "${2:-outputs/${EXPERIMENT_NAME}}")"
result_dir="$(absolute_path "${OUTPUT_DIR:-rag_eval/results/${EXPERIMENT_NAME}}")"
judge_dir="$(absolute_path "${EVAL_OUTPUT_ROOT:-rag_eval/evaluation/${EXPERIMENT_NAME}/mssbench_protocol}")"
data_root="$(absolute_path "${DATA_ROOT:-rag_eval/data}")"
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
printf 'Prompt mode: %s\nTraining entry: %s\n' "$PROMPT_MODE" "$train_script"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
artifacts() { "$PYTHON_BIN" scripts/pipeline_artifacts.py "$@"; }
is_dry() { [[ "${DRY_RUN:-false}" == true ]]; }
run_step() {
  if is_dry; then printf '[PLAN]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
resolve_stage1_adapter() {
  local stage1_name="${STAGE1_EXPERIMENT_NAME:-}" stage1_dir init_adapter
  if [[ -z "$stage1_name" ]]; then
    unset LORA_INIT_PATH
    echo '[INIT] No stage-1 experiment specified; training from base model without LoRA initialization'
    return 0
  else
    case "$stage1_name" in
      .|..|*/*) echo 'STAGE1_EXPERIMENT_NAME must be an experiment name, not a path' >&2; exit 2 ;;
    esac
    stage1_dir="$repo_root/outputs/$stage1_name"
    [[ "$stage1_dir" != "$train_dir" && ! "$stage1_dir" -ef "$train_dir" ]] || {
      echo 'Stage 1 and stage 2 must use different output directories' >&2; exit 2;
    }
    init_adapter="$(artifacts adapter "$stage1_dir" 2>/dev/null || true)"
    if [[ -n "$init_adapter" ]]; then
      echo "[SKIP stage1 export] Latest stage-1 adapter exists: $init_adapter"
    else
      artifacts checkpoint "$stage1_dir" >/dev/null || {
        echo "No complete latest stage-1 checkpoint found: $stage1_dir" >&2; exit 2;
      }
      echo "[INIT] Exporting latest stage-1 checkpoint: $stage1_dir"
      if is_dry; then
        run_step bash scripts/latest_adapter.sh "$stage1_dir"
        unset LORA_INIT_PATH
        echo '[PLAN] Set LORA_INIT_PATH to the exported stage-1 adapter before training'
        return 0
      fi
      bash scripts/latest_adapter.sh "$stage1_dir" | tee "$stage1_dir/pipeline_adapter_export.log"
      init_adapter="$(tail -n 1 "$stage1_dir/pipeline_adapter_export.log")"
      init_adapter="$(absolute_path "$init_adapter")"
      artifacts valid-adapter "$init_adapter" >/dev/null || {
        echo "Incomplete stage-1 adapter: $init_adapter" >&2; exit 2;
      }
    fi
  fi
  export LORA_INIT_PATH="$init_adapter"
  echo "[INIT] Stage-2 student starts from: $LORA_INIT_PATH"
}
adapter="${ADAPTER_PATH:-}"
if [[ -n "$adapter" ]]; then adapter="$(absolute_path "$adapter")"; fi
existing_checkpoint="$(artifacts checkpoint "$train_dir" 2>/dev/null || true)"
if (( start == 1 )); then
  if [[ -n "$existing_checkpoint" ]]; then
    echo "[SKIP 1/4] Training weights exist: $existing_checkpoint (reuse checkpoint, not proof all epochs finished)"
  elif [[ -n "$adapter" ]] && artifacts valid-adapter "$adapter" >/dev/null 2>&1; then
    echo "[SKIP 1/4] Using existing adapter: $adapter"
  else
    resolve_stage1_adapter
    configure_runtime
    if ! is_dry; then
      [[ -f "$source_file" && -f "$train_script" && -d "$TEACHER_MODEL_PATH" ]] || {
        echo 'Missing training source, training script, or teacher model' >&2; exit 2;
      }
      for dataset in "${datasets[@]}"; do
        if compgen -G "$result_dir/$dataset/answers*.jsonl" >/dev/null; then
          echo "No reusable checkpoint, but answers exist in $result_dir/$dataset; use a fresh OUTPUT_DIR." >&2; exit 2
        fi
      done
      mkdir -p "$train_dir"
      if [[ -n "${auto_settings:-}" ]]; then printf '%s\n' "$auto_settings" > "$train_dir/pipeline_auto_config.tsv"; fi
    fi
    echo '[1/4] Training'
    run_step env MAX_IMAGE_PIXELS="${TRAIN_MAX_IMAGE_PIXELS:-50176}" bash "$train_script" "$source_file" "$train_dir"
  fi
fi

if (( start <= 3 )); then
  if [[ -z "$adapter" ]]; then adapter="$(artifacts adapter "$train_dir" 2>/dev/null || true)"; fi
  if [[ -n "$adapter" ]] && artifacts valid-adapter "$adapter" >/dev/null 2>&1; then
    echo "[SKIP 2/4] Exported adapter exists: $adapter"
  elif [[ -n "$adapter" ]]; then
    echo "Invalid ADAPTER_PATH: $adapter" >&2; exit 2
  elif (( start <= 2 )); then
    echo '[2/4] Exporting latest checkpoint adapter'
    if is_dry; then
      run_step bash scripts/latest_adapter.sh "$train_dir"
    else
      mkdir -p "$train_dir"
      bash scripts/latest_adapter.sh "$train_dir" | tee "$train_dir/pipeline_adapter_export.log"
      adapter="$(tail -n 1 "$train_dir/pipeline_adapter_export.log")"
      adapter="$(absolute_path "$adapter")"
      artifacts valid-adapter "$adapter" >/dev/null || { echo "Incomplete exported adapter: $adapter" >&2; exit 2; }
    fi
  else
    echo 'No exported adapter found; use START_STAGE=export or supply ADAPTER_PATH.' >&2; exit 2
  fi

  pending_datasets=()
  for dataset in "${datasets[@]}"; do
    link="$result_dir/$dataset/retrieval.jsonl"
    source_retrieval="$data_root/$dataset/test_mixed_top10.jsonl"
    if [[ -f "$link" ]]; then
      echo "[SKIP data] $dataset retrieval exists: $link"
    else
      [[ -f "$source_retrieval" ]] || { echo "Missing retrieval input: $source_retrieval" >&2; exit 2; }
      run_step mkdir -p "$result_dir/$dataset"
      run_step ln -sfnT "$source_retrieval" "$link"
    fi
    check_source="$link"
    [[ -f "$check_source" ]] || check_source="$source_retrieval"
    check_status=0
    check_message="$(artifacts answers "$check_source" "$result_dir/$dataset/answers.jsonl" "$adapter" 2>&1)" || check_status=$?
    if (( check_status == 2 )); then echo "$check_message" >&2; exit 2; fi
    if (( check_status == 0 )); then
      echo "[SKIP 3/4] $dataset answers complete: $result_dir/$dataset/answers.jsonl"
    else
      echo "[3/4] $dataset answers missing/incomplete; generation will resume available shards"
      pending_datasets+=("$dataset")
    fi
  done
  if (( ${#pending_datasets[@]} )); then
    configure_runtime
    generation_command=(bash rag_eval/run_rag_eval.sh)
    if [[ "${OVERLAP_GENERATION_JUDGING:-true}" == true ]]; then
      generation_command=("$PYTHON_BIN" -m rag_eval.overlap_pipeline --datasets "${datasets[@]}")
      overlap_completed=true
      echo '[QUEUE] Generation and ordered judging will overlap'
    fi
    run_step env BASE_MODEL="$MODEL_PATH" ADAPTER_PATH="$adapter" OUTPUT_DIR="$result_dir" DATA_ROOT="$data_root" \
      EVAL_OUTPUT_ROOT="$judge_dir" MODEL_NAME="$EXPERIMENT_NAME" \
      DATASETS="${pending_datasets[*]}" RETRIEVAL_ROOT="$result_dir" STAGE=generate DIRECT_GENERATION=false \
      TOP_K="${TOP_K:-3}" MULTI_GPU=true GPU_IDS="${GPU_IDS:-$CUDA_VISIBLE_DEVICES}" \
      GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-64}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}" \
      MAX_IMAGE_PIXELS="${EVAL_MAX_IMAGE_PIXELS:-262144}" DTYPE="${DTYPE:-bfloat16}" LIMIT="${LIMIT:-0}" \
      "${generation_command[@]}"
  fi
fi

pending_judges=()
for dataset in "${datasets[@]}"; do
  if artifacts judge "$result_dir/$dataset/answers.jsonl" "$judge_dir/$dataset" 2>/dev/null; then
    echo "[SKIP 4/4] $dataset scoring complete: $judge_dir/$dataset/summary.json"
  elif [[ "${overlap_completed:-false}" != true ]]; then
    pending_judges+=("$dataset")
  elif ! is_dry; then
    echo "[QUEUE] $dataset has remaining judge errors; retained for retry on next run"
  fi
done
if (( ${#pending_judges[@]} )); then
  if ! is_dry; then : "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY to score unfinished datasets}"; fi
  echo "[4/4] Scoring unfinished datasets: ${pending_judges[*]}"
  run_step env INPUT_ROOT="$result_dir" EVAL_OUTPUT_ROOT="$judge_dir" MODEL_NAME="$EXPERIMENT_NAME" \
    DATASETS="${pending_judges[*]}" MAX_IMAGE_PIXELS="${JUDGE_MAX_IMAGE_PIXELS:-786432}" LIMIT="${LIMIT:-0}" \
    bash rag_eval/run_mssbench_protocol_eval.sh
fi
# The evaluator writes a root summary for only the datasets passed to it.
# Restore a combined summary, including previously completed datasets.
run_step artifacts summarize "$judge_dir" "${datasets[@]}"
printf 'Pipeline completed. Summary: %s/summary.json\n' "$judge_dir"
