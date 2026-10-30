#!/usr/bin/env bash
# Evaluate all seven datasets with the precomputed 8B mixed retrieval v2 files.
# LoRA example (activate the inference environment first):
# BASE_MODEL=/path/to/base-model ADAPTER_PATH=/path/to/lora_adapter \
#   RUN_NAME=my_checkpoint_v2 GPU_IDS=0,1 STAGE=all \
#   bash scripts/eval_mixed_top10_v2.sh
# Merged/full model: BASE_MODEL=/path/to/merged-model ADAPTER_PATH='' ...
# STAGE=generate: local inference only; STAGE=judge: score existing answers.
# STAGE=all (default): both; scoring requires DASHSCOPE_API_KEY in the environment.
# DRY_RUN=true checks inputs and prints commands without inference/API calls/writes.
# TOP_K defaults to 3 (existing pipeline convention); set TOP_K=10 to use all ten.
# Scoring uses the shared MSSBench protocol, not each benchmark's official metric.
# The v2 corpus includes test-derived knowledge, as recorded in retrieval_config.
# Multiple models: MODELS_FILE=scripts/eval_mixed_top10_v2.models.example.json
# STAGE=all streams completed answers to the judge while generation continues.
# 8x4090 defaults: GPU_IDS=0,1,2,3,4,5,6,7, BF16, one replica per GPU,
# merged LoRA, auto FlashAttention2/SDPA, adaptive judge concurrency up to 16.
# EVAL_PROFILE=auto reads the base config; explicit choices: qwen3vl-2b,
# qwen3vl-4b, qwen3vl-8b, qwen25vl-7b. GENERATION_BATCH_SIZE overrides the profile.
# FlashAttention2 batches at TOP_K<=3: 8/4/2/2; SDPA or TOP_K>3 halves these.
# Initial throughput presets, not measured maxima; batch OOM retries halve again.
# For a separate smoke run: LIMIT=16 RESULTS_ROOT=/tmp/v2-smoke-results
# SCORES_ROOT=/tmp/v2-smoke-scores MODELS_FILE=... STAGE=all bash this-script
# Remote data must preserve/relocate absolute image paths in JSONL.
# The original mixed corpus is not loaded or validated for precomputed v2 runs.
# Automatic checkpoint/base lookup:
# bash scripts/eval_mixed_top10_v2.sh opd_s2b_t8b_stage2_entropy_jsd_v2
# Searches outputs/<name>, then outputs/<name without _v2>; latest numeric step.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
if (( $# > 1 )); then
  echo 'Usage: eval_mixed_top10_v2.sh [EXPERIMENT_NAME_OR_RUN_NAME]' >&2; exit 2
fi
if (( $# == 1 )); then
  export RUN_NAME="$1"
fi
if [[ -n "${MODELS_FILE:-}" ]]; then
  exec "$PYTHON_BIN" -m scripts.eval_mixed_v2_runner --models "$MODELS_FILE"
fi
export DATASETS="${DATASETS:-MM-SafetyBench VLGuard MSSBench SPA-VL SIUO-Gen VLSU VLSBench}"
export RUN_NAME="${RUN_NAME:-mixed_top10_v2}"
output_name="$RUN_NAME"
[[ "$output_name" == *_mixed_v2 ]] || output_name="${output_name}_mixed_v2"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/rag_eval/results/${output_name}}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${repo_root}/rag_eval/evaluation/${output_name}/mssbench_protocol}"
export SOURCE_DATA_ROOT="${DATA_ROOT:-${repo_root}/rag_eval/data}"
export DATA_ROOT="${OUTPUT_DIR}/inputs/data"
export RETRIEVAL_ROOT="${OUTPUT_DIR}/inputs/retrieval_v2"
export TOP_K="${TOP_K:-3}"
export PROMPT_MODE="${PROMPT_MODE:-guided}"
export MULTI_GPU="${MULTI_GPU:-true}"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:0}"
export AUTO_GPU_POLICY="${AUTO_GPU_POLICY:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export JUDGE_ADAPTIVE="${JUDGE_ADAPTIVE:-true}"
export FLUSH_EVERY="${FLUSH_EVERY:-4}"
export DIRECT_GENERATION=false
export SKIP_CORPUS_VALIDATION=true
export LIMIT="${LIMIT:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
export MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export MERGE_LORA_FOR_INFERENCE="${MERGE_LORA_FOR_INFERENCE:-true}"
export DTYPE="${DTYPE:-bfloat16}"
export EVAL_ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-${ATTN_IMPLEMENTATION:-auto}}"
export DRY_RUN="${DRY_RUN:-false}"
stage="${STAGE:-all}"
case "$stage" in all|generate|judge) ;; *) echo 'STAGE must be all, generate, or judge' >&2; exit 2 ;; esac
[[ "$TOP_K" =~ ^([1-9]|10)$ ]] || { echo 'TOP_K must be 1..10' >&2; exit 2; }
[[ "$DRY_RUN" == true || "$DRY_RUN" == false ]] || { echo 'DRY_RUN must be true or false' >&2; exit 2; }
if [[ "$stage" != generate && "$DRY_RUN" != true && "${RECOMPUTE_ONLY:-false}" != true ]]; then
  : "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for judging, or use STAGE=generate}"
fi

if [[ "$stage" != judge ]]; then
  if [[ ! ${ADAPTER_PATH+x} ]]; then
    resolved_paths="$("$PYTHON_BIN" -m scripts.resolve_eval_checkpoint)"
    mapfile -t model_paths <<< "$resolved_paths"
    export BASE_MODEL="${model_paths[0]}" ADAPTER_PATH="${model_paths[1]}"
  fi
  : "${BASE_MODEL:?Set BASE_MODEL to the trained model or the matching LoRA base model}"
  # Explicitly empty ADAPTER_PATH selects a full/merged model; never inherit an unrelated adapter.
  [[ ${ADAPTER_PATH+x} ]] || { echo "Set ADAPTER_PATH to the exported LoRA directory, or '' for a full/merged model." >&2; exit 2; }
  export BASE_MODEL ADAPTER_PATH
  profile_settings="$("$PYTHON_BIN" -m scripts.eval_mixed_v2_profiles)"
  read -r EVAL_PROFILE profile_batch EVAL_ATTN_IMPLEMENTATION <<< "$profile_settings"
  export EVAL_PROFILE EVAL_ATTN_IMPLEMENTATION
  export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$profile_batch}"
  echo "4090 profile: $EVAL_PROFILE; per-GPU batch=$GENERATION_BATCH_SIZE; attention=$EVAL_ATTN_IMPLEMENTATION"
  [[ -d "$BASE_MODEL" ]] || { echo "Missing model: $BASE_MODEL" >&2; exit 1; }
  if [[ -n "$ADAPTER_PATH" ]]; then
    [[ -f "$ADAPTER_PATH/adapter_config.json" ]] || { echo "Missing adapter_config.json: $ADAPTER_PATH" >&2; exit 1; }
    [[ -s "$ADAPTER_PATH/adapter_model.safetensors" || -s "$ADAPTER_PATH/adapter_model.bin" ]] || {
      echo "Missing exported adapter weights: $ADAPTER_PATH" >&2; exit 1;
    }
  fi

  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

env = os.environ
source = Path(env['SOURCE_DATA_ROOT']).resolve()
output = Path(env['OUTPUT_DIR']).resolve()
links = []
inputs = {}
for name in env['DATASETS'].split():
    test = source / name / 'test.jsonl'
    v2 = source / name / 'test_mixed_top10_v2.jsonl'
    test_rows = [json.loads(line) for line in test.read_text().splitlines() if line.strip()]
    raw = v2.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [str(row['id']) for row in rows]
    if not rows or len(set(ids)) != len(ids) or ids != [str(row['id']) for row in test_rows]:
        raise ValueError(f'{name}: v2 IDs/order must match the original test split')
    for row in rows:
        config = row['retrieval_config']
        if config.get('version') != 2 or len(row['retrieval']) < int(env['TOP_K']):
            raise ValueError(f'{name}/{row["id"]}: invalid v2 retrieval or insufficient cases')
    # Keep the corpus name as provenance only; no file access is required.
    corpus = rows[0].get('retrieval_config', {}).get('corpus', '')
    data = output / 'inputs/data' / name
    retrieval = output / 'inputs/retrieval_v2' / name
    links.extend([(data / 'test.jsonl', test),
                  (retrieval / 'retrieval.jsonl', v2)])
    inputs[name] = {'v2': str(v2), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'corpus': str(corpus), 'count': len(rows)}
    print(f'{name}: {len(rows)} samples, {v2}')

keys = ('BASE_MODEL', 'ADAPTER_PATH', 'TOP_K', 'PROMPT_MODE', 'LIMIT',
        'MAX_NEW_TOKENS', 'MAX_IMAGE_PIXELS', 'ENABLE_THINKING', 'DTYPE',
        'EVAL_ATTN_IMPLEMENTATION', 'MERGE_LORA_FOR_INFERENCE', 'MULTI_GPU', 'GPU_IDS',
        'EVAL_PROFILE', 'GENERATION_BATCH_SIZE', 'EVAL_DEVICE_MAP')
manifest = {'settings': {key: env[key] for key in keys}, 'inputs': inputs}
manifest_path = output / 'mixed_v2_run.json'
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Run settings/inputs changed. Use a different RUN_NAME or OUTPUT_DIR to avoid mixing answers.')
elif output.exists() and any(output.glob('*/answers*.jsonl')):
    raise ValueError('Existing answers have no v2 manifest. Use a new RUN_NAME or OUTPUT_DIR.')
if env['DRY_RUN'] != 'true':
    output.mkdir(parents=True, exist_ok=True)
    for link, target in links:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f'Refusing to replace {link}')
        link.symlink_to(target)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
PY
fi

echo "Datasets: $DATASETS"
echo "Answers: $OUTPUT_DIR/<dataset>/answers.jsonl"
echo "Scores: $EVAL_OUTPUT_ROOT"
echo "Stage=$stage; TOP_K=$TOP_K; GPUs=$GPU_IDS; prompt=$PROMPT_MODE"
if [[ "$stage" == all ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    echo 'Would stream completed answer records to the judge concurrently with GPU generation.'
  else
    STAGE=generate "$PYTHON_BIN" -m scripts.eval_mixed_v2_runner --stream
  fi
  exit 0
fi
if [[ "$stage" != judge ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    echo 'Would run: STAGE=generate bash rag_eval/run_rag_eval.sh (with the settings above)'
  else
    STAGE=generate bash rag_eval/run_rag_eval.sh
  fi
fi
if [[ "$stage" != generate ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    echo 'Would run: bash rag_eval/run_mssbench_protocol_eval.sh (INPUT_ROOT=OUTPUT_DIR)'
  else
    INPUT_ROOT="$OUTPUT_DIR" MODEL_NAME="$RUN_NAME" bash rag_eval/run_mssbench_protocol_eval.sh
  fi
fi
