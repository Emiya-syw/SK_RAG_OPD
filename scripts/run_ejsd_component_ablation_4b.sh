#!/usr/bin/env bash
# 4B OPSD/OPD Entropy JSD component ablations: E=1, D=1, rho=1.
# Each variant trains consistent -> controlled, then evaluates seven benchmarks.
# Host launcher: training runs in syw_opd; evaluation runs in syw_clip2.
# STAGE=train|eval|all (default all); DRY_RUN=true prints commands only.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

stage="${STAGE:-all}"
case "$stage" in train|eval|all) ;; *) echo 'STAGE must be train, eval, or all' >&2; exit 2 ;; esac
case "${DRY_RUN:-false}" in true|false) ;; *) echo 'DRY_RUN must be true or false' >&2; exit 2 ;; esac
(( $# == 0 )) || { echo 'Usage: STAGE=train|eval|all DRY_RUN=true|false bash scripts/run_ejsd_component_ablation_4b.sh' >&2; exit 2; }

student="${STUDENT_MODEL:-/home/sunyw/SK_RAG/models/Qwen3-VL-4B-Thinking}"
opsd_teacher="${OPSD_TEACHER_MODEL:-$student}"
# Use the same 8B teacher checkpoint as the full OPD EJSD baseline.
opd_teacher="${OPD_TEACHER_MODEL:-$repo_root/models/opsd_s8b_stage1_entropy_jsd_merged}"
gpu_ids="${GPU_IDS:-0,1,2,3,4,5,6,7}"
eval_gpu_ids="${EVAL_GPU_IDS:-$gpu_ids}"
results_root="${RESULTS_ROOT:-$repo_root/rag_eval/results/ejsd_components_4b}"
scores_root="${SCORES_ROOT:-$repo_root/rag_eval/evaluation/ejsd_components_4b}"
eval_retrieval="${RETRIEVAL_FILE:-test_mixed_top10_v2.jsonl}"
eval_top_k="${TOP_K:-3}"
eval_datasets='MM-SafetyBench VLGuard MSSBench SPA-VL SIUO-Gen VLSU VLSBench'
train_container="${TRAIN_CONTAINER:-syw_opd}"
eval_container="${EVAL_CONTAINER:-syw_clip2}"
train_overrides=()
# Forward optional manual veRL settings when TRAIN_AUTO_CONFIG=false (e.g. RTX 4090).
for key in TRAINER_GPUS_PER_NODE TEACHER_GPUS_PER_NODE TEACHER_NUM_REPLICAS TEACHER_TP \
  TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH \
  MAX_TOKEN_LEN_PER_GPU TRAIN_MAX_IMAGE_PIXELS ROLLOUT_GPU_MEMORY_UTILIZATION \
  TEACHER_GPU_MEMORY_UTILIZATION PARAM_OFFLOAD OPTIMIZER_OFFLOAD USE_CHUNKED_TOPK \
  CHUNKED_TOPK_CHUNK_SIZE; do
  if [[ -v "$key" ]]; then train_overrides+=(-e "$key=${!key}"); fi
done
read -r -a methods <<< "${METHODS:-opsd opd}"
(( ${#methods[@]} > 0 )) || { echo 'METHODS must contain opsd and/or opd' >&2; exit 2; }
for method in "${methods[@]}"; do
  case "$method" in opsd|opd) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
done

if [[ "${DRY_RUN:-false}" == false && "$stage" != eval ]]; then
  [[ -f "$student/config.json" ]] || {
    echo "Missing 4B student: $student" >&2; exit 2;
  }
  if [[ " ${methods[*]} " == *' opsd '* && ! -f "$opsd_teacher/config.json" ]]; then
    echo 'Missing 4B student or OPSD teacher model config.json' >&2; exit 2;
  fi
  if [[ " ${methods[*]} " == *' opd '* && ! -f "$opd_teacher/config.json" ]]; then
    echo "Missing OPD 8B teacher: $opd_teacher; set OPD_TEACHER_MODEL to the teacher used by full EJSD" >&2; exit 2;
  fi
fi

run() {
  if [[ "${DRY_RUN:-false}" == true ]]; then
    printf '[PLAN]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

for method in "${methods[@]}"; do
  if [[ "$method" == opsd ]]; then teacher="$opsd_teacher"; else teacher="$opd_teacher"; fi
  for ablation in wo_entropy wo_coverage wo_mass_gate; do
    use_entropy=true use_coverage=true use_mass_gate=true
    case "$ablation" in
      wo_entropy) use_entropy=false ;;
      wo_coverage) use_coverage=false ;;
      wo_mass_gate) use_mass_gate=false ;;
    esac
    prefix="${method}_s4b_ejsd_${ablation}"
    stage1="${prefix}_stage1_consistent"
    stage2="${prefix}_stage2_controlled"
    case "$ablation" in
      wo_entropy) setting='E=1' ;;
      wo_coverage) setting='D=1' ;;
      wo_mass_gate) setting='rho=1' ;;
    esac
    echo "[$(date -u +%FT%TZ)] $method $ablation ($setting)"

    if [[ "$stage" == train || "$stage" == all ]]; then
      for phase in 1 2; do
        if (( phase == 1 )); then
          name="$stage1" source=data/category_consistent_abstract.jsonl init='' lr=5e-6
        else
          name="$stage2" source=data/category_controlled_abstract.jsonl init="$stage1" lr=1e-6
        fi
        run docker exec -w "$repo_root" \
          -e "PYTHON_BIN=${TRAIN_PYTHON_BIN:-/opt/syw_opd_venv/bin/python}" \
          -e "AUTO_CONFIG=${TRAIN_AUTO_CONFIG:-true}" "${train_overrides[@]}" \
          -e "CUDA_VISIBLE_DEVICES=$gpu_ids" -e "MODEL_PATH=$student" -e "TEACHER_MODEL_PATH=$teacher" \
          -e "STAGE1_EXPERIMENT_NAME=$init" -e "EXPERIMENT_NAME=$name" \
          -e DISTILLATION_LOSS_MODE=entropy_jsd -e ENTROPY_JSD_NORMALIZE=true \
          -e "ENTROPY_JSD_USE_ENTROPY_MODULATION=$use_entropy" \
          -e "ENTROPY_JSD_USE_COVERAGE_MODULATION=$use_coverage" \
          -e "ENTROPY_JSD_USE_MASS_GATE=$use_mass_gate" \
          -e "TRAIN_DATA_FILE=$source" -e PROMPT_MODE=guided -e ENABLE_THINKING=false \
          -e USE_POLICY_GRADIENT=false -e TEACHER_USE_PRIVILEGED_INFO=true \
          -e "LR=$lr" -e "SEED=${SEED:-42}" -e SAVE_FREQ=100 -e START_STAGE=train -e TRAIN_ONLY=true \
          "$train_container" bash scripts/train_eval_pipeline.sh
      done
      run docker exec -w "$repo_root" -e "PYTHON_BIN=${TRAIN_PYTHON_BIN:-/opt/syw_opd_venv/bin/python}" \
        -e "PYTHONPATH=$repo_root/verl-runtime:$repo_root" \
        "$train_container" bash scripts/latest_adapter.sh "outputs/$stage2"
    fi

    if [[ "$stage" == eval || "$stage" == all ]]; then
      if [[ "${DRY_RUN:-false}" == false && "${EVAL_STAGE:-all}" != generate && -z "${DASHSCOPE_API_KEY:-}" ]]; then
        secret=/home/sunyw/training_free_vl_bench/.secrets/dashscope_api_key
        [[ -r "$secret" ]] || { echo "Missing judge credential: $secret" >&2; exit 2; }
        export DASHSCOPE_API_KEY="$(<"$secret")"
      fi
      run docker exec -w "$repo_root" -e "PYTHON_BIN=${EVAL_PYTHON_BIN:-/opt/conda/envs/qwen3vl/bin/python}" \
        -e "GPU_IDS=$eval_gpu_ids" -e "DATASETS=$eval_datasets" -e "TOP_K=$eval_top_k" \
        -e "RETRIEVAL_FILE=$eval_retrieval" -e ENABLE_THINKING=false -e PROMPT_MODE=guided \
        -e "STAGE=${EVAL_STAGE:-all}" -e "RUN_NAME=$stage2" -e "MODEL_NAME=$stage2" \
        -e "OUTPUT_DIR=$results_root/$stage2" \
        -e "EVAL_OUTPUT_ROOT=$scores_root/$stage2/mssbench_protocol" \
        -e DASHSCOPE_API_KEY "$eval_container" bash scripts/eval_mixed_top10_v2.sh
    fi
  done
done
if [[ "$stage" == eval || "$stage" == all ]]; then
  run python3 scripts/summarize_ejsd_component_ablation_4b.py \
    --scores-root "$scores_root" --output "$repo_root/reports/ejsd_components_4b.tsv"
fi
