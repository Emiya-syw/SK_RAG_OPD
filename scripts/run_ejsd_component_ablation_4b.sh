#!/usr/bin/env bash
# 4B OPSD/OPD Entropy JSD component ablations: E=1, D=1, rho=1.
# Each variant trains consistent -> controlled, then evaluates seven benchmarks.
# Run in the same 8xA100 conda environment as run_2b_4b_controlled_stage1.sh.
# STAGE=train|eval|all (default all); DRY_RUN=true prints commands only.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

stage="${STAGE:-all}"
case "$stage" in train|eval|all) ;; *) echo 'STAGE must be train, eval, or all' >&2; exit 2 ;; esac
case "${DRY_RUN:-false}" in true|false) ;; *) echo 'DRY_RUN must be true or false' >&2; exit 2 ;; esac
(( $# == 0 )) || { echo 'Usage: STAGE=train|eval|all DRY_RUN=true|false bash scripts/run_ejsd_component_ablation_4b.sh' >&2; exit 2; }

student="${STUDENT_MODEL:-$repo_root/models/Qwen3-VL-4B-Thinking}"
opsd_teacher="${OPSD_TEACHER_MODEL:-$student}"
# Use the same 8B teacher checkpoint as the full OPD EJSD baseline.
opd_teacher="${OPD_TEACHER_MODEL:-$repo_root/models/opsd_s8b_stage1_entropy_jsd_merged}"
gpu_ids="${GPU_IDS:-0,1,2,3,4,5,6,7}"
train_python_bin="${TRAIN_PYTHON_BIN:-${PYTHON_BIN:-/opt/conda/envs/sk_rag_opd_verl/bin/python}}"
eval_python_bin="${EVAL_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
results_root="${RESULTS_ROOT:-$repo_root/rag_eval/results/ejsd_components_4b}"
scores_root="${SCORES_ROOT:-$repo_root/rag_eval/evaluation/ejsd_components_4b}"
eval_retrieval="${RETRIEVAL_FILE:-test_mixed_top10_v2.jsonl}"
eval_top_k="${TOP_K:-3}"
eval_datasets='MM-SafetyBench VLGuard MSSBench SPA-VL SIUO-Gen VLSU VLSBench'
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

selected_method() {
  [[ " ${methods[*]} " == *" $1 "* ]]
}

run_variant() {
  local method="$1" ablation="$2" teacher="$3"
  local use_entropy="$4" use_coverage="$5" use_mass_gate="$6" setting="$7"
  local prefix="${method}_s4b_ejsd_${ablation}"
  local stage1="${prefix}_stage1_consistent" stage2="${prefix}_stage2_controlled"
  local phase name source init lr
  echo "[$(date -u +%FT%TZ)] $method $ablation ($setting)"

  if [[ "$stage" == train || "$stage" == all ]]; then
    for phase in 1 2; do
      if (( phase == 1 )); then
        name="$stage1" source=data/category_consistent_abstract.jsonl init='' lr=5e-6
      else
        name="$stage2" source=data/category_controlled_abstract.jsonl init="$stage1" lr=1e-6
      fi
      run env PYTHON_BIN="$train_python_bin" CUDA_VISIBLE_DEVICES="$gpu_ids" \
        MODEL_PATH="$student" TEACHER_MODEL_PATH="$teacher" \
        STAGE1_EXPERIMENT_NAME="$init" EXPERIMENT_NAME="$name" \
        DISTILLATION_LOSS_MODE=entropy_jsd ENTROPY_JSD_NORMALIZE=true \
        ENTROPY_JSD_USE_ENTROPY_MODULATION="$use_entropy" \
        ENTROPY_JSD_USE_COVERAGE_MODULATION="$use_coverage" \
        ENTROPY_JSD_USE_MASS_GATE="$use_mass_gate" \
        TRAIN_DATA_FILE="$source" PROMPT_MODE=guided ENABLE_THINKING=false \
        USE_POLICY_GRADIENT=false TEACHER_USE_PRIVILEGED_INFO=true \
        LR="$lr" SEED="${SEED:-42}" SAVE_FREQ=100 START_STAGE=train TRAIN_ONLY=true \
        bash scripts/train_eval_pipeline.sh
    done
    run env PYTHON_BIN="$train_python_bin" PYTHONPATH="$repo_root/verl-runtime:$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
      bash scripts/latest_adapter.sh "outputs/$stage2"
  fi

  if [[ "$stage" == eval || "$stage" == all ]]; then
    if [[ "${DRY_RUN:-false}" == false && "${EVAL_STAGE:-all}" != generate && -z "${DASHSCOPE_API_KEY:-}" ]]; then
      echo 'Set DASHSCOPE_API_KEY for judging, or use EVAL_STAGE=generate' >&2
      exit 2
    fi
    run env PYTHON_BIN="$eval_python_bin" GPU_IDS="$gpu_ids" DATASETS="$eval_datasets" \
      TOP_K="$eval_top_k" RETRIEVAL_FILE="$eval_retrieval" \
      ENABLE_THINKING=false PROMPT_MODE=guided STAGE="${EVAL_STAGE:-all}" \
      RUN_NAME="$stage2" MODEL_NAME="$stage2" \
      OUTPUT_DIR="$results_root/$stage2" \
      EVAL_OUTPUT_ROOT="$scores_root/$stage2/mssbench_protocol" \
      bash scripts/eval_mixed_top10_v2.sh
  fi
}

# ===== 4B OPSD / EJSD w/o entropy modulation / E=1 =====
if selected_method opsd; then
  run_variant opsd wo_entropy "$opsd_teacher" false true true 'E=1'
fi

# ===== 4B OPSD / EJSD w/o coverage modulation / D=1 =====
if selected_method opsd; then
  run_variant opsd wo_coverage "$opsd_teacher" true false true 'D=1'
fi

# ===== 4B OPSD / EJSD w/o mass gate / rho=1 =====
if selected_method opsd; then
  run_variant opsd wo_mass_gate "$opsd_teacher" true true false 'rho=1'
fi

# ===== 4B OPD / EJSD w/o entropy modulation / E=1 =====
if selected_method opd; then
  run_variant opd wo_entropy "$opd_teacher" false true true 'E=1'
fi

# ===== 4B OPD / EJSD w/o coverage modulation / D=1 =====
if selected_method opd; then
  run_variant opd wo_coverage "$opd_teacher" true false true 'D=1'
fi

# ===== 4B OPD / EJSD w/o mass gate / rho=1 =====
if selected_method opd; then
  run_variant opd wo_mass_gate "$opd_teacher" true true false 'rho=1'
fi
if [[ "$stage" == eval || "$stage" == all ]]; then
  run python3 scripts/summarize_ejsd_component_ablation_4b.py \
    --scores-root "$scores_root" --output "$repo_root/reports/ejsd_components_4b.tsv"
fi
