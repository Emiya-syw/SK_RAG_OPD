#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 数据入口参数：$1 为源 JSONL，$2 为输出目录。
# 交互式运行且未提供 $2 时，会询问权重目录名；直接回车使用当前日期和时间。
# 核心训练参数及可选值统一写在 scripts/run_opd.sh 顶部“训练参数配置区”。
source_file="${1:-data/category_consistent.jsonl}"
if (( $# >= 2 )); then
  output_dir="$2"
elif [[ -t 0 && -t 1 && "${PROMPT_OUTPUT_DIR:-true}" == "true" ]]; then
  default_output_name="$(date +%Y-%m-%d_%H-%M)"
  read -r -p "请输入保存权重的文件夹名称（直接回车使用 ${default_output_name}）: " output_name
  output_name="${output_name:-${default_output_name}}"
  if [[ "${output_name}" == "." || "${output_name}" == ".." || "${output_name}" == */* ]]; then
    printf '错误：文件夹名称不能为 .、.. 或包含路径分隔符：%s\n' "${output_name}" >&2
    exit 2
  fi
  output_dir="outputs/${output_name}"
else
  # 无交互终端（例如后台任务/作业系统）保持原有默认路径。
  output_dir="outputs/consistent"
fi
# 可选：指定转换后 veRL JSONL/Parquet 路径；默认保存在当前阶段输出目录。
prepared_file="${VERL_DATA_FILE:-${output_dir}/train.verl.jsonl}"
# Python 可执行文件；默认 python3，也可设为 conda 环境中的绝对路径。
python_bin="${PYTHON_BIN:-python3}"
# 可选：SKIP_IMAGE_CHECK=false 会在训练前检查图片；true 跳过检查、准备更快但错误会延迟到 rollout。
prepare_args=()
if [[ "${SKIP_IMAGE_CHECK:-false}" == "true" ]]; then prepare_args+=(--skip-image-check); fi

"${python_bin}" scripts/prepare_verl_data.py "${source_file}" "${prepared_file}" "${prepare_args[@]}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-consistent}" scripts/run_opd.sh "${prepared_file}" "${output_dir}"
