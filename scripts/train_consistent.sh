#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 数据入口参数：$1 为源 JSONL，$2 为输出目录；不传时使用下面的默认值。
# 核心训练参数及可选值统一写在 scripts/run_opd.sh 顶部“训练参数配置区”。
source_file="${1:-data/category_consistent.jsonl}"
output_dir="${2:-outputs/consistent}"
# 可选：指定转换后 veRL JSONL/Parquet 路径；默认保存在当前阶段输出目录。
prepared_file="${VERL_DATA_FILE:-${output_dir}/train.verl.jsonl}"
# Python 可执行文件；默认 python3，也可设为 conda 环境中的绝对路径。
python_bin="${PYTHON_BIN:-python3}"
# 可选：SKIP_IMAGE_CHECK=false 会在训练前检查图片；true 跳过检查、准备更快但错误会延迟到 rollout。
prepare_args=()
if [[ "${SKIP_IMAGE_CHECK:-false}" == "true" ]]; then prepare_args+=(--skip-image-check); fi

"${python_bin}" scripts/prepare_verl_data.py "${source_file}" "${prepared_file}" "${prepare_args[@]}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-consistent}" scripts/run_opd.sh "${prepared_file}" "${output_dir}"
