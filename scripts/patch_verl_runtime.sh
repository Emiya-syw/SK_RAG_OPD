#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_source_dir="${VERL_SOURCE_DIR:-$(dirname "${project_root}")/verl-runtime}"
patch_file="${project_root}/patches/verl-qwen3-vl-opd-teacher-alignment.patch"

if [[ ! -d "${verl_source_dir}/.git" ]]; then
  echo "veRL source repository does not exist: ${verl_source_dir}" >&2
  exit 2
fi

if git -C "${verl_source_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "veRL Qwen3-VL OPD teacher alignment patch is already applied."
elif git -C "${verl_source_dir}" apply --check "${patch_file}"; then
  git -C "${verl_source_dir}" apply "${patch_file}"
  echo "Applied veRL Qwen3-VL OPD teacher alignment patch."
else
  echo "Cannot apply ${patch_file}; veRL source differs from the pinned runtime." >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi
"${python_bin}" -m py_compile \
  "${verl_source_dir}/verl/experimental/agent_loop/agent_loop.py" \
  "${verl_source_dir}/verl/experimental/teacher_loop/teacher_manager.py"
