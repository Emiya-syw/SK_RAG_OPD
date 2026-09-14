#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_source_dir="${VERL_SOURCE_DIR:-${project_root}/verl-runtime}"
patch_files=(
  "${project_root}/patches/verl-qwen3-vl-opd-teacher-alignment.patch"
  "${project_root}/patches/verl-opd-synthetic-padding-teacher-fields.patch"
  "${project_root}/patches/verl-opd-jagged-teacher-padding.patch"
)

# The runtime may be either a standalone veRL checkout (with its own .git)
# or a vendored source tree tracked by this project.  The latter deliberately
# has no nested .git directory, so validate the source files instead of
# requiring a second repository.
if [[ ! -d "${verl_source_dir}/verl" ]]; then
  echo "veRL source repository does not exist: ${verl_source_dir}" >&2
  exit 2
fi

git_root="${verl_source_dir}"
git_directory_args=()
if [[ ! -d "${verl_source_dir}/.git" ]]; then
  # A vendored runtime must live below the project repository so that git
  # apply can update the tracked files without treating the patch paths as
  # absolute filesystem paths.
  case "${verl_source_dir}/" in
    "${project_root}/"*)
      git_root="${project_root}"
      git_directory_args=(--directory="${verl_source_dir#${project_root}/}")
      ;;
    *)
      echo "veRL source has no .git directory and is outside the project: ${verl_source_dir}" >&2
      exit 2
      ;;
  esac
fi

for patch_file in "${patch_files[@]}"; do
  if git -C "${git_root}" apply "${git_directory_args[@]}" --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "veRL patch is already applied: $(basename "${patch_file}")"
  elif git -C "${git_root}" apply "${git_directory_args[@]}" --check "${patch_file}"; then
    git -C "${git_root}" apply "${git_directory_args[@]}" "${patch_file}"
    echo "Applied veRL patch: $(basename "${patch_file}")"
  else
    echo "Cannot apply ${patch_file}; veRL source differs from the pinned runtime." >&2
    exit 2
  fi
done

python_bin="${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi
"${python_bin}" -m py_compile \
  "${verl_source_dir}/verl/experimental/agent_loop/agent_loop.py" \
  "${verl_source_dir}/verl/experimental/teacher_loop/teacher_manager.py" \
  "${verl_source_dir}/verl/trainer/ppo/padding_utils.py" \
  "${verl_source_dir}/verl/workers/utils/padding.py"
