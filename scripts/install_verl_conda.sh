#!/usr/bin/env bash
set -euo pipefail

# Install the tested veRL FSDP + vLLM runtime directly into the currently
# activated Python 3.12 Conda environment. veRL's lock selects prebuilt
# CUDA 13 / Torch 2.11 / FlashAttention wheels, so no local CUDA build is used.

readonly VERL_COMMIT="10db40d0da4d59150bb389960b77585f81a89b8d"
readonly VERL_REPOSITORY="https://github.com/verl-project/verl.git"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No Conda environment is active. Run: conda activate YOUR_ENV" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "Refusing to install the training runtime into the Conda base environment." >&2
  exit 2
fi

python_bin="${CONDA_PREFIX}/bin/python"
python_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.12" ]]; then
  echo "The active Conda environment uses Python ${python_version}; veRL's locked GPU runtime requires Python 3.12." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_source_dir="${VERL_SOURCE_DIR:-${project_root}/verl-runtime}"
uv_cache_dir="${UV_CACHE_DIR:-${project_root}/.cache/uv-sk-rag-opd}"
mkdir -p "${uv_cache_dir}"

echo "Conda environment : ${CONDA_PREFIX}"
echo "Python            : $("${python_bin}" --version 2>&1)"
echo "veRL source       : ${verl_source_dir}"
echo "veRL commit       : ${VERL_COMMIT}"
echo "uv cache          : ${uv_cache_dir}"

if ! "${python_bin}" -m pip --version >/dev/null 2>&1; then
  conda install -y --prefix "${CONDA_PREFIX}" pip
fi
"${python_bin}" -m pip install --upgrade "uv"

if [[ -d "${verl_source_dir}/.git" ]]; then
  git -C "${verl_source_dir}" fetch origin "${VERL_COMMIT}" --depth=1
else
  if [[ -e "${verl_source_dir}" ]]; then
    echo "${verl_source_dir} exists but is not a Git repository; set VERL_SOURCE_DIR to another path." >&2
    exit 2
  fi
  git clone --filter=blob:none --no-checkout "${VERL_REPOSITORY}" "${verl_source_dir}"
  git -C "${verl_source_dir}" fetch origin "${VERL_COMMIT}" --depth=1
fi
git -C "${verl_source_dir}" checkout --detach "${VERL_COMMIT}"

# Qwen3-VL may expand visual placeholder runs to a different prompt width in
# vLLM than in the agent-loop HF processor. Preserve and verify the response
# suffix before aligning teacher logprobs to the student tensor width.
VERL_SOURCE_DIR="${verl_source_dir}" PYTHON_BIN="${python_bin}" \
  "${project_root}/scripts/patch_verl_runtime.sh"

# Point project sync at the active Conda prefix. --inexact preserves Conda's
# management packages while installing the locked veRL backend versions.
UV_CACHE_DIR="${uv_cache_dir}" \
UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}" \
  "${CONDA_PREFIX}/bin/uv" sync \
    --project "${verl_source_dir}" \
    --python "${python_bin}" \
    --extra fsdp \
    --extra vllm \
    --frozen \
    --inexact

# A sync interrupted by a full disk can leave dist-info present while package
# files are incomplete. Repair only the affected packages from the lock file.
if ! "${python_bin}" -c 'import sympy.core; from vllm import LLM' >/dev/null 2>&1; then
  echo "Repairing incomplete SymPy/vLLM package files..."
  UV_CACHE_DIR="${uv_cache_dir}" \
  UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}" \
    "${CONDA_PREFIX}/bin/uv" sync \
      --project "${verl_source_dir}" \
      --python "${python_bin}" \
      --extra fsdp \
      --extra vllm \
      --frozen \
      --inexact \
      --reinstall-package sympy \
      --reinstall-package vllm
fi

if ! "${python_bin}" -c 'import flashinfer, flashinfer_cubin; assert flashinfer_cubin.__version__' >/dev/null 2>&1; then
  echo "Repairing incomplete FlashInfer package files..."
  UV_CACHE_DIR="${uv_cache_dir}" \
  UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}" \
    "${CONDA_PREFIX}/bin/uv" sync \
      --project "${verl_source_dir}" \
      --python "${python_bin}" \
      --extra fsdp \
      --extra vllm \
      --frozen \
      --inexact \
      --reinstall-package flashinfer-python \
      --reinstall-package flashinfer-cubin
fi

"${python_bin}" - <<'PY'
import sys
from importlib.metadata import version

import flash_attn
import flashinfer
import flashinfer_cubin
import sympy
import torch
import transfer_queue
import transformers
import verl
import vllm
from transformers.utils import is_flash_attn_2_available

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert is_flash_attn_2_available(), "Transformers cannot use FlashAttention2"
assert flashinfer_cubin.__version__ == version("flashinfer-cubin")

print("\nveRL Conda runtime is ready")
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("vLLM:", version("vllm"))
print("Transformers:", transformers.__version__)
print("FlashAttention:", flash_attn.__version__)
print("FlashInfer:", version("flashinfer-python"))
print("FlashInfer cubin:", flashinfer_cubin.__version__)
print("SymPy:", sympy.__version__)
print("TransferQueue:", transfer_queue.__file__)
print("veRL:", verl.__file__)
PY

cat <<EOF

Use this environment for training:
  conda activate ${CONDA_DEFAULT_ENV}
  cd ${project_root}
  export PYTHON_BIN=${python_bin}
  ATTN_IMPLEMENTATION=flash_attention_2 bash scripts/train_consistent.sh
EOF
