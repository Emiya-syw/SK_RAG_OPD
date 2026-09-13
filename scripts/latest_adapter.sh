#!/usr/bin/env bash
set -euo pipefail

checkpoint_root="${1:?usage: scripts/latest_adapter.sh CHECKPOINT_ROOT}"
latest_step="$(find "${checkpoint_root}" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V | tail -n 1)"
if [[ -z "${latest_step}" || ! -d "${latest_step}/actor" ]]; then
  echo "No veRL actor checkpoint found below ${checkpoint_root}" >&2
  exit 1
fi

adapter_config="$({ find "${latest_step}/actor" -type f -name adapter_config.json -printf '%T@ %p\n' 2>/dev/null || true; } | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -n "${adapter_config}" ]]; then
  dirname "${adapter_config}"
  exit 0
fi

python_bin="${PYTHON_BIN:-python3}"
target_dir="${latest_step}/actor/huggingface_merged"
"${python_bin}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${latest_step}/actor" \
  --target_dir "${target_dir}"

adapter_dir="${target_dir}/lora_adapter"
if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
  echo "veRL model merger did not produce a LoRA adapter in ${adapter_dir}" >&2
  exit 1
fi
printf '%s\n' "${adapter_dir}"
