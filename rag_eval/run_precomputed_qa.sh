#!/usr/bin/env bash
set -euo pipefail

# One-click QA evaluation using the precomputed Qwen3-VL-Embedding Top-K files.
# Edit the values in this block only when changing the model or evaluation scope.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Fixed defaults: direct `bash rag_eval/run_precomputed_qa.sh` is sufficient.
python_bin="/opt/conda/envs/qwen3vl/bin/python"
base_model="/home/sunyw/SK_RAG/models/Qwen3-VL-2B-Instruct"
datasets="all"
top_k=3
limit=0
output_dir="${repo_root}/rag_eval/results/precomputed_qa"
max_new_tokens=256
dtype="bfloat16"
attn_implementation="eager"

# Automatically use the newest local OPD adapter when one exists. Leave empty
# to evaluate the base model. Set this explicitly to pin a checkpoint.
adapter=""
for candidate_root in "${repo_root}/outputs/controlled" "${repo_root}/outputs/consistent"; do
  if [[ -d "${candidate_root}" ]]; then
    candidate="$(find "${candidate_root}" -type f -name adapter_config.json -printf '%h\n' | sort -V | tail -1)"
    if [[ -n "${candidate}" ]]; then adapter="${candidate}"; fi
  fi
done

args=(
  --stage generate
  --datasets ${datasets}
  --data-root "${repo_root}/rag_eval/data"
  --output-dir "${output_dir}"
  --top-k "${top_k}"
  --base-model "${base_model}"
  --max-image-pixels 262144
  --max-new-tokens "${max_new_tokens}"
  --dtype "${dtype}"
  --attn-implementation "${attn_implementation}"
  --limit "${limit}"
)

if [[ -n "${adapter}" ]]; then
  args+=(--adapter "${adapter}")
fi

echo "Using precomputed Top-${top_k} files from ${repo_root}/rag_eval/data" >&2
echo "Model: ${base_model}" >&2
if [[ -n "${adapter}" ]]; then echo "Adapter: ${adapter}" >&2; fi

"${python_bin}" -m rag_eval.run_rag_eval "${args[@]}"
