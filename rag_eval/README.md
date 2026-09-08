# RAG evaluation

This directory evaluates an SK-RAG-OPD checkpoint with the consolidated split
files under `rag_eval/data`. It supports BeaverTails-V,
FigStep, MM-SafetyBench, MSSBench, SIUO-Gen, SIUO-MCQA, and VLGuard.

The pipeline has three stages:

1. `validate`: check IDs, split overlap, annotated retrieval answers, and image paths.
2. `retrieve`: use Qwen3-VL-Embedding-2B to retrieve top-k examples from each
   dataset's retrieval split for every item in its test split.
3. `generate`: supply the current image/question and retrieved image/question/answer
   examples to the base model plus the optional OPD LoRA adapter.

Precomputed Top-3 annotations are stored as
`data/<dataset>/test_qwen3vl_embedding_top3.jsonl`. Therefore `--stage generate`
does not load the embedding model or perform retrieval.

The enhanced `retrieval_qwen3.7_plus_merged.jsonl` file is selected first. Its
`natural_response` is used as the demonstration answer. Test metadata is retained
in the output for later scoring, but reference answers inside that metadata are
never included in the model prompt.

The local JSONL annotations and all images are stored under `rag_eval/data`,
organized as `images/<dataset>/`. The 6,855 dataset entries represent 6,688
unique source images; shared SIUO images use hard links. The JSONL files point to
these local copies, so evaluation no longer depends on source benchmark
directories. The entire `rag_eval/data` directory is ignored by Git; it remains
available in the working tree but is not included in GitHub or source archives.

## Quick checks

Validation does not load a model:

```bash
cd /home/sunyw/SK_RAG_OPD
STAGE=validate bash rag_eval/run_rag_eval.sh
```

Run a small smoke test on two datasets:

```bash
cd /home/sunyw/SK_RAG_OPD
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/path/to/conda/env/bin/python \
ADAPTER_PATH=/path/to/opd/output \
DATASETS="SIUO-Gen VLGuard" \
LIMIT=5 \
OUTPUT_DIR=rag_eval/results/smoke \
 bash rag_eval/run_rag_eval.sh
```

Run all test samples after the smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/path/to/conda/env/bin/python \
ADAPTER_PATH=/path/to/opd/output \
OUTPUT_DIR=rag_eval/results/opd_all \
 bash rag_eval/run_rag_eval.sh
```

Leave `ADAPTER_PATH` empty to evaluate the base model. Embeddings are cached in
`rag_eval/cache`, so subsequent runs with the same data and embedding settings do
not recompute them. Set `REBUILD_CACHE=true` to force rebuilding.

`run_rag_eval.sh` is the supported entry point. With `STAGE=generate` (the
default), it uses the already generated `test_qwen3vl_embedding_top3.jsonl`
files when available and does not load the embedding model. Use
`STAGE=retrieve` or `STAGE=all` only when you explicitly want to recompute
retrieval.

QA generation supports batching. The default is `1` for broad GPU compatibility;
set `GENERATION_BATCH_SIZE=2` or `4` in `run_rag_eval.sh` when the model and
image workload fit in GPU memory. A batch is written to the JSONL output only
after all of its samples finish, and ID-based resume remains enabled.

For separate retrieval and generation GPUs or environments, run `STAGE=retrieve`
first, then run the same command with `STAGE=generate`. The output layout is:

```text
rag_eval/results/<run>/<dataset>/retrieval.jsonl
rag_eval/results/<run>/<dataset>/answers.jsonl
```

`retrieval.jsonl` records every retrieved ID, cosine score, image, question, and
demonstration answer. `answers.jsonl` additionally contains `raw_prediction` and
`prediction`, and is appended safely with ID-based resume enabled by default.

## Online safety evaluation

`evaluate_online.py` reuses the prompts, API protocol, parsers, and summaries
from `/home/sunyw/SK_RAG/scripts/eval/evaluate_safety_alignment_outputs.py`.
It reads `answers.jsonl` and writes `judgments.jsonl` plus `summary.json` per
dataset:

```bash
cd /home/sunyw/SK_RAG_OPD
export DASHSCOPE_API_KEY='your-key'
INPUT_ROOT=rag_eval/results/test_V1 \
EVAL_OUTPUT_ROOT=rag_eval/evaluation/test_V1 \
bash rag_eval/run_online_eval.sh
```

The source evaluator's official local protocols remain enabled by default for
VLGuard (keyword ASR) and SIUO-MCQA (option parser). Set
`ONLINE_VLGUARD=true` to use its online general judge for VLGuard, or
`HARMFUL_REQUEST_EVAL=true` to use the six-category harmful-request prompt.
Judgments resume from valid existing rows and use `JUDGE_WORKERS` concurrent
requests.

The reported benchmark metrics are: ASR for BeaverTails-V, FigStep,
MM-SafetyBench, and SIUO-Gen; VLGuard reports ASR on unsafe samples and FRR
(false refusal rate) on safe samples; MSSBench reports ASR and FRR separately
for `chat` and `embodied`; SIUO-MCQA reports option accuracy.
