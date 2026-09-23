# SK-RAG retrieval tool

This directory is a standalone RAG workflow. It does not import or require any
other RAG framework and can be copied to a remote server together with its data
and model. The runtime contract is `Config -> Dataset -> Retriever -> Prompt /
Generator -> Evaluator -> Pipeline`; retriever training is intentionally outside
the package.

The package includes configuration merging, dataset/item containers, split and
merge helpers, dense retrieval, retrieval-result caching, prompt construction,
optional Transformers generation, evaluation helpers, and a sequential RAG
pipeline. All components are independently importable from `rag_tool`.

## Build an index

The corpus is JSONL, JSON, or Parquet. JSONL rows may use `contents`, `text`,
`prompt`, `question`, or `metadata.question`; all original fields are retained.

```bash
python -m rag_tool.build_index \
  --corpus rag_eval/data/MM-SafetyBench/retrieval_qwen3.7_plus_merged_with_retrieval_and_test_knowledge.jsonl \
  --output rag_tool/indexes/mm_safetybench \
  --model /path/to/Qwen3-VL-Embedding-2B \
  --backend qwen3vl
```

If the Qwen embedding implementation is stored separately from the model,
pass its local directory explicitly with `--code-path /path/to/qwen-code`.
There is no hard-coded dependency on any local RAG repository.

The output directory contains `corpus.jsonl`, `embeddings.npy`, optionally
`index.faiss`, and `manifest.json`. When FAISS is unavailable, exact NumPy
inner-product search is used automatically.

## Retrieve

```bash
python -m rag_tool.retrieve \
  --index rag_tool/indexes/mm_safetybench \
  --query-file rag_eval/data/MM-SafetyBench/test.jsonl \
  --output rag_tool/results/mm_safetybench_test.jsonl \
  --top-k 10
```

For a single query, use `--query`. Query files use the same row format as the
corpus. The output preserves each query row and adds `retrieval` plus a
`retrieval_config` object, matching the existing RAG evaluation format.

`--backend sentence-transformers` uses a SentenceTransformers model;
`--backend transformers` uses mean-pooled HuggingFace output. Qwen3-VL is the
default when the supplied model directory contains the Qwen embedding code.

For Qwen3-VL, the runtime must provide the Qwen3-VL Transformers architecture.
FAISS is optional because the tool has a NumPy fallback.
