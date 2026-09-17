# Evaluation data

This directory is a minimal, consolidated copy of the evaluated portions of
`/home/sunyw/safety_alignment_retrieval`.

Each dataset directory contains:

- `retrieval_qwen3.7_plus_merged.jsonl`: retrieval corpus with the final
  Qwen3.7-Plus `natural_response` annotation.
- `test.jsonl`: held-out evaluation queries.
- `test_qwen3vl_embedding_top10.jsonl`: held-out queries annotated with the ten
  highest-scoring retrieval cases, which can be sliced to the desired Top-K at
  evaluation time.
- `split_summary.json`: split counts and provenance.

`split_index.json` records the global deterministic split. Intermediate files
(`retrieval.jsonl` and `retrieval_qwen3.7_plus.jsonl`) are omitted because the
evaluation pipeline only consumes the merged retrieval annotation.

Images are organized as `images/<dataset>/`. There are 6,855 dataset image
entries representing 6,688 unique source images (about 1.31 GB); the 167 images
shared by SIUO-Gen and SIUO-MCQA use hard links and do not consume duplicate
payload space. `image_manifest.json` records the dataset, original path, local
path, and file size. The JSONL `images` fields point exclusively to these local
copies, so evaluation no longer reads image files from the source directories.

The image payload is intentionally ignored by Git because of its size. To
rebuild it from source paths before they are rewritten, run:

```bash
python -m rag_eval.materialize_images
```

Source snapshot: `/home/sunyw/safety_alignment_retrieval`, 2026-09-07.
