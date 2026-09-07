#!/usr/bin/env python3
"""Precompute Qwen3-VL-Embedding Top-K annotations for every test split."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.data import (
    DEFAULT_DATA_ROOT,
    DatasetFiles,
    image_paths_of,
    read_jsonl,
    retrieval_text_of,
    select_datasets,
    write_jsonl,
)
from rag_eval.retrieve import _encode_rows, _slim_example, make_embedder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLASHRAG = Path("/home/sunyw/FlashRAG")
DEFAULT_EMBEDDING_REPO = Path("/home/sunyw/Qwen3-VL-Embedding")
DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_REPO / "models/Qwen3-VL-Embedding-2B"
FLASH_INDEXES = {
    "FigStep": ("FigStep/qwen3vl_embedding/corpus", "FigStep/qwen3vl_embedding/test"),
    "MM-SafetyBench": (
        "MM-SafetyBench/qwen3vl_embedding/corpus",
        "MM-SafetyBench/qwen3vl_embedding/test",
    ),
    "MSSBench": ("MSSBench/qwen3vl_embedding/corpus", "MSSBench/qwen3vl_embedding/test"),
    "VLGuard": (
        "VLGuard/qwen3vl_embedding/mixed_corpus",
        "VLGuard/qwen3vl_embedding/test_start0_limitall",
    ),
}


def normalized_id(dataset: str, row_id: str) -> str:
    if dataset == "FigStep":
        return row_id.removeprefix("figstep:")
    if dataset == "VLGuard":
        return row_id.removeprefix("vlguard:train:").removeprefix("vlguard:")
    return row_id


def load_flash_vectors(
    files: DatasetFiles, flashrag_root: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    relative_dirs = FLASH_INDEXES.get(files.name)
    if relative_dirs is None:
        return None
    indexed: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for relative in relative_dirs:
        directory = flashrag_root / "indexes" / relative
        metadata_path = directory / "metadata.jsonl"
        vectors_path = directory / "fused.npy"
        if not metadata_path.is_file() or not vectors_path.is_file():
            return None
        metadata = read_jsonl(metadata_path)
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.shape[0] != len(metadata):
            raise ValueError(f"Metadata/vector count mismatch in {directory}")
        for row, vector in zip(metadata, vectors):
            row_id = str(row["id"])
            if row_id in indexed:
                raise ValueError(f"Duplicate FlashRAG ID for {files.name}: {row_id}")
            indexed[row_id] = (row, vector)

    corpus = read_jsonl(files.retrieval)
    test = read_jsonl(files.test)
    ordered: list[np.ndarray] = []
    for current in corpus + test:
        current_id = normalized_id(files.name, str(current["id"]))
        if current_id not in indexed:
            raise ValueError(f"FlashRAG cache lacks {files.name} ID {current_id}")
        cached_row, vector = indexed[current_id]
        cached_text = cached_row.get("question") or cached_row.get("text")
        if retrieval_text_of(current).strip() != str(cached_text).strip():
            raise ValueError(f"FlashRAG text mismatch for {files.name} ID {current_id}")
        ordered.append(np.asarray(vector, dtype=np.float32))
    matrix = np.stack(ordered)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix[: len(corpus)], matrix[len(corpus) :]


def rank_and_write(
    files: DatasetFiles,
    corpus_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    output: Path,
    embedding_model: Path,
    top_k: int,
    search_batch_size: int,
    embedding_source: str,
) -> None:
    corpus = read_jsonl(files.retrieval)
    test = read_jsonl(files.test)
    if not 1 <= top_k <= len(corpus):
        raise ValueError(f"Invalid Top-K {top_k} for {files.name}")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(test), search_batch_size):
        scores = test_embeddings[start : start + search_batch_size] @ corpus_embeddings.T
        candidates = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        values = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-values, axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        for offset, (indices, ranked_scores) in enumerate(zip(candidates, values)):
            query = dict(test[start + offset])
            query["retrieval"] = [
                _slim_example(corpus[int(index)], float(score))
                for index, score in zip(indices, ranked_scores)
            ]
            query["retrieval_config"] = {
                "method": "qwen3-vl-embedding",
                "model": str(embedding_model),
                "representation": "image_and_metadata_question",
                "corpus": str(files.retrieval),
                "top_k": top_k,
                "similarity": "cosine",
                "embedding_source": embedding_source,
            }
            rows.append(query)
    write_jsonl(output, rows)
    print(f"[{files.name}] wrote {len(rows)} rows to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--flashrag-root", type=Path, default=DEFAULT_FLASHRAG)
    parser.add_argument("--embedding-repo", type=Path, default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--no-reuse-flashrag", action="store_true")
    args = parser.parse_args()

    datasets = select_datasets(args.datasets, args.data_root)
    pending: list[DatasetFiles] = []
    for files in datasets:
        vectors = None if args.no_reuse_flashrag else load_flash_vectors(files, args.flashrag_root)
        if vectors is None:
            pending.append(files)
            continue
        rank_and_write(
            files, *vectors,
            output=files.test.parent / f"test_qwen3vl_embedding_top{args.top_k}.jsonl",
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            search_batch_size=args.search_batch_size,
            embedding_source="reused_verified_flashrag_cache",
        )

    if not pending:
        return
    embedder = make_embedder(
        args.embedding_repo, args.embedding_model, args.max_length, args.max_pixels
    )
    for files in pending:
        print(f"[{files.name}] computing embeddings", flush=True)
        corpus = read_jsonl(files.retrieval)
        test = read_jsonl(files.test)
        corpus_embeddings = _encode_rows(embedder, corpus, args.batch_size)
        test_embeddings = _encode_rows(embedder, test, args.batch_size)
        rank_and_write(
            files, corpus_embeddings, test_embeddings,
            output=files.test.parent / f"test_qwen3vl_embedding_top{args.top_k}.jsonl",
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            search_batch_size=args.search_batch_size,
            embedding_source="computed_for_rag_eval",
        )
        gc.collect()


if __name__ == "__main__":
    main()
