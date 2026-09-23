#!/usr/bin/env python3
"""Build real-label-category Top-10 retrieval files with rag_tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_eval.data import DEFAULT_DATA_ROOT, DatasetFiles, read_jsonl, write_jsonl
from rag_eval.retrieve import _slim_example
from rag_tool.common import prepare_query_rows, prepare_rows
from rag_tool.encoders import Encoder
from rag_tool.index import DenseIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_REPO = Path("/home/sunyw/Qwen3-VL-Embedding")
DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_REPO / "models/Qwen3-VL-Embedding-8B"
DEFAULT_RETRIEVAL_NAME = "retrieval_qwen3.7_plus_merged_with_retrieval_and_test_knowledge.jsonl"
DEFAULT_OUTPUT_NAME = "test_real_label_category_top10.jsonl"


def dataset_files(data_root: Path, names: list[str], retrieval_name: str) -> list[DatasetFiles]:
    files = []
    for name in names:
        directory = data_root / name
        retrieval = directory / retrieval_name
        test = directory / "test.jsonl"
        if not retrieval.is_file():
            raise FileNotFoundError(retrieval)
        if not test.is_file():
            raise FileNotFoundError(test)
        files.append(DatasetFiles(name=name, retrieval=retrieval, test=test))
    return files


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or not (path / "embeddings.npy").is_file():
        return False
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    return all(actual.get(key) == value for key, value in expected.items())


def build_or_load_index(
    encoder: Encoder,
    files: DatasetFiles,
    index_root: Path,
    model_path: Path,
    embedding_repo: Path,
    batch_size: int,
    max_length: int,
    max_pixels: int,
    rebuild_index: bool,
) -> tuple[DenseIndex, list[dict[str, Any]]]:
    corpus = read_jsonl(files.retrieval)
    prepared = prepare_rows(corpus)
    index_dir = index_root / files.name
    expected_manifest = {
        "corpus": str(files.retrieval.resolve()),
        "rows": len(corpus),
        "ids": [str(row["id"]) for row in prepared],
        "backend": "qwen3vl",
        "model": str(model_path.resolve()),
        "code_path": str(embedding_repo.resolve()),
        "max_length": max_length,
        "max_pixels": max_pixels,
        "metric": "inner_product_on_l2_normalized_vectors",
    }
    if not rebuild_index and manifest_matches(index_dir, expected_manifest):
        print(f"[{files.name}] using cached rag_tool index {index_dir}", flush=True)
        return DenseIndex.load(index_dir), corpus

    print(f"[{files.name}] encoding retrieval corpus", flush=True)
    vectors = encoder.encode(prepared, batch_size)
    index = DenseIndex(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    storage = index.save(index_dir)
    manifest = {**expected_manifest, "storage": storage}
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(index_dir / "corpus.jsonl", corpus)
    return index, corpus


def retrieve_dataset(
    encoder: Encoder,
    files: DatasetFiles,
    index: DenseIndex,
    corpus: list[dict[str, Any]],
    output_name: str,
    model_path: Path,
    top_k: int,
    batch_size: int,
    query_batch_size: int,
    limit: int,
) -> None:
    test = read_jsonl(files.test)
    if limit:
        test = test[:limit]
    if top_k < 1 or top_k > len(corpus):
        raise ValueError(f"top_k must be between 1 and {len(corpus)} for {files.name}")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(test), query_batch_size):
        query_rows = test[start : start + query_batch_size]
        prepared_queries = prepare_query_rows(query_rows)
        vectors = encoder.encode(prepared_queries, batch_size)
        scores, indices = index.search(vectors, top_k)
        for query, row_scores, row_indices in zip(query_rows, scores, indices):
            output = dict(query)
            output["retrieval"] = [
                _slim_example(corpus[int(index)], float(score))
                for score, index in zip(row_scores, row_indices)
            ]
            output["retrieval_config"] = {
                "method": "qwen3-vl-embedding",
                "model": str(model_path),
                "representation": "image_and_metadata_question",
                "corpus": str(files.retrieval),
                "top_k": top_k,
                "similarity": "cosine",
                "embedding_source": "computed_with_rag_tool",
            }
            rows.append(output)
        print(f"[{files.name}] retrieved {min(start + query_batch_size, len(test))}/{len(test)}", flush=True)

    output_path = files.test.parent / output_name
    write_jsonl(output_path, rows)
    print(f"[{files.name}] wrote {len(rows)} rows to {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["MM-SafetyBench", "VLGuard", "MSSBench"])
    parser.add_argument("--retrieval-name", default=DEFAULT_RETRIEVAL_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--embedding-repo", type=Path, default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--index-root", type=Path, default=ROOT / "rag_tool/indexes/real_label_category_top10_8b")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = Encoder(
        "qwen3vl",
        str(args.embedding_model),
        args.max_length,
        args.max_pixels,
        code_path=str(args.embedding_repo),
    )
    for files in dataset_files(args.data_root, args.datasets, args.retrieval_name):
        index, corpus = build_or_load_index(
            encoder=encoder,
            files=files,
            index_root=args.index_root,
            model_path=args.embedding_model,
            embedding_repo=args.embedding_repo,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
            rebuild_index=args.rebuild_index,
        )
        retrieve_dataset(
            encoder=encoder,
            files=files,
            index=index,
            corpus=corpus,
            output_name=args.output_name,
            model_path=args.embedding_model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
