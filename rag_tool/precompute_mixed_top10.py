#!/usr/bin/env python3
"""Build mixed-corpus Top-10 retrieval files with Qwen3-VL-Embedding."""
from __future__ import annotations

import argparse
import json
import numpy as np
from datetime import datetime, timezone
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
DEFAULT_OUTPUT_NAME = "test_mixed_top10.jsonl"
DEFAULT_RUN_NAME = datetime.now(timezone.utc).strftime("%Y%m%d%H%M_mixed_top10_qwen3vl_embedding_8b")


def dataset_files(data_root: Path, names: list[str], retrieval_name: str) -> list[DatasetFiles]:
    files: list[DatasetFiles] = []
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


def build_mixed_corpus(files_list: list[DatasetFiles], mixed_corpus_path: Path) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    for files in files_list:
        rows = read_jsonl(files.retrieval)
        for source_index, row in enumerate(rows):
            output = dict(row)
            output["source_dataset"] = files.name
            output["source_index"] = source_index
            output["source_retrieval_file"] = str(files.retrieval)
            output["mixed_id"] = f"{files.name}:{source_index}:{output.get('id', source_index)}"
            output["id"] = output["mixed_id"]
            if output.get("dataset") is None:
                output["dataset"] = files.name
            corpus.append(output)
        print(f"[mixed] added {len(rows)} rows from {files.name}", flush=True)
    write_jsonl(mixed_corpus_path, corpus)
    print(f"[mixed] wrote {len(corpus)} rows to {mixed_corpus_path}", flush=True)
    return corpus


def source_index_matches(index_dir: Path, files: DatasetFiles, model_path: Path, embedding_repo: Path, max_length: int, max_pixels: int) -> bool:
    manifest_path = index_dir / "manifest.json"
    embeddings_path = index_dir / "embeddings.npy"
    if not manifest_path.is_file() or not embeddings_path.is_file():
        return False
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        actual.get("corpus") == str(files.retrieval.resolve())
        and actual.get("rows") == len(read_jsonl(files.retrieval))
        and actual.get("backend") == "qwen3vl"
        and actual.get("model") == str(model_path.resolve())
        and actual.get("code_path") == str(embedding_repo.resolve())
        and actual.get("max_length") == max_length
        and actual.get("max_pixels") == max_pixels
        and embeddings_path.is_file()
    )


def build_or_load_index(
    encoder: Encoder | None,
    files_list: list[DatasetFiles],
    mixed_corpus_path: Path,
    index_dir: Path,
    model_path: Path,
    embedding_repo: Path,
    batch_size: int,
    max_length: int,
    max_pixels: int,
    rebuild_index: bool,
    source_index_root: Path | None,
) -> tuple[DenseIndex, list[dict[str, Any]]]:
    if not mixed_corpus_path.is_file() or rebuild_index:
        corpus = build_mixed_corpus(files_list, mixed_corpus_path)
    else:
        corpus = read_jsonl(mixed_corpus_path)
        print(f"[mixed] using existing corpus {mixed_corpus_path} ({len(corpus)} rows)", flush=True)

    prepared = prepare_rows(corpus)
    expected_manifest = {
        "corpus": str(mixed_corpus_path.resolve()),
        "source_retrievals": [str(files.retrieval.resolve()) for files in files_list],
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
        print(f"[mixed] using cached rag_tool index {index_dir}", flush=True)
        return DenseIndex.load(index_dir), corpus

    vectors = None
    if source_index_root is not None:
        source_vectors = []
        reusable = True
        for files in files_list:
            source_dir = source_index_root / files.name
            if source_index_matches(source_dir, files, model_path, embedding_repo, max_length, max_pixels):
                print(f"[mixed] reusing cached embeddings {source_dir / 'embeddings.npy'}", flush=True)
                source_vectors.append(np.load(source_dir / "embeddings.npy", mmap_mode="r"))
            else:
                reusable = False
                print(f"[mixed] cache not reusable for {files.name}: {source_dir}", flush=True)
                break
        if reusable:
            vectors = np.concatenate([np.asarray(v, dtype="float32") for v in source_vectors], axis=0)
            if vectors.shape[0] != len(corpus):
                raise ValueError(f"Concatenated vector rows {vectors.shape[0]} != mixed corpus rows {len(corpus)}")

    if vectors is None:
        if encoder is None:
            raise ValueError("encoder is required when source embeddings cannot be reused")
        print(f"[mixed] encoding mixed retrieval corpus", flush=True)
        vectors = encoder.encode(prepared, batch_size)
    else:
        print(f"[mixed] built mixed embeddings from cached source indexes: {vectors.shape}", flush=True)
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


def slim_mixed_example(row: dict[str, Any], score: float) -> dict[str, Any]:
    example = _slim_example(row, score)
    example["source_dataset"] = row.get("source_dataset")
    example["source_index"] = row.get("source_index")
    example["source_retrieval_file"] = row.get("source_retrieval_file")
    example["mixed_id"] = row.get("mixed_id")
    return example


def retrieve_dataset(
    encoder: Encoder,
    files: DatasetFiles,
    index: DenseIndex,
    corpus: list[dict[str, Any]],
    mixed_corpus_path: Path,
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
                slim_mixed_example(corpus[int(index)], float(score))
                for score, index in zip(row_scores, row_indices)
            ]
            output["retrieval_config"] = {
                "method": "qwen3-vl-embedding",
                "model": str(model_path),
                "representation": "query_image_and_question_against_mixed_image_question_or_question_corpus",
                "corpus": str(mixed_corpus_path),
                "source_retrieval_name": DEFAULT_RETRIEVAL_NAME,
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
    parser.add_argument("--work-dir", type=Path, default=ROOT / "mid_file" / DEFAULT_RUN_NAME)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--source-index-root", type=Path, default=ROOT / "rag_tool/indexes/real_label_category_top10_8b",
                        help="Optional per-dataset index root whose embeddings can be concatenated into the mixed index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files_list = dataset_files(args.data_root, args.datasets, args.retrieval_name)
    work_dir = args.work_dir
    mixed_corpus_path = work_dir / "mixed_corpus.jsonl"
    index_dir = work_dir / "index"
    source_index_root = args.source_index_root if args.source_index_root else None
    can_reuse_all = bool(source_index_root) and all(
        source_index_matches(source_index_root / files.name, files, args.embedding_model, args.embedding_repo, args.max_length, args.max_pixels)
        for files in files_list
    )
    encoder = None if can_reuse_all and not args.rebuild_index else Encoder(
        "qwen3vl",
        str(args.embedding_model),
        args.max_length,
        args.max_pixels,
        code_path=str(args.embedding_repo),
    )
    if encoder is None:
        print("[mixed] source embedding caches are reusable; delaying model load until query retrieval", flush=True)
    index, corpus = build_or_load_index(
        encoder=encoder,
        files_list=files_list,
        mixed_corpus_path=mixed_corpus_path,
        index_dir=index_dir,
        model_path=args.embedding_model,
        embedding_repo=args.embedding_repo,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        rebuild_index=args.rebuild_index,
        source_index_root=source_index_root,
    )
    if encoder is None:
        encoder = Encoder(
            "qwen3vl",
            str(args.embedding_model),
            args.max_length,
            args.max_pixels,
            code_path=str(args.embedding_repo),
        )
    for files in files_list:
        retrieve_dataset(
            encoder=encoder,
            files=files,
            index=index,
            corpus=corpus,
            mixed_corpus_path=mixed_corpus_path,
            output_name=args.output_name,
            model_path=args.embedding_model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
