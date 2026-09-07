from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.data import (
    DatasetFiles,
    example_answer_of,
    file_fingerprint,
    image_paths_of,
    prompt_of,
    read_jsonl,
    retrieval_text_of,
    write_jsonl,
)


RETRIEVAL_INSTRUCTION = "Retrieve images or text relevant to the user's query."


def _open_images(row: dict[str, Any]) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for value in image_paths_of(row):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Image for {row.get('id')}: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def make_embedder(
    model_repo: Path, model_path: Path, max_length: int, max_pixels: int
):
    import torch

    sys.path.insert(0, str(model_repo.resolve()))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder(
        model_name_or_path=str(model_path.resolve()),
        max_length=max_length,
        max_pixels=max_pixels,
        default_instruction="Represent the user's multimodal input.",
        torch_dtype=torch.bfloat16,
    )


def _encode_rows(embedder, rows: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    import torch

    parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        opened: list[list[Any]] = []
        try:
            opened = [_open_images(row) for row in batch]
            inputs = [
                {
                    "text": retrieval_text_of(row),
                    "image": images or None,
                    "instruction": RETRIEVAL_INSTRUCTION,
                }
                for row, images in zip(batch, opened)
            ]
            with torch.inference_mode():
                encoded = embedder.process(inputs, normalize=True)
            parts.append(encoded.detach().cpu().float().numpy())
        finally:
            for image_group in opened:
                for image in image_group:
                    image.close()
        if start == 0 or start + batch_size >= len(rows) or start % (batch_size * 25) == 0:
            print(f"  embedded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    if not parts:
        raise ValueError("Cannot embed an empty split")
    return np.concatenate(parts).astype(np.float32, copy=False)


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    return cache_dir / f"{split}.npy", cache_dir / f"{split}.meta.json"


def _load_or_encode(
    embedder,
    rows: list[dict[str, Any]],
    source: Path,
    cache_dir: Path,
    split: str,
    model_path: Path,
    max_length: int,
    max_pixels: int,
    batch_size: int,
    rebuild: bool,
) -> np.ndarray:
    matrix_path, meta_path = _cache_paths(cache_dir, split)
    expected = {
        "source": str(source.resolve()),
        "source_sha256": file_fingerprint(source),
        "ids": [str(row["id"]) for row in rows],
        "model": str(model_path.resolve()),
        "max_length": max_length,
        "max_pixels": max_pixels,
        "instruction": RETRIEVAL_INSTRUCTION,
    }
    if not rebuild and matrix_path.is_file() and meta_path.is_file():
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
        if actual == expected:
            matrix = np.load(matrix_path, mmap_mode="r")
            if matrix.shape[0] == len(rows):
                print(f"  using cache {matrix_path}", flush=True)
                return matrix
    print(f"  building cache {matrix_path}", flush=True)
    matrix = _encode_rows(embedder, rows, batch_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    meta_path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
    return matrix


def _slim_example(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": row.get("dataset"),
        "subset": row.get("subset"),
        "prompt": prompt_of(row),
        "images": image_paths_of(row),
        "answer": example_answer_of(row),
        "score": score,
    }


def retrieve_dataset(
    embedder,
    files: DatasetFiles,
    output_path: Path,
    cache_root: Path,
    model_path: Path,
    top_k: int,
    batch_size: int,
    search_batch_size: int,
    max_length: int,
    max_pixels: int,
    rebuild_cache: bool,
    limit: int = 0,
) -> None:
    corpus = read_jsonl(files.retrieval)
    queries = read_jsonl(files.test)
    if limit > 0:
        queries = queries[:limit]
    if top_k < 1 or top_k > len(corpus):
        raise ValueError(f"--top-k must be in [1, {len(corpus)}] for {files.name}")
    cache_dir = cache_root / files.name
    print(f"[{files.name}] corpus", flush=True)
    corpus_embeddings = _load_or_encode(
        embedder, corpus, files.retrieval, cache_dir, "retrieval", model_path,
        max_length, max_pixels, batch_size, rebuild_cache,
    )
    print(f"[{files.name}] test", flush=True)
    query_embeddings = _load_or_encode(
        embedder, queries, files.test, cache_dir, f"test_limit{limit or 'all'}", model_path,
        max_length, max_pixels, batch_size, rebuild_cache,
    )

    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(queries), search_batch_size):
        scores = np.asarray(query_embeddings[start : start + search_batch_size]) @ np.asarray(corpus_embeddings).T
        candidate_ids = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(scores, candidate_ids, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        ranked_ids = np.take_along_axis(candidate_ids, order, axis=1)
        ranked_scores = np.take_along_axis(candidate_scores, order, axis=1)
        for offset, (indices, values) in enumerate(zip(ranked_ids, ranked_scores)):
            query = queries[start + offset]
            output_rows.append(
                {
                    "id": query["id"],
                    "dataset": query.get("dataset"),
                    "subset": query.get("subset"),
                    "prompt": prompt_of(query),
                    "images": image_paths_of(query),
                    "metadata": query.get("metadata", {}),
                    "retrieval": [
                        _slim_example(corpus[int(index)], float(score))
                        for index, score in zip(indices, values)
                    ],
                    "retrieval_config": {
                        "method": "qwen3-vl-embedding",
                        "model": str(model_path),
                        "representation": "image_and_prompt",
                        "source": str(files.retrieval),
                        "top_k": top_k,
                        "similarity": "cosine",
                    },
                }
            )
    write_jsonl(output_path, output_rows)
    print(f"[{files.name}] saved {len(output_rows)} retrieval rows to {output_path}", flush=True)
    gc.collect()
