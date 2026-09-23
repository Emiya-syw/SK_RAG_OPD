from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else value.get("data", value.get("rows", []))
    if suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id", row.get("doc_id", index)))


def row_text(row: dict[str, Any]) -> str:
    # Dataset-specific metadata questions take precedence over presentation
    # prompts. MSSBench prompts contain a long system instruction that is not
    # part of the retrieval representation.
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("question")
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in ("contents", "text", "prompt", "question"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if isinstance(metadata, dict):
        for key in ("question", "text", "scenario"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def row_images(row: dict[str, Any]) -> list[str]:
    value = row.get("images", row.get("image_path", row.get("image", [])))
    if value is None or value == "":
        return []
    return [str(value)] if isinstance(value, str) else [str(item) for item in value]


def prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["id"] = row_id(row, index)
        row["contents"] = row_text(row)
        row["images"] = row_images(row)
        if not row["contents"] and not row["images"]:
            raise ValueError(f"Row {row['id']} has neither text nor images")
        prepared.append(row)
    return prepared


def prepare_query_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare retrieval queries using only the user question and images.

    MSSBench rows also contain a long system prompt. That prompt is metadata for
    the answering model and must not leak into the retrieval embedding.
    """
    prepared = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["id"] = row_id(row, index)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        question = metadata.get("question") or row.get("question") or row.get("user_question")
        if question is None:
            question = row.get("prompt", row.get("contents", row.get("text", "")))
        row["contents"] = str(question or "").strip()
        row["images"] = row_images(row)
        if not row["contents"] and not row["images"]:
            raise ValueError(f"Query {row['id']} has neither user question nor image")
        prepared.append(row)
    return prepared
