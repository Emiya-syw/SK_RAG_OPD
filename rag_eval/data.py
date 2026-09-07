from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
RETRIEVAL_CANDIDATES = (
    "retrieval_qwen3.7_plus_merged.jsonl",
    "retrieval_qwen3.7_plus.jsonl",
    "retrieval.jsonl",
)


@dataclass(frozen=True)
class DatasetFiles:
    name: str
    retrieval: Path
    test: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
        target.flush()


def discover_datasets(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, DatasetFiles]:
    datasets: dict[str, DatasetFiles] = {}
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    for directory in sorted(path for path in data_root.iterdir() if path.is_dir()):
        test = directory / "test.jsonl"
        retrieval = next(
            (directory / name for name in RETRIEVAL_CANDIDATES if (directory / name).is_file()),
            None,
        )
        if test.is_file() and retrieval is not None:
            datasets[directory.name] = DatasetFiles(directory.name, retrieval, test)
    if not datasets:
        raise ValueError(f"No retrieval/test dataset pairs found under {data_root}")
    return datasets


def select_datasets(
    requested: list[str], data_root: Path = DEFAULT_DATA_ROOT
) -> list[DatasetFiles]:
    available = discover_datasets(data_root)
    if not requested or any(name.lower() == "all" for name in requested):
        return list(available.values())
    by_lower = {name.lower(): files for name, files in available.items()}
    missing = [name for name in requested if name.lower() not in by_lower]
    if missing:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )
    return [by_lower[name.lower()] for name in requested]


def prompt_of(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "problem", "text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Row {row.get('id', '<unknown>')} has no prompt")


def retrieval_text_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        question = metadata.get("question")
        if question is not None and str(question).strip():
            return str(question).strip()
    return prompt_of(row)


def image_paths_of(row: dict[str, Any]) -> list[str]:
    value = row.get("images", row.get("image_path", row.get("image")))
    if value is None or value == "":
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(path) for path in values if path is not None and str(path)]


def example_answer_of(row: dict[str, Any]) -> str:
    for key in ("natural_response", "answer", "response", "safe_alternative"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def validate_pair(files: DatasetFiles) -> dict[str, Any]:
    retrieval = read_jsonl(files.retrieval)
    test = read_jsonl(files.test)
    retrieval_ids = [str(row.get("id", "")) for row in retrieval]
    test_ids = [str(row.get("id", "")) for row in test]
    errors: list[str] = []
    if not retrieval or not test:
        errors.append("empty split")
    if any(not item for item in retrieval_ids + test_ids):
        errors.append("missing id")
    if len(set(retrieval_ids)) != len(retrieval_ids):
        errors.append("duplicate retrieval id")
    if len(set(test_ids)) != len(test_ids):
        errors.append("duplicate test id")
    overlap = set(retrieval_ids) & set(test_ids)
    if overlap:
        errors.append(f"retrieval/test overlap ({len(overlap)})")
    missing_answers = sum(not example_answer_of(row) for row in retrieval)
    missing_images = sum(
        not Path(path).is_file()
        for row in retrieval + test
        for path in image_paths_of(row)
    )
    for row in retrieval + test:
        prompt_of(row)
    return {
        "dataset": files.name,
        "retrieval_file": str(files.retrieval),
        "test_file": str(files.test),
        "retrieval_count": len(retrieval),
        "test_count": len(test),
        "missing_retrieval_answers": missing_answers,
        "missing_images": missing_images,
        "errors": errors,
        "ok": not errors and missing_answers == 0 and missing_images == 0,
    }


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
