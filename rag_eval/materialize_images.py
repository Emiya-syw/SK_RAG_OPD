#!/usr/bin/env python3
"""Copy all evaluation images into rag_eval/data and rewrite JSONL paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def image_paths(row: dict[str, Any]) -> list[str]:
    value = row.get("images", [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def target_for(source: Path, image_root: Path, dataset: str) -> Path:
    source = source.resolve()
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:24]
    suffix = source.suffix.lower() or ".img"
    return image_root / dataset / f"{digest}{suffix}"


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    image_root = data_root / "images"
    jsonl_paths = sorted(data_root.glob("*/test.jsonl")) + sorted(
        data_root.glob("*/retrieval_qwen3.7_plus_merged.jsonl")
    )
    if not jsonl_paths:
        raise SystemExit(f"No evaluation JSONL files found under {data_root}")

    manifest_path = data_root / "image_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else []
    )
    original_by_local = {
        Path(item["local"]).resolve(): Path(item["source"]).resolve()
        for item in previous_manifest
    }

    loaded = {path: read_jsonl(path) for path in jsonl_paths}
    references = sorted(
        {
            (path.parent.name, Path(value).resolve())
            for path, rows in loaded.items()
            for row in rows
            for value in image_paths(row)
        },
        key=lambda item: (item[0], str(item[1])),
    )
    copy_sources = {
        reference: (
            reference[1]
            if reference[1].is_file()
            else original_by_local.get(reference[1], reference[1])
        )
        for reference in references
    }
    missing = [reference for reference, source in copy_sources.items() if not source.is_file()]
    if missing:
        preview = "\n".join(f"{dataset}: {path}" for dataset, path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} source images:\n{preview}")

    mapping = {
        reference: target_for(
            original_by_local.get(reference[1], reference[1]), image_root, reference[0]
        )
        for reference in references
    }
    copied = 0
    linked = 0
    for index, reference in enumerate(references, start=1):
        source = copy_sources[reference]
        destination = mapping[reference]
        if destination != source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"Existing destination has the wrong size: {destination}")
            if not destination.is_file():
                if is_inside(source.resolve(), image_root.resolve()):
                    try:
                        os.link(source, destination)
                        linked += 1
                    except OSError:
                        shutil.copy2(source, destination)
                        copied += 1
                else:
                    shutil.copy2(source, destination)
                    copied += 1
        if index % 500 == 0 or index == len(references):
            print(
                f"images {index}/{len(references)}; copied {copied}; linked {linked}",
                flush=True,
            )

    manifest: list[dict[str, Any]] = []
    for (dataset, referenced_path), destination in mapping.items():
        original = original_by_local.get(referenced_path, referenced_path)
        manifest.append(
            {
                "dataset": dataset,
                "source": str(original),
                "local": str(destination),
                "bytes": destination.stat().st_size,
            }
        )
    manifest.sort(key=lambda item: (item["dataset"], item["source"]))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for path, rows in loaded.items():
        dataset = path.parent.name
        for row in rows:
            row["images"] = [
                str(mapping[(dataset, Path(value).resolve())]) for value in image_paths(row)
            ]
        write_jsonl_atomic(path, rows)

    desired = set(mapping.values())
    removed = 0
    for path in image_root.rglob("*"):
        if path.is_file() and path not in desired:
            path.unlink()
            removed += 1
    for path in sorted((item for item in image_root.rglob("*") if item.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass

    total_bytes = sum(item["bytes"] for item in manifest)
    print(
        f"materialized {len(mapping)} dataset image entries ({total_bytes} referenced bytes); "
        f"removed {removed} old files; rewrote {len(jsonl_paths)} JSONL files",
        flush=True,
    )
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
