#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.data import read_jsonl, write_jsonl


def merge_dataset(directory: Path, num_shards: int) -> None:
    rows = []
    for index in range(num_shards):
        path = directory / f"answers.rank{index}of{num_shards}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed shard: {path}")
        rows.extend(read_jsonl(path))
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate answer IDs in shards for {directory.name}")
    rows.sort(key=lambda row: str(row["id"]))
    temporary = directory / "answers.jsonl.tmp"
    write_jsonl(temporary, rows)
    temporary.replace(directory / "answers.jsonl")
    print(f"[{directory.name}] merged {len(rows)} rows -> {directory / 'answers.jsonl'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-GPU RAG answer shards.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    for dataset in args.datasets:
        merge_dataset(args.output_dir / dataset, args.num_shards)


if __name__ == "__main__":
    main()
