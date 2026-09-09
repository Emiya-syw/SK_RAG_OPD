#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.data import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-GPU RAG answer shards.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    for dataset in args.datasets:
        directory = args.output_dir / dataset
        rows = []
        for index in range(args.num_shards):
            path = directory / f"answers.rank{index}of{args.num_shards}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed shard: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: str(row.get("id", "")))
        write_jsonl(directory / "answers.jsonl", rows)
        print(f"[{dataset}] merged {len(rows)} rows -> {directory / 'answers.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
