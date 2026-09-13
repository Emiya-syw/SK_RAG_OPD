#!/usr/bin/env python3
"""Convert SK-RAG JSONL into a veRL JSONL or Parquet dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opd_rag.verl_data import convert_records, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--data-source", default="sk_rag_opd")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_records(
        iter_jsonl(args.input),
        image_root=args.image_root,
        data_source=args.data_source,
        check_images=not args.skip_image_check,
    )
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        from itertools import islice

        rows = islice(rows, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("Parquet output requires `pip install datasets pyarrow`") from exc
        materialized = list(rows)
        if not materialized:
            raise ValueError("no records were converted")
        Dataset.from_list(materialized).to_parquet(str(args.output))
        count = len(materialized)
    elif args.output.suffix in {".json", ".jsonl"}:
        count = 0
        with args.output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    else:
        raise ValueError("output extension must be .jsonl, .json, or .parquet")
    print(f"wrote {count} veRL records to {args.output}")


if __name__ == "__main__":
    main()
