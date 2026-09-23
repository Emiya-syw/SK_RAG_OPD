from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import prepare_rows, read_rows, write_jsonl
from .encoders import Encoder


def run(args) -> None:
    import numpy as np
    from .index import DenseIndex
    rows = prepare_rows(read_rows(args.corpus))
    code_path = getattr(args, "code_path", None)
    encoder = Encoder(args.backend, args.model, args.max_length, args.max_pixels, code_path=code_path)
    vectors = encoder.encode(rows, args.batch_size)
    index = DenseIndex(vectors)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "corpus.jsonl", rows)
    storage = index.save(args.output)
    (args.output / "manifest.json").write_text(json.dumps({"corpus": str(args.corpus.resolve()), "rows": len(rows), "dimension": int(vectors.shape[1]), "backend": args.backend, "model": args.model, "code_path": code_path, "max_length": args.max_length, "max_pixels": args.max_pixels, "storage": storage, "metric": "inner_product_on_l2_normalized_vectors"}, indent=2), encoding="utf-8")
    print(f"indexed {len(rows)} rows at {args.output} ({storage})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone dense index")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["qwen3vl", "sentence-transformers", "transformers"], default="qwen3vl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--code-path", help="local Qwen3-VL embedding code directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
