from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import prepare_query_rows, read_rows, write_jsonl
from .encoders import Encoder


def run(args) -> None:
    import json
    from .index import DenseIndex
    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    corpus = read_rows(args.index / "corpus.jsonl")
    queries = prepare_query_rows(read_rows(args.query_file)) if args.query_file else prepare_query_rows([{"id": "query:0", "question": args.query}])
    if args.top_k < 1 or args.top_k > len(corpus):
        raise ValueError(f"top_k must be between 1 and {len(corpus)}")
    encoder = Encoder(manifest["backend"], manifest["model"], manifest.get("max_length", 8192), manifest.get("max_pixels", 262144), code_path=manifest.get("code_path"))
    vectors = encoder.encode(queries, args.batch_size)
    scores, indices = DenseIndex.load(args.index).search(vectors, args.top_k)
    rows = []
    for query, row_scores, row_indices in zip(queries, scores, indices):
        output = dict(query)
        output["retrieval"] = []
        for score, index in zip(row_scores, row_indices):
            item = dict(corpus[int(index)])
            item["score"] = float(score)
            output["retrieval"].append(item)
        output["retrieval_config"] = {"method": "dense_inner_product", "model": manifest["model"], "corpus": manifest["corpus"], "top_k": args.top_k, "similarity": "cosine", "query_inputs": ["user_question", "images"]}
        rows.append(output)
    write_jsonl(args.output, rows)
    print(f"retrieved {len(rows)} queries to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve from a standalone rag_tool index")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if bool(args.query_file) == bool(args.query):
        parser.error("provide exactly one of --query-file or --query")
    run(args)


if __name__ == "__main__":
    main()
