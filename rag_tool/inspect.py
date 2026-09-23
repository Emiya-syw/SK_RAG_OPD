from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_rows


def inspect_index(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    corpus = read_rows(path / "corpus.jsonl")
    ids = [str(row["id"]) for row in corpus]
    result = {
        "manifest": manifest,
        "corpus_rows": len(corpus),
        "unique_ids": len(set(ids)),
        "has_faiss": (path / "index.faiss").is_file(),
        "has_numpy_vectors": (path / "embeddings.npy").is_file(),
        "valid": len(corpus) == manifest.get("rows") and len(set(ids)) == len(ids),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a rag_tool index")
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_index(args.index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
