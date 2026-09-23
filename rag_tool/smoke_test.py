from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .build_index import run
from .common import write_jsonl
from .inspect import inspect_index
from .retrieve import run as retrieve_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a self-contained rag_tool smoke test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="transformers", choices=["qwen3vl", "sentence-transformers", "transformers"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rag_tool_smoke_") as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        write_jsonl(corpus, [{"id": "doc-0", "prompt": "alpha document"}, {"id": "doc-1", "prompt": "beta document"}])
        build_args = argparse.Namespace(corpus=corpus, output=root / "index", model=args.model, backend=args.backend, batch_size=2, max_length=256, max_pixels=65536, code_path=None)
        run(build_args)
        retrieve_args = argparse.Namespace(index=root / "index", query=None, query_file=None, output=root / "results.jsonl", top_k=2, batch_size=1)
        retrieve_args.query = "alpha query"
        retrieve_run(retrieve_args)
        result = inspect_index(root / "index")
        assert result["valid"]
        assert len(json.loads((root / "results.jsonl").read_text().splitlines()[0])["retrieval"]) == 2
        print("rag_tool smoke test passed")


if __name__ == "__main__":
    main()
