from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-tool")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a dense index")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--model", required=True)
    build.add_argument("--backend", choices=["qwen3vl", "sentence-transformers", "transformers"], default="qwen3vl")
    build.add_argument("--batch-size", type=int, default=4)
    build.add_argument("--max-length", type=int, default=8192)
    build.add_argument("--max-pixels", type=int, default=262144)
    build.add_argument("--code-path")
    config = sub.add_parser("config", help="validate and normalize a config file")
    config.add_argument("path", type=Path)
    retrieve = sub.add_parser("retrieve", help="retrieve top-k rows")
    retrieve.add_argument("--index", type=Path, required=True)
    retrieve.add_argument("--query-file", type=Path)
    retrieve.add_argument("--query")
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--batch-size", type=int, default=4)
    inspect = sub.add_parser("inspect", help="inspect an index")
    inspect.add_argument("index")
    args, rest = parser.parse_known_args()
    if args.command == "build":
        from .build_index import run
        run(args)
    elif args.command == "retrieve":
        from .retrieve import run
        run(args)
    elif args.command == "inspect":
        from .inspect import inspect_index
        import json
        print(json.dumps(inspect_index(Path(args.index)), ensure_ascii=False, indent=2))
    elif args.command == "config":
        from .config import Config
        import json
        print(json.dumps(dict(Config(args.path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
