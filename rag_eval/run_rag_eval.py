#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from rag_eval.data import DEFAULT_DATA_ROOT, select_datasets, validate_pair


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_REPO = Path("/home/sunyw/Qwen3-VL-Embedding")
DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_REPO / "models/Qwen3-VL-Embedding-2B"
DEFAULT_BASE_MODEL = Path("/home/sunyw/SK_RAG/models/Qwen3-VL-2B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve demonstrations and evaluate SK-RAG-OPD on safety benchmarks."
    )
    parser.add_argument("--stage", choices=("validate", "retrieve", "generate", "all"), default="all")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rag_eval/results/default")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "rag_eval/cache")
    parser.add_argument("--embedding-repo", type=Path, default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--limit", type=int, default=0, help="Test rows per dataset; 0 means all")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-suffix", default="", help="Suffix before .jsonl for sharded outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = select_datasets(args.datasets, args.data_root)
    reports = [validate_pair(files) for files in datasets]
    print(json.dumps(reports, indent=2, ensure_ascii=False), flush=True)
    invalid = [report for report in reports if not report["ok"]]
    if invalid:
        raise SystemExit("Dataset validation failed; fix the reported split/image/answer errors first.")
    if args.stage == "validate":
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("retrieve", "all"):
        from rag_eval.retrieve import make_embedder, retrieve_dataset

        embedder = make_embedder(
            args.embedding_repo, args.embedding_model,
            args.embedding_max_length, args.max_image_pixels,
        )
        for files in datasets:
            retrieve_dataset(
                embedder=embedder,
                files=files,
                output_path=args.output_dir / files.name / "retrieval.jsonl",
                cache_root=args.cache_dir,
                model_path=args.embedding_model,
                top_k=args.top_k,
                batch_size=args.embedding_batch_size,
                search_batch_size=args.search_batch_size,
                max_length=args.embedding_max_length,
                max_pixels=args.max_image_pixels,
                rebuild_cache=args.rebuild_cache,
                limit=args.limit,
            )
        del embedder
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    if args.stage == "retrieve":
        return

    from rag_eval.generate import generate_dataset, load_generator

    for path, label in ((args.base_model, "base model"), (args.adapter, "adapter")):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    model, processor = load_generator(
        args.base_model, args.adapter, args.processor_path, args.dtype, args.attn_implementation
    )
    for files in datasets:
        retrieval_path = args.output_dir / files.name / "retrieval.jsonl"
        precomputed_path = (
            files.test.parent / f"test_qwen3vl_embedding_top{args.top_k}.jsonl"
        )
        if not retrieval_path.is_file() and precomputed_path.is_file():
            retrieval_path = precomputed_path
        if not retrieval_path.is_file():
            raise FileNotFoundError(
                f"No runtime or precomputed retrieval file for {files.name}; "
                "run --stage retrieve first or use --stage all."
            )
        print(f"[{files.name}] generation", flush=True)
        generate_dataset(
            model=model,
            processor=processor,
            retrieval_path=retrieval_path,
            output_path=args.output_dir / files.name / (
                f"answers{args.output_suffix}.jsonl" if args.output_suffix else "answers.jsonl"
            ),
            base_model=args.base_model,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            resume=args.resume,
            progress_every=args.progress_every,
            batch_size=args.generation_batch_size,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            enable_thinking=args.enable_thinking,
        )


if __name__ == "__main__":
    main()
