#!/usr/bin/env python3
"""Evaluate RAG answers with SK_RAG's online safety-alignment judge."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/sunyw/SK_RAG/scripts/eval/evaluate_safety_alignment_outputs.py")
DATASETS = {
    "BeaverTails-V": "beavertails_v",
    "FigStep": "figstep",
    "MM-SafetyBench": "mm_safetybench_sd_typo",
    "MSSBench": "mssbench",
    "SIUO-Gen": "siuo_gen",
    "SIUO-MCQA": "siuo_mcqa",
    "VLGuard": "vlguard",
}


def load_source() -> Any:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing reference evaluator: {SOURCE}")
    spec = importlib.util.spec_from_file_location("sk_rag_safety_evaluator", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def find_answer_files(input_root: Path, requested: list[str]) -> list[tuple[str, Path]]:
    selected = set(requested)
    found = []
    for directory in sorted(input_root.iterdir() if input_root.is_dir() else []):
        if not directory.is_dir() or ("all" not in selected and directory.name not in selected):
            continue
        answer_path = directory / "answers.jsonl"
        if answer_path.is_file():
            found.append((DATASETS.get(directory.name, directory.name.lower()), answer_path))
    if not found:
        raise SystemExit(f"No answers.jsonl found under {input_root}")
    return found


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        value = dict(row)
        value["response"] = str(
            row.get("prediction") or row.get("raw_prediction") or row.get("response") or ""
        ).strip()
        normalized.append(value)
    return normalized


def evaluator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        api_key=args.api_key, base_url=args.base_url, judge_model=args.judge_model,
        workers=args.workers, max_retries=args.max_retries, retry_sleep=args.retry_sleep,
        flush_every=args.flush_every, max_image_pixels=args.max_image_pixels,
        enable_thinking=args.enable_thinking, harmlessness_score=args.harmlessness_score,
        harmful_request_eval=args.harmful_request_eval, dry_run=args.dry_run,
    )


def evaluate_one(source: Any, dataset: str, path: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = normalize_rows(read_jsonl(path))
    if args.limit:
        rows = rows[: args.limit]
    output_dir = output_root / path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dataset = dataset
    if dataset == "siuo_mcqa":
        return source.evaluate_siuo_mcqa(rows, output_dir)
    if dataset == "vlguard" and not args.online_vlguard:
        return source.evaluate_vlguard(rows, output_dir)
    tasks = [source.build_general_task(path.parent.name, source_dataset, row,
                                       harmlessness_score=args.harmlessness_score,
                                       harmful_request_eval=args.harmful_request_eval)
             for row in rows]
    if args.dry_run:
        summary = {"protocol": "LLM judge", "dataset": dataset, "total": len(rows),
                   "pending_tasks": len(tasks), "dry_run": True}
        write_json(output_dir / "summary.json", summary)
        return summary
    judge_args = evaluator_args(args)
    judgments = source.run_judge_tasks(tasks, output_dir / "judgments.jsonl", judge_args)
    summary = source.summarize_judgments(source_dataset, rows, judgments, output_dir)
    if dataset == "siuo_gen":
        effectiveness_tasks = [source.build_siuo_effective_task(path.parent.name, row) for row in rows]
        effectiveness = source.run_judge_tasks(effectiveness_tasks,
                                               output_dir / "effectiveness_judgments.jsonl", judge_args)
        summary = source.merge_siuo_gen_effectiveness(output_dir, summary, effectiveness)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "rag_eval/results/test_V1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "rag_eval/evaluation")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--judge-model", default="qwen3.6-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-image-pixels", type=int, default=768 * 768)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--harmlessness-score", action="store_true")
    parser.add_argument("--harmful-request-eval", action="store_true")
    parser.add_argument("--online-vlguard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_source()
    files = find_answer_files(args.input_root, args.datasets)
    if not args.dry_run and not args.api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or pass --api-key for online judging")
    summaries = []
    for dataset, path in files:
        print(f"Evaluating {path.parent.name} ({dataset})", flush=True)
        summary = evaluate_one(source, dataset, path, args.output_root, args)
        summaries.append({"dataset_dir": path.parent.name, **summary})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    write_json(args.output_root / "summary.json", {"results": summaries})
    print(f"Saved global summary to {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
