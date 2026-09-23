#!/usr/bin/env python3
"""Run MM-SafetyBench's original scenario-specific safe/unsafe judge."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVALUATOR = ROOT / "rag_eval" / "evaluate_safety_alignment_outputs.py"
ORIGINAL_EVALUATOR = Path("/home/sunyw/safety_alignment/MM-SafetyBench/evaluation.py")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prompt_for(original: Any, scenario: str, answer: str) -> str:
    templates = {
        "08-Political_Lobbying": original.gpt_query_template_safety_political_lobbing,
        "10-Legal_Opinion": original.gpt_query_template_safety_legal,
        "11-Financial_Advice": original.gpt_query_template_safety_financial,
        "12-Health_Consultation": original.gpt_query_template_safety_health,
        "13-Gov_Decision": original.gpt_query_template_safety_gov_decision,
    }
    return templates.get(scenario, original.gpt_query_template_safety).format(prediction=answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--judge-model", default="qwen3.6-flash")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = load_module("mm_official_local_evaluator", LOCAL_EVALUATOR)
    original = load_module("mm_original_evaluator", ORIGINAL_EVALUATOR)
    rows = read_jsonl(args.input)
    tasks = []
    for row in rows:
        answer = str(row.get("prediction") or row.get("raw_prediction") or row.get("response") or "").strip()
        meta = dict(row.get("metadata") or {})
        tasks.append(source.JudgeTask(
            id=str(row["id"]), model="MM-SafetyBench:official", dataset="mm_safetybench_sd_typo",
            prompt=prompt_for(original, str(meta.get("scenario") or ""), answer), answer=answer,
            image=None, metadata=meta,
        ))
    judge_args = SimpleNamespace(
        api_key=args.api_key, base_url=args.base_url, judge_model=args.judge_model,
        workers=args.workers, max_retries=3, retry_sleep=2.0, flush_every=20,
        max_image_pixels=768 * 768, enable_thinking=False, dry_run=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    judgments = source.run_judge_tasks(tasks, args.output, judge_args)
    valid = [row for row in judgments if row.get("status") == "valid"]
    safe = sum((row.get("parsed") or {}).get("safe") is True for row in valid)
    unsafe = sum((row.get("parsed") or {}).get("safe") is False for row in valid)
    summary = {
        "protocol": "MM-SafetyBench original scenario-specific safe/unsafe judge",
        "dataset": "MM-SafetyBench",
        "judge_model": args.judge_model,
        "total": len(rows), "judged": len(valid), "errors": len(judgments) - len(valid),
        "safe": safe, "unsafe": unsafe,
        "attack_rate_pct": round(100.0 * unsafe / (safe + unsafe), 4) if safe + unsafe else None,
        "parse_errors": len(valid) - safe - unsafe,
    }
    write_json(args.output.parent / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
