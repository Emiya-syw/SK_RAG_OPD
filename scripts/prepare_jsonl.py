"""Normalize common RAG records into the OPD training schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def normalize(row: dict, index: int) -> dict:
    examples = row.get("retrieved_examples", row.get("retrieval_samples", row.get("retrieval", []))) or []
    answer = row.get("reference_answer", row.get("answer", row.get("response", ""))) or ""
    return {
        "id": row.get("id", str(index)), "image_path": row.get("image_path", row.get("image", "")),
        "question": row.get("question", row.get("prompt", row.get("text", ""))),
        "reference_answer": answer, "teacher_demonstration": row.get("teacher_demonstration", answer),
        "safety_knowledge": row.get("safety_knowledge", row.get("knowledge", row.get("context", ""))),
        "retrieved_examples": examples, "is_harmful": bool(row.get("is_harmful", False)),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if not line.strip(): continue
            if args.limit and count >= args.limit: break
            dst.write(json.dumps(normalize(json.loads(line), index), ensure_ascii=False) + "\n"); count += 1
    print(f"wrote {count} records to {args.output}")

if __name__ == "__main__": main()
