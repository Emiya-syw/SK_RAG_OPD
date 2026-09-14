from __future__ import annotations

import json
from pathlib import Path

from rag_eval.data import DatasetFiles, example_answer_of, retrieval_text_of, validate_pair
from rag_eval.evaluate_online import is_incomplete_response, normalize_rows
from rag_eval.generate import build_rag_messages, visible_answer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_pair_and_answer_priority(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"not decoded during validation")
    retrieval = tmp_path / "retrieval.jsonl"
    test = tmp_path / "test.jsonl"
    row = {
        "id": "r1",
        "prompt": "retrieval question",
        "images": [str(image)],
        "natural_response": "merged answer",
        "answer": "draft answer",
    }
    _write_jsonl(retrieval, [row])
    _write_jsonl(test, [{"id": "q1", "prompt": "test question", "images": [str(image)]}])

    report = validate_pair(DatasetFiles("demo", retrieval, test))

    assert report["ok"]
    assert example_answer_of(row) == "merged answer"
    assert retrieval_text_of({**row, "metadata": {"question": "original question"}}) == "original question"


def test_rag_message_image_order_and_visible_answer() -> None:
    row = {
        "id": "q1",
        "prompt": "current question",
        "images": ["current.jpg"],
        "retrieval": [
            {
                "id": "r1",
                "prompt": "example question",
                "images": ["one.jpg", "two.jpg"],
                "answer": "example answer",
                "score": 0.75,
            }
        ],
    }

    content = build_rag_messages(row)[0]["content"]

    assert sum(block["type"] == "image" for block in content) == 3
    assert "example answer" in content[-2]["text"]
    assert visible_answer("reasoning</think>final") == "final"


def test_online_eval_keeps_truncated_and_unclosed_thinking_outputs() -> None:
    rows = normalize_rows([
        {
            "id": "truncated",
            "raw_prediction": "<think>unfinished but still judgeable",
            "prediction": "<think>unfinished but still judgeable",
            "generation_truncated": True,
        },
        {
            "id": "no-think",
            "prediction": "A direct answer without thinking tags.",
        },
        {"id": "empty", "prediction": ""},
    ])

    assert not is_incomplete_response(rows[0])
    assert not is_incomplete_response(rows[1])
    assert is_incomplete_response(rows[2])
