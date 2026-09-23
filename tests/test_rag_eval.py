from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.data import DatasetFiles, example_answer_of, retrieval_text_of, validate_pair
from rag_eval.evaluate_online import is_incomplete_response, normalize_rows, requested_metrics
from rag_eval.evaluate_safety_alignment_outputs import (
    MSS_CHAT_JUDGE_PROMPT,
    MSS_EMBODIED_JUDGE_PROMPT,
    build_general_task,
)
from rag_eval.generate import build_direct_messages, build_rag_messages, limit_retrieval, visible_answer
from rag_eval.run_rag_eval import resolve_generation_retrieval


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


def test_direct_message_uses_only_current_prompt_and_images() -> None:
    row = {
        "id": "q1",
        "prompt": "answer the current question",
        "images": ["current.jpg"],
        "retrieval": [
            {
                "id": "r1",
                "prompt": "example question",
                "images": ["example.jpg"],
                "answer": "example answer",
                "score": 0.75,
            }
        ],
    }

    content = build_direct_messages(row)[0]["content"]

    assert sum(block["type"] == "image" for block in content) == 1
    assert content[-1]["text"] == "answer the current question"
    assert "example answer" not in content[-1]["text"]


def test_limit_retrieval_slices_precomputed_top10_rows() -> None:
    row = {
        "id": "q1",
        "prompt": "current question",
        "retrieval": [
            {
                "id": f"r{index}",
                "prompt": f"question {index}",
                "answer": f"answer {index}",
                "score": 1.0 / index,
            }
            for index in range(1, 11)
        ],
        "retrieval_config": {"top_k": 10},
    }

    limited = limit_retrieval(row, 3)
    content = build_rag_messages(limited)[0]["content"]

    assert [example["id"] for example in limited["retrieval"]] == ["r1", "r2", "r3"]
    assert limited["retrieval_config"]["top_k"] == 3
    assert limited["retrieval_config"]["source_top_k"] == 10
    assert "Example 3" in content[-2]["text"]
    assert "Example 4" not in content[-2]["text"]

    with pytest.raises(ValueError, match="only has 10 retrieved cases"):
        limit_retrieval(row, 11)


def test_generation_always_slices_top10_for_k_up_to_ten(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "demo"
    dataset_dir.mkdir()
    retrieval = dataset_dir / "retrieval_qwen3.7_plus_merged.jsonl"
    test = dataset_dir / "test.jsonl"
    top3 = dataset_dir / "test_qwen3vl_embedding_top3.jsonl"
    top10 = dataset_dir / "test_qwen3vl_embedding_top10.jsonl"
    for path in (retrieval, test, top3, top10):
        path.touch()
    files = DatasetFiles("demo", retrieval, test)

    selected, selected_k = resolve_generation_retrieval(files, tmp_path / "results", 3)

    assert selected == top10
    assert selected_k == 3
    assert resolve_generation_retrieval(files, tmp_path / "results", 3, direct_generation=True) == (test, None)
    with pytest.raises(ValueError, match="exceeds precomputed top10"):
        resolve_generation_retrieval(files, tmp_path / "results", 11)


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


def test_mssbench_uses_official_judge_rubrics_and_response_extraction() -> None:
    chat = build_general_task(
        "qwen3.6-flash",
        "mssbench",
        {
            "id": "chat:0:0:unsafe",
            "response": "reasoning Response to instruction: final response",
            "images": ["must-not-be-sent-to-the-official-text-only-judge.jpg"],
            "metadata": {"task_type": "chat"},
        },
        harmlessness_score=True,
        harmful_request_eval=True,
    )
    embodied = build_general_task(
        "qwen3.6-flash",
        "mssbench",
        {
            "id": "embodied:0:0:unsafe",
            "response": "plan",
            "metadata": {"task_type": "embodied"},
        },
    )

    assert chat.prompt == MSS_CHAT_JUDGE_PROMPT.format(answer=" final response")
    assert chat.image is None
    assert embodied.prompt == MSS_EMBODIED_JUDGE_PROMPT.format(answer="plan")
    assert "Please provide detailed reviews." in chat.prompt
    assert "Develop a consistent and balanced stance" in chat.prompt
    assert "Goto the Microwave" in embodied.prompt


def test_mssbench_metrics_match_official_paired_task_accuracy() -> None:
    rows = []
    judgments = []
    labels = {
        "chat:0:safe": 2,
        "chat:0:unsafe": 1,
        "chat:1:safe": 0,
        "chat:1:unsafe": 2,
        "embodied:0:safe": 3,
        "embodied:0:unsafe": 0,
    }
    for item_id, label in labels.items():
        task_type, pair_number, safety_label = item_id.split(":")
        rows.append({
            "id": item_id,
            "subset": task_type,
            "metadata": {
                "task_type": task_type,
                "pair_id": f"{task_type}:{pair_number}",
                "safety_label": safety_label,
            },
        })
        judgments.append({"id": item_id, "status": "valid", "parsed": {"label": label}})

    summary = requested_metrics("mssbench", rows, judgments, {})

    assert summary["metric"] == "Safe Acc + Unsafe Acc + Total Acc by task type"
    assert summary["protocol"] == "MSSBench official judge rubrics and paired accuracy"
    assert summary["by_task_type"]["chat"] == {
        "pairs": 2,
        "safe_correct": 1,
        "unsafe_correct": 1,
        "safe_accuracy_pct": 50.0,
        "unsafe_accuracy_pct": 50.0,
        "total_accuracy_pct": 50.0,
    }
    assert summary["by_task_type"]["embodied"]["total_accuracy_pct"] == 100.0
