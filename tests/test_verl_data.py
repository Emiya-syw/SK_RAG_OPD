import json

import pytest

from opd_rag.verl_data import build_prompt, convert_record, iter_jsonl
from opd_rag.zero_reward import compute_score


def sample_record(tmp_path):
    current = tmp_path / "current.jpg"
    retrieved = tmp_path / "retrieved.jpg"
    current.write_bytes(b"image")
    retrieved.write_bytes(b"image")
    return {
        "id": "sample-1",
        "image_path": str(current),
        "question": "What is shown?",
        "reference_answer": "A safe answer.",
        "input_safety_category": "safe",
        "retrieved_examples": [
            {
                "image_path": str(retrieved),
                "question": "Related question",
                "answer": "Related answer",
                "is_harmful": False,
            }
        ],
    }


def test_convert_record_matches_verl_multimodal_schema(tmp_path):
    row = convert_record(sample_record(tmp_path), 7)

    assert row["data_source"] == "sk_rag_opd"
    assert row["extra_info"]["index"] == 7
    assert row["reward_model"]["ground_truth"] == "A safe answer."
    assert row["extra_info"]["teacher_demonstration"] == ""
    assert len(row["images"]) == 2
    assert row["prompt"][0]["content"].count("<image>") == len(row["images"])


def test_prompt_keeps_retrieval_context_without_case_specific_teacher_hint(tmp_path):
    record = sample_record(tmp_path)
    record["teacher_demonstration"] = "privileged secret"
    prompt, images = build_prompt(record)

    assert "Related question" in prompt
    assert "Related answer" in prompt
    assert "privileged secret" not in prompt
    assert len(images) == 1


def test_convert_record_checks_image_paths(tmp_path):
    record = sample_record(tmp_path)
    record["image_path"] = str(tmp_path / "missing.jpg")
    with pytest.raises(FileNotFoundError, match="missing image"):
        convert_record(record, 0)


def test_iter_jsonl_reports_line_number(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps({"question": "ok"}) + "\nnot-json\n", encoding="utf-8")
    rows = iter_jsonl(source)
    assert next(rows)["question"] == "ok"
    with pytest.raises(ValueError, match=r"rows.jsonl:2"):
        next(rows)


def test_pure_distillation_reward_is_zero():
    assert compute_score("sk_rag_opd", "answer", "reference") == 0.0
