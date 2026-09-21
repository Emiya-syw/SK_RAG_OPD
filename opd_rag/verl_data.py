"""Convert SK-RAG records to the native veRL multimodal RL dataset schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


QUESTION_KEYS = ("question", "prompt", "problem", "text")
IMAGE_KEYS = ("image_path", "image", "images")
ANSWER_KEYS = ("response", "reference_answer", "answer", "solution")
GOLDEN_RESPONSE_KEYS = ("reference_answer", "response", "answer", "solution")
EXAMPLE_KEYS = ("retrieved_examples", "retrieval_samples", "examples", "retrieval")


INSTRUCTION = """Instruction

Analyze the current image and question before answering.

Use the retrieved examples as supporting evidence when they are relevant.
Compare the current case with the relevant examples and briefly explain the
shared or contrasting principle that applies here. Do not list every example,
repeat their full answers, or describe the retrieval pipeline.

Do not copy a retrieved answer blindly. Decide independently whether the
retrieved examples are relevant to the current question. If they are not
relevant, ignore them and answer based on the current image and question.

Keep the reasoning concise and focused on the current question. Avoid repeating
the same analysis or continuing after the answer strategy is clear.

After explaining the relevant connection, provide a direct and complete answer
to the current question. The answer should be useful, factual, and appropriately
cautious when the image or information is insufficient. If the request is unsafe,
apply the relevant safety principle from the examples, refuse only the unsafe
part, and provide a safe alternative. If the image or information is
insufficient, say so instead of guessing."""


def _first(record: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def _resolve_image(value: Any, image_root: str | Path | None) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("image list is empty")
        value = value[0]
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and image_root is not None:
        path = Path(image_root).expanduser() / path
    return str(path.resolve())


def _examples(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_golden_response(record: dict[str, Any]) -> str:
    """Build the off-policy target as teacher reasoning followed by the answer."""
    demonstration = str(record.get("teacher_demonstration") or "").strip()
    answer = str(_first(record, GOLDEN_RESPONSE_KEYS)).strip()
    if demonstration:
        # Keep the serialized target valid even if an upstream record already
        # included thinking tags.
        if demonstration.startswith("<think>") and demonstration.endswith("</think>"):
            thinking = demonstration
        else:
            thinking = f"<think>\n{demonstration}\n</think>"
        return f"{thinking}\n{answer}" if answer else thinking
    return answer


def build_prompt(record: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build text with one ``<image>`` marker for each path in ``images``."""
    question = str(_first(record, QUESTION_KEYS)).strip()
    if not question:
        raise ValueError("record has no question")

    examples = _examples(_first(record, EXAMPLE_KEYS, []))
    parts = ["Current Image\n\n<image>\n\n", f"Current Question\n\n{question}\n\n", "Retrieved Examples\n\n"]
    retrieved_with_images: list[Any] = []
    for index, example in enumerate(examples, 1):
        if isinstance(example, dict):
            image_path = _first(example, IMAGE_KEYS, None)
            if image_path:
                parts.append("<image>\n")
                retrieved_with_images.append(image_path)
            example_question = str(_first(example, QUESTION_KEYS)).strip()
            example_answer = str(_first(example, ANSWER_KEYS)).strip()
            label = example.get("label")
            if label is None and "is_harmful" in example:
                label = "harmful" if example["is_harmful"] else "safe/helpful"
            parts.append(f"Example {index}\n")
            if label:
                parts.append(f"Label: {label}\n")
            parts.append(f"Question: {example_question}\nResponse: {example_answer}\n\n")
        else:
            parts.append(f"Example {index}\n{example}\n\n")
    if not examples:
        parts.append("N/A\n\n")
    parts.append(INSTRUCTION)
    return "".join(parts), retrieved_with_images


def convert_record(
    record: dict[str, Any],
    index: int,
    *,
    image_root: str | Path | None = None,
    data_source: str = "sk_rag_opd",
    check_images: bool = True,
) -> dict[str, Any]:
    """Convert one source row into the schema consumed by veRL's RLHFDataset."""
    current_image = _first(record, IMAGE_KEYS, None)
    if current_image is None:
        raise ValueError("record has no image")
    prompt, retrieved_images = build_prompt(record)
    images = [_resolve_image(current_image, image_root)]
    images.extend(_resolve_image(value, image_root) for value in retrieved_images)
    if check_images:
        missing = [path for path in images if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing image(s): {', '.join(missing[:3])}")

    answer = str(_first(record, ANSWER_KEYS)).strip()
    golden_response = _build_golden_response(record)
    record_id = str(record.get("id") or record.get("source_id") or index)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "images": images,
        "ability": str(record.get("input_safety_category") or record.get("harmful_category") or "vision_safety"),
        # OPD runs with use_task_rewards=False. Keeping ground truth makes the
        # generated file useful for optional validation/reward extensions.
        "reward_model": {"style": "model", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "id": record_id,
            "question_type": record.get("question_type"),
            "input_safety_label": record.get("input_safety_label"),
            # Kept out of the student prompt.  The agent loop may optionally
            # use this as teacher-only privileged context.
            "teacher_demonstration": record.get("teacher_demonstration", ""),
            # Kept out of the student prompt. Rollout-mixture distillation can
            # use this as an off-policy/golden response.
            "golden_response": golden_response,
        },
    }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def convert_records(records: Iterable[dict[str, Any]], **kwargs: Any) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        yield convert_record(record, index, **kwargs)
