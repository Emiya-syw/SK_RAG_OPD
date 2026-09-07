from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image


QUESTION_KEYS = ("question", "prompt", "problem", "text")
IMAGE_KEYS = ("image_path", "image", "images")
KNOWLEDGE_KEYS = ("safety_knowledge", "knowledge", "retrieved_context", "context")
ANSWER_KEYS = ("response", "reference_answer", "answer", "solution")
EXAMPLE_KEYS = ("retrieved_examples", "retrieval_samples", "examples", "retrieval")
DEMONSTRATION_KEYS = ("response_recommendation", "teacher_demonstration", "natural_answer", "demonstration")


TEACHER_EXAMPLE_RUBRIC = """When using the retrieved examples, apply this decision framework:
1. Judge whether each example is visually and semantically relevant to the current image and request.
2. Separate surface similarity from policy similarity.
3. Extract the shared safety/helpfulness rule from relevant examples.
4. Identify conflicts between retrieved examples and the current visual evidence.
5. Do not copy example answers. Use examples only to support the decision for the current case.
6. Prefer a concise final answer grounded in the current image and request."""


def _first_present(example: dict[str, Any], keys: tuple[str, ...], default: str = "") -> Any:
    for key in keys:
        value = example.get(key)
        if value is not None and value != "":
            return value
    return default


def _load_image(value: Any, image_root: str | None = None, max_pixels: int | None = None) -> Image.Image:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty image list")
        value = value[0]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = Path(str(value))
    if not path.is_absolute() and image_root:
        path = Path(image_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if max_pixels and image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / float(image.width * image.height))
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


class OPDDataCollator:
    """Build student and privileged-teacher Qwen-VL inputs for OPD.

    Student view:
      image + user question + retrieved examples

    Teacher view:
      image + user question + retrieved examples + example-analysis rubric
    """

    def __init__(
        self,
        processor,
        max_prompt_length: int = 4096,
        include_reference_answer: bool = True,
        image_root: str | None = None,
        max_image_pixels: int = 262144,
        teacher_processor=None,
    ) -> None:
        self.processor = processor
        self.teacher_processor = teacher_processor or processor
        self.tokenizer = processor.tokenizer
        self.max_prompt_length = max_prompt_length
        self.include_reference_answer = include_reference_answer
        self.image_root = image_root
        self.max_image_pixels = max_image_pixels
        self.tokenizer.padding_side = "left"

    def _format_examples(self, examples: Any) -> str:
        if not examples:
            return "N/A"
        if isinstance(examples, str):
            return examples
        if not isinstance(examples, (list, tuple)):
            examples = [examples]

        formatted = []
        for index, item in enumerate(examples, start=1):
            if isinstance(item, dict):
                q = item.get("question") or item.get("prompt") or item.get("text") or ""
                a = item.get("answer") or item.get("response") or item.get("reference_answer") or ""
                label = item.get("label")
                if label is None:
                    label = "harmful" if item.get("is_harmful") else "safe/helpful"
                score = item.get("score") or item.get("fused_score") or item.get("image_score") or item.get("text_score")
                relation = item.get("relation_to_current_case")
                role = item.get("experience_role")
                parts = [f"Example {index}"]
                if score is not None:
                    parts.append(f"score={float(score):.4f}")
                if label:
                    parts.append(f"label={label}")
                if role:
                    parts.append(f"role={role}")
                header = " | ".join(parts)
                body = f"{header}\nQuestion: {q}\nAnswer: {a}"
                if relation:
                    body += f"\nRelation to current case: {relation}"
                formatted.append(body)
            else:
                formatted.append(f"Example {index}\n{item}")
        return "\n\n".join(formatted)

    def _format_example_responses(self, examples: Any) -> str:
        if not examples:
            return "N/A"
        if isinstance(examples, str):
            return examples
        if not isinstance(examples, (list, tuple)):
            examples = [examples]

        formatted = []
        for index, item in enumerate(examples, start=1):
            if isinstance(item, dict):
                answer = item.get("answer") or item.get("response") or item.get("reference_answer") or ""
                formatted.append(f"Example {index} response:\n{answer}")
            else:
                formatted.append(f"Example {index} response:\n{item}")
        return "\n\n".join(formatted)

    def _format_example_questions(self, examples: Any) -> str:
        if not examples:
            return "N/A"
        if isinstance(examples, str):
            return examples
        if not isinstance(examples, (list, tuple)):
            examples = [examples]

        formatted = []
        for index, item in enumerate(examples, start=1):
            if isinstance(item, dict):
                question = item.get("question") or item.get("prompt") or item.get("text") or ""
                formatted.append(f"Example {index}\nImage\nQuestion\n{question}")
            else:
                formatted.append(f"Example {index}\n{item}")
        return "\n\n".join(formatted)

    def _append_multimodal_examples(self, content: list[dict[str, Any]], examples: Any) -> None:
        if examples and not isinstance(examples, str):
            if not isinstance(examples, (list, tuple)):
                examples = [examples]
            for index, item in enumerate(examples, start=1):
                if not isinstance(item, dict):
                    continue
                content.extend(
                    [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                f"Example {index}\n"
                                "Image\n"
                                f"Question\n{item.get('question') or item.get('prompt') or item.get('text') or ''}\n\n"
                                f"Response\n{item.get('answer') or item.get('response') or item.get('reference_answer') or ''}\n\n"
                            ),
                        },
                    ]
                )
        else:
            content.append({"type": "text", "text": f"{self._format_examples(examples)}\n\n"})

    def _student_messages(self, question: str, examples: Any) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Current Image\n\n"},
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"\n\nCurrent Question\n\n{question}\n\n"
                    "Retrieved Examples\n\n"
                ),
            },
        ]
        self._append_multimodal_examples(content, examples)
        content.append(
            {
                "type": "text",
                "text": (
                    "Instruction\n\n"
                    "Use the retrieved examples as prior experience to determine an appropriate response for the current case.\n"
                    "Keep the reasoning concise and stop as soon as the response strategy is clear.\n"
                    "Please answer the current question."
                ),
            }
        )
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _teacher_messages(
        self,
        question: str,
        examples: Any,
        knowledge: str,
        answer: str,
        demonstration: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Current Image\n\n"},
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"\n\nCurrent Question\n\n{question}\n\n"
                    "Retrieved Examples\n\n"
                ),
            },
        ]

        self._append_multimodal_examples(content, examples)

        thought = demonstration.strip() or "N/A"
        response = answer.strip() or "N/A"
        teacher_text = (
            "\nDemonstration\n\n"
            "<think>\n"
            f"{thought}\n"
            "</think>\n\n"
            f"{response}"
        )
        teacher_text = (
            "\nPrivileged Safety Knowledge\n\n"
            f"{knowledge.strip() or 'N/A'}\n\n"
            + teacher_text
        )
        teacher_text += (
            "\n\nInstruction\n\n"
            "Use the retrieved examples as prior experience to determine an appropriate response for the current case.\n"
            "The demonstration is not evidence for the current case.\n"
            "Do not mention or rely on it during reasoning.\n"
            "Use it only to learn the desired reasoning process and response style.\n"
            "Keep the reasoning concise and stop as soon as the response strategy is clear.\n"
            "Please answer the current question."
        )
        content.append({"type": "text", "text": teacher_text})
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _encode(self, texts: list[str], images: list[Image.Image]) -> dict[str, Any]:
        return self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_texts = []
        teacher_texts = []
        student_images = []
        teacher_images = []

        for example in features:
            question = str(_first_present(example, QUESTION_KEYS))
            image = _load_image(_first_present(example, IMAGE_KEYS), self.image_root, self.max_image_pixels)
            knowledge = str(_first_present(example, KNOWLEDGE_KEYS))
            answer = str(_first_present(example, ANSWER_KEYS))
            if not self.include_reference_answer:
                answer = ""
            demonstration = str(_first_present(example, DEMONSTRATION_KEYS))
            examples = _first_present(example, EXAMPLE_KEYS, [])
            retrieved_images = []
            if examples and not isinstance(examples, str):
                if not isinstance(examples, (list, tuple)):
                    examples = [examples]
                for item in examples:
                    if isinstance(item, dict) and item.get("image_path"):
                        retrieved_images.append(_load_image(item["image_path"], self.image_root, self.max_image_pixels))

            student_texts.append(
                    self.processor.apply_chat_template(
                    self._student_messages(question, examples),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            teacher_texts.append(
                self.teacher_processor.apply_chat_template(
                    self._teacher_messages(question, examples, knowledge, answer, demonstration),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            student_images.append([image, *retrieved_images])
            teacher_images.append([image, *retrieved_images])

        student = self._encode_with(self.processor, student_texts, student_images, self.max_prompt_length)
        teacher = self._encode_with(self.teacher_processor, teacher_texts, teacher_images, self.max_prompt_length)

        batch = {
            "student_input_ids": student["input_ids"],
            "student_attention_mask": student["attention_mask"],
            "student_prompt_length": student["input_ids"].shape[1],
            "student_prompt_lengths": student["attention_mask"].sum(dim=1),
            "teacher_input_ids": teacher["input_ids"],
            "teacher_attention_mask": teacher["attention_mask"],
            "teacher_prompt_length": teacher["input_ids"].shape[1],
            "teacher_prompt_lengths": teacher["attention_mask"].sum(dim=1),
        }

        for key, value in student.items():
            if key not in ("input_ids", "attention_mask"):
                batch[f"student_{key}"] = value
        for key, value in teacher.items():
            if key not in ("input_ids", "attention_mask"):
                batch[f"teacher_{key}"] = value

        return batch

    @staticmethod
    def _encode_with(processor, texts, images, max_prompt_length):
        return processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
