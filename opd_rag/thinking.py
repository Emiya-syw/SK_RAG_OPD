"""Model-aware chat templates shared by training and evaluation."""

from __future__ import annotations

import json

from functools import lru_cache
from pathlib import Path
from typing import Any


THINKING_OFF_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "chat_templates" / "qwen3_vl_thinking_off.jinja"
)


@lru_cache(maxsize=1)
def thinking_off_template() -> str:
    """Return a Qwen3-VL template that closes an empty thinking block before generation."""
    return THINKING_OFF_TEMPLATE_PATH.read_text(encoding="utf-8")


def chat_template_kwargs(enable_thinking: bool, model_type: str = "qwen3_vl") -> dict[str, Any]:
    """Use native templates outside Qwen3; preserve the existing Qwen3 thinking switch."""
    if model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
        return {}
    if enable_thinking:
        return {"template_kwargs": {"enable_thinking": True}}
    return {"chat_template": thinking_off_template()}


def resolve_model_type(model_path: str | Path) -> str:
    """Read local configs without ML dependencies; resolve Hub IDs with AutoConfig."""
    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return config["model_type"]
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(model_path), trust_remote_code=True).model_type


def validate_distillation_pair(student: str, teacher: str) -> None:
    """OPD consumes student token IDs directly, so vocabulary mappings must agree."""
    if Path(student).resolve() == Path(teacher).resolve():
        return
    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("OPD requires identical student/teacher token-to-ID vocabularies; "
                         "choose a teacher compatible with the student tokenizer.")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student_tokenizer, key) != getattr(teacher_tokenizer, key):
            raise ValueError(f"OPD student/teacher {key} differs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--teacher")
    args = parser.parse_args()
    if args.teacher:
        validate_distillation_pair(args.model, args.teacher)
    print(resolve_model_type(args.model))
