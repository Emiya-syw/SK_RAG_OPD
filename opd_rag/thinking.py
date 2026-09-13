"""Chat-template helpers for running Qwen3-VL-Thinking without reasoning tokens."""

from __future__ import annotations

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


def chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    """Build Transformers processor kwargs for the requested Qwen3-VL mode."""
    if enable_thinking:
        return {"template_kwargs": {"enable_thinking": True}}
    return {"chat_template": thinking_off_template()}
