from opd_rag.thinking import chat_template_kwargs, thinking_off_template


def test_thinking_off_template_closes_reasoning_before_generation() -> None:
    template = thinking_off_template()

    assert "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n" in template
    assert chat_template_kwargs(False) == {"chat_template": template}


def test_thinking_on_uses_native_template_switch() -> None:
    assert chat_template_kwargs(True) == {"template_kwargs": {"enable_thinking": True}}
