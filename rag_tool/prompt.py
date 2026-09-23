from __future__ import annotations

from typing import Any


class PromptTemplate:
    def __init__(self, system_prompt: str = "", user_prompt: str = "Question: {question}\n\nReferences:\n{reference}"):
        self.system_prompt, self.user_prompt = system_prompt, user_prompt

    def get_string(self, question: str, retrieval_result: list[dict[str, Any]] | None = None, **kwargs: Any) -> str:
        reference = "\n".join(str(doc.get("contents", doc.get("text", ""))) for doc in (retrieval_result or []))
        body = self.user_prompt.format(question=question, reference=reference, **kwargs)
        return f"{self.system_prompt}\n{body}".strip() if self.system_prompt else body
