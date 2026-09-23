"""Generator adapters. Retrieval-only experiments can use NullGenerator."""
from __future__ import annotations

from typing import Any


class NullGenerator:
    def generate(self, prompts: list[str]) -> list[str]:
        return ["" for _ in prompts]


class TransformersGenerator:
    def __init__(self, model_path: str, device: str = "auto", generation_params: dict[str, Any] | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(target).eval()
        self.params = generation_params or {"max_new_tokens": 128}

    def generate(self, prompts: list[str]) -> list[str]:
        output = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                tokens = self.model.generate(**inputs, **self.params)
            output.append(self.tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return output
