from __future__ import annotations

from typing import Any


class Reranker:
    def __init__(self, model_path: str, device: str = "auto", max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.max_length, self.batch_size = torch, max_length, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).eval()
        target = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.model.to(target)

    def rerank(self, queries: list[str] | str, docs: list[list[dict[str, Any]]] | list[dict[str, Any]], top_k: int | None = None):
        queries = [queries] if isinstance(queries, str) else queries
        if docs and isinstance(docs[0], dict):
            docs = [docs]
        output = []
        for query, candidates in zip(queries, docs):
            pairs = [[query, item.get("contents", item.get("text", ""))] for item in candidates]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                scores = self.model(**inputs).logits.reshape(-1).float().cpu().tolist()
            ranked = sorted((dict(doc, score=float(score)) for doc, score in zip(candidates, scores)), key=lambda x: x["score"], reverse=True)
            output.append(ranked[:top_k] if top_k else ranked)
        return output
