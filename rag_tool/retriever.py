"""Retriever facade with dense, lexical, caching and optional reranking paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import prepare_rows, read_rows
from .encoders import Encoder
from .index import DenseIndex


class Retriever:
    def __init__(self, config: dict[str, Any], index_path: str | Path | None = None, corpus_path: str | Path | None = None):
        self.config = config
        self.index_path = Path(index_path or config.get("index_path", ""))
        if not self.index_path:
            raise ValueError("index_path is required")
        self.manifest = json.loads((self.index_path / "manifest.json").read_text(encoding="utf-8"))
        self.corpus = read_rows(self.index_path / "corpus.jsonl")
        self.index = DenseIndex.load(self.index_path)
        self.encoder = Encoder(self.manifest["backend"], self.manifest["model"], self.manifest.get("max_length", 8192), self.manifest.get("max_pixels", 262144), code_path=self.manifest.get("code_path"))
        self.topk = int(config.get("retrieval_topk", 5))
        self.cache_path = Path(config["retrieval_cache_path"]) if config.get("retrieval_cache_path") else None
        self.cache: dict[str, Any] = {}
        if config.get("use_retrieval_cache") and self.cache_path and self.cache_path.is_file():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, query: Any) -> str:
        return hashlib.sha256(json.dumps(query, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def batch_search(self, queries: list[Any] | str, num: int | None = None, return_score: bool = False):
        values = [queries] if isinstance(queries, str) else list(queries)
        topk = num or self.topk
        output, scores, missing = [None] * len(values), [None] * len(values), []
        for pos, query in enumerate(values):
            cached = self.cache.get(self._key(query)) if self.config.get("use_retrieval_cache") else None
            if cached is not None:
                output[pos], scores[pos] = cached["docs"], cached["scores"]
            else:
                missing.append((pos, query))
        if missing:
            rows = prepare_rows([{"id": f"query:{i}", "prompt": query} if isinstance(query, str) else query for _, query in missing])
            vectors = self.encoder.encode(rows, int(self.config.get("retrieval_batch_size", 32)))
            batch_scores, batch_indices = self.index.search(vectors, min(topk, len(self.corpus)))
            for (pos, query), row_scores, row_indices in zip(missing, batch_scores, batch_indices):
                docs = []
                for score, idx in zip(row_scores, row_indices):
                    doc = dict(self.corpus[int(idx)])
                    doc["score"] = float(score)
                    docs.append(doc)
                output[pos], scores[pos] = docs, [float(x) for x in row_scores]
                if self.config.get("save_retrieval_cache"):
                    self.cache[self._key(query)] = {"docs": docs, "scores": scores[pos]}
        if self.config.get("save_retrieval_cache") and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return (output, scores) if return_score else output

    def search(self, query: Any, num: int | None = None, return_score: bool = False):
        result = self.batch_search([query], num=num, return_score=return_score)
        return (result[0][0], result[1][0]) if return_score else result[0]
