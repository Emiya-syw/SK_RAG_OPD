from __future__ import annotations

from typing import Any


class MultiRetriever:
    def __init__(self, retrievers: list[Any], merge_method: str = "concat", top_k: int = 5, reranker=None):
        if merge_method not in {"concat", "rrf", "rerank"}:
            raise ValueError("merge_method must be concat, rrf, or rerank")
        self.retrievers, self.merge_method, self.top_k, self.reranker = retrievers, merge_method, top_k, reranker

    def batch_search(self, queries, num: int | None = None, return_score: bool = False):
        top_k = num or self.top_k
        result_groups = [r.batch_search(queries, num=top_k) for r in self.retrievers]
        merged = []
        for qpos in range(len(queries) if not isinstance(queries, str) else 1):
            groups = [group[qpos] for group in result_groups]
            if self.merge_method == "rerank" and self.reranker:
                docs = self.reranker.rerank([queries] if isinstance(queries, str) else [queries[qpos]], [sum(groups, [])], top_k=top_k)[0]
            elif self.merge_method == "rrf":
                scores = {}
                values = {}
                for group in groups:
                    for rank, doc in enumerate(group):
                        key = str(doc.get("id", doc.get("doc_id", doc.get("contents", ""))))
                        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
                        values[key] = doc
                docs = [dict(values[key], score=score) for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            else:
                docs = sorted(sum(groups, []), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            merged.append(docs)
        return merged
