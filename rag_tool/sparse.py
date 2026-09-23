"""Dependency-light BM25-style lexical retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


class BM25Index:
    def __init__(self, corpus: list[dict[str, Any]], field: str = "contents", k1: float = 1.5, b: float = 0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        self.tokens = [tokenize(row.get(field, row.get("text", ""))) for row in corpus]
        self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(token for row in self.tokens for token in set(row))
        n = len(self.tokens)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, queries: list[str] | str, top_k: int = 10):
        values = [queries] if isinstance(queries, str) else queries
        all_docs, all_scores = [], []
        for query in values:
            qtokens = tokenize(query)
            scores = []
            for tokens, length in zip(self.tokens, self.lengths):
                counts = Counter(tokens)
                score = 0.0
                for term in qtokens:
                    if term not in counts:
                        continue
                    tf = counts[term]
                    score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1)))
                scores.append(score)
            order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
            all_docs.append([[dict(self.corpus[i], score=float(scores[i])) for i in order]])
            all_scores.append([[float(scores[i]) for i in order]])
        return [x[0] for x in all_docs], [x[0] for x in all_scores]
