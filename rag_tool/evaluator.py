from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, answers: list[str]) -> float:
    pred = normalize_answer(pred)
    return float(any(pred == normalize_answer(answer) for answer in answers))


def f1(pred: str, answers: list[str]) -> float:
    pred_tokens = normalize_answer(pred).split()
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        common = Counter(pred_tokens) & Counter(gold)
        count = sum(common.values())
        if count:
            best = max(best, 2 * count / (len(pred_tokens) + len(gold)))
    return best


class Evaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate(self, dataset) -> dict[str, float]:
        results = {}
        answers = [getattr(item, "golden_answers", getattr(item, "answer", [])) for item in dataset]
        preds = [item.output.get("pred", "") for item in dataset]
        answers = [[x] if isinstance(x, str) else x for x in answers]
        if "em" in self.config.get("metrics", []):
            results["em"] = sum(exact_match(p, a) for p, a in zip(preds, answers)) / max(len(preds), 1)
        if "f1" in self.config.get("metrics", []):
            results["f1"] = sum(f1(p, a) for p, a in zip(preds, answers)) / max(len(preds), 1)
        return results

    def retrieval_recall(self, dataset, answer_keys=("id", "doc_id", "knowledge_id"), top_k=None) -> float:
        hits = 0
        for item in dataset:
            gold = {str(item.data[key]) for key in answer_keys if key in item.data}
            docs = item.output.get("retrieval_result", [])[:top_k]
            retrieved = {str(doc.get(key)) for doc in docs for key in answer_keys if doc.get(key) is not None}
            hits += bool(gold & retrieved)
        return hits / max(len(dataset), 1)
