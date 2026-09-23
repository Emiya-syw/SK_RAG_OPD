from __future__ import annotations

from typing import Any

from .evaluator import Evaluator
from .generator import NullGenerator, TransformersGenerator
from .prompt import PromptTemplate


class SequentialPipeline:
    """query -> retrieval -> prompt -> generation -> evaluation."""
    def __init__(self, config: dict[str, Any], retriever, generator=None, prompt_template=None):
        self.config = config
        self.retriever = retriever
        self.generator = generator or (TransformersGenerator(config["generator_model_path"], config.get("device", "auto"), config.get("generation_params")) if config.get("generator_model_path") else NullGenerator())
        self.prompt_template = prompt_template or PromptTemplate()
        self.evaluator = Evaluator(config)

    def run(self, dataset, do_eval: bool = True):
        retrieval = self.retriever.batch_search(dataset.question, num=self.config.get("retrieval_topk"))
        dataset.update_output("retrieval_result", retrieval)
        prompts = [self.prompt_template.get_string(item.contents, docs) for item, docs in zip(dataset, retrieval)]
        dataset.update_output("prompt", prompts)
        dataset.update_output("pred", self.generator.generate(prompts))
        if do_eval:
            dataset.update_output("evaluation", [self.evaluator.evaluate(dataset)] * len(dataset))
        return dataset

    def retrieve(self, dataset):
        results, scores = self.retriever.batch_search(dataset.question, num=self.config.get("retrieval_topk"), return_score=True)
        dataset.update_output("retrieval_result", results)
        dataset.update_output("retrieval_scores", scores)
        return dataset
