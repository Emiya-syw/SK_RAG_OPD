"""Check exact entropy and top-k compaction for stock vLLM prompt logprobs."""

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE = (
    Path(__file__).resolve().parents[1]
    / "verl-runtime/verl/workers/rollout/vllm_rollout/prompt_entropy.py"
)
spec = importlib.util.spec_from_file_location("prompt_entropy", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PromptEntropyTests(unittest.TestCase):
    def test_full_entropy_and_topk_keep_sampled_token_outside_topk(self):
        probabilities = [0.5, 0.25, 0.125, 0.125]
        row = {
            token_id: SimpleNamespace(logprob=math.log(probability), rank=rank)
            for rank, (token_id, probability) in enumerate(zip([7, 8, 9, 10], probabilities), 1)
        }
        ids, top_logprobs, sampled_lp, entropy = module.compact_full_prompt_logprobs(row, 10, 2)
        self.assertEqual(ids, [7, 8])
        self.assertEqual(top_logprobs, [math.log(0.5), math.log(0.25)])
        self.assertAlmostEqual(sampled_lp, math.log(0.125))
        self.assertAlmostEqual(entropy, -sum(p * math.log(p) for p in probabilities))

    def test_missing_rank_fails_instead_of_returning_invalid_topk(self):
        row = {7: SimpleNamespace(logprob=math.log(0.5), rank=1),
               8: SimpleNamespace(logprob=math.log(0.5), rank=3)}
        with self.assertRaisesRegex(ValueError, "missing ranked top-k"):
            module.compact_full_prompt_logprobs(row, 7, 2)


if __name__ == "__main__":
    unittest.main()
