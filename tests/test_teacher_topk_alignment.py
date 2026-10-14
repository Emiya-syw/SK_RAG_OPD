"""Verify multimodal teacher length realignment keeps the response suffix safe."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager


class FakeClient:
    def __init__(self, rows, scored_ids=None, sampled_logprobs=None):
        self.rows = rows
        self.scored_ids = scored_ids
        self.sampled_logprobs = sampled_logprobs

    async def generate(self, **kwargs):
        width = len(self.rows[0]) if self.rows else 1
        extra_fields = {
            'prompt_ids': self.rows,
            'prompt_logprobs': [[-1.0] * width for _ in self.rows],
            'scored_prompt_token_ids': self.scored_ids,
        }
        if self.sampled_logprobs is not None:
            extra_fields['prompt_sampled_logprobs'] = self.sampled_logprobs
        return SimpleNamespace(extra_fields=extra_fields)


def manager(client, mode='forward_kl_topk'):
    obj = AsyncTeacherLLMServerManager.__new__(AsyncTeacherLLMServerManager)
    obj.teacher_model_configs = {'teacher': SimpleNamespace(inference=SimpleNamespace(temperature=1.0))}
    obj.teacher_client = {'teacher': client}
    obj.distillation_loss_config = SimpleNamespace(
        topk=2, loss_mode=mode, loss_settings=SimpleNamespace(use_topk=True))
    return obj


def score(client, expected, response_len=2):
    return asyncio.run(manager(client).compute_teacher_logprobs_single(
        expected, response_len, multi_modal_data={'images': ['image']}))


class TeacherTopkAlignmentTests(unittest.TestCase):
    def test_shorter_multimodal_prompt_with_topk(self):
        # vLLM compresses the image span from three student tokens to one.
        expected = [11, 12, 13, 14, 21, 22]
        rows = [[101, 102], [111, 112], [121, 122], [131, 132]]
        ids, logs = score(FakeClient(rows, [11, 14, 21, 22]), expected)
        self.assertEqual(ids.shape, (6, 2))
        self.assertEqual(ids[:2].tolist(), [[0, 0], [0, 0]])
        self.assertEqual(ids[-2:].tolist(), rows[-2:])
        self.assertEqual(logs[:2].tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_longer_multimodal_prompt_with_topk(self):
        expected = [11, 14, 21, 22]
        rows = [[i, i+100] for i in range(6)]
        ids, _ = score(FakeClient(rows, [11, 12, 13, 14, 21, 22]), expected)
        self.assertEqual(ids.tolist(), rows[-4:])

    def test_entropy_kl_preserves_sampled_logprob_outside_teacher_topk(self):
        expected = [11, 12, 13, 14, 21, 22]
        rows = [[101, 102], [111, 112], [121, 122], [131, 132]]
        client = FakeClient(rows, [11, 14, 21, 22], [-.1, -.2, -4., 0.])
        ids, logs = asyncio.run(manager(client, mode='entropy_kl').compute_teacher_logprobs_single(
            expected, 2, multi_modal_data={'images': ['image']}))
        self.assertEqual(ids.shape, (6, 3))
        self.assertEqual(logs.shape, (6, 3))
        self.assertEqual(ids[-2:, :2].tolist(), rows[-2:])
        # Rows immediately preceding the two response tokens are their scores.
        self.assertAlmostEqual(logs[-3, -1].item(), -.2)
        self.assertAlmostEqual(logs[-2, -1].item(), -4.)
        self.assertEqual(logs[:2, -1].tolist(), [0., 0.])

    def test_entropy_kl_requires_sampled_logprob(self):
        with self.assertRaisesRegex(ValueError, 'requires the teacher logprob'):
            asyncio.run(manager(FakeClient([[1, 2]] * 4), mode='entropy_kl').compute_teacher_logprobs_single(
                [11, 14, 21, 22], 2))

    def test_response_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'response-token suffix differs'):
            score(FakeClient([[1, 2]] * 4, [11, 14, 21, 99]), [11, 12, 13, 14, 21, 22])

    def test_topk_without_actual_ids_fails(self):
        with self.assertRaisesRegex(ValueError, 'requires scored_prompt_token_ids'):
            score(FakeClient([[1, 2]] * 4), [11, 12, 13, 14, 21, 22])

    def test_text_only_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'cannot be safely aligned'):
            asyncio.run(manager(FakeClient([[1, 2]] * 4, [11, 14, 21, 22])).compute_teacher_logprobs_single(
                [11, 12, 13, 14, 21, 22], 2))

    def test_estimator_fallback(self):
        expected = [11, 12, 13, 14, 21, 22]
        # Existing estimator protocol: row i scores original token i+1.
        rows = [[14], [21], [22], [0]]
        client = FakeClient(rows)
        client.scored_ids = None
        with patch('verl.experimental.teacher_loop.teacher_manager._get_teacher_sampling_params', return_value={}):
            ids, _ = score(client, expected)
        self.assertEqual(ids.shape, (6, 1))


if __name__ == '__main__':
    unittest.main()
