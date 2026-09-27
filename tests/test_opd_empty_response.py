"""CPU regressions for failed rollouts padded to the actor DP size.

Run with PYTHONPATH=verl-runtime python -m unittest discover -s tests
-p test_opd_empty_response.py -v. No models or GPUs are required.
"""

import asyncio
import unittest
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.trainer.distillation.losses import compute_distillation_loss_range, distillation_ppo_loss
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.metric import Metric, reduce_metrics
from verl.utils.py_functional import append_to_dict


def run_loss(valid, mode="k3", nested=False):
    student = torch.tensor([-1.0, -0.8, -1.2], requires_grad=True)
    mask = torch.tensor([[int(valid)]])
    if nested:
        mask = torch.nested.nested_tensor([mask[0]])
    data = {
        "prompts": torch.tensor([[1, 2]]),
        "responses": torch.tensor([[3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "response_mask": mask,
        "teacher_logprobs": torch.full((3, 1), -0.5),
        "dp_size": 4,
        "batch_num_tokens": 3,
        "global_batch_size": 3,
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss_config = SimpleNamespace(
        loss_mode=mode, loss_max_clamp=10.0, use_policy_gradient=False, use_task_rewards=False, topk=2
    )
    output = {"log_probs": student}
    if mode == "forward_kl_topk":
        output.update(
            distillation_losses=student.square(),
            student_mass=student.exp(),
            teacher_mass=torch.full((3,), 0.7),
            overlap_count=torch.ones(3),
            overlap_token_advantage=student.square(),
        )
    loss, metrics = distillation_ppo_loss(config, SimpleNamespace(distillation_loss=loss_config), output, data)
    loss.backward()
    return loss.detach(), student.grad, metrics


class EmptyResponseTests(unittest.TestCase):
    def test_three_real_ranks_and_one_padding_rank(self):
        # Reproduce the reported global batch: 3 successful rollouts + padding.
        outputs = [run_loss(valid) for valid in (True, True, True, False)]
        loss = sum(result[0] for result in outputs) / 4
        grad = sum(result[1] for result in outputs) / 4
        student = torch.tensor(-0.8, requires_grad=True)
        delta = torch.tensor(-0.5) - student
        expected_loss = delta.exp() - delta - 1
        expected_loss.backward()
        torch.testing.assert_close(loss, expected_loss.detach())
        torch.testing.assert_close(grad[1], student.grad)
        torch.testing.assert_close(grad[[0, 2]], torch.zeros(2))

    def test_padding_keeps_backward_and_zero_gradient(self):
        for mode in ("k3", "forward_kl_topk"):
            for nested in (False, True):
                with self.subTest(mode=mode, nested=nested):
                    loss, grad, metrics = run_loss(False, mode, nested)
                    self.assertEqual(loss.item(), 0.0)
                    torch.testing.assert_close(grad, torch.zeros_like(grad))
                    self.assertEqual(set(metrics), {"distillation/loss"})

    def test_valid_and_padding_dp_metrics(self):
        # Match engine microbatch accumulation, DP gather and final reduction.
        # Different ranks can omit different numbers of diagnostic observations.
        for mode in ("k3", "forward_kl_topk"):
            rank_metrics = []
            valid_loss, valid_grad, valid_metrics = run_loss(True, mode)
            self.assertGreater(valid_loss.item(), 0)
            self.assertGreater(valid_grad.abs().sum().item(), 0)
            for flags in ((True, True), (True, False), (False, True), (False, False)):
                accumulated = {}
                for valid in flags:
                    append_to_dict(accumulated, run_loss(valid, mode)[2])
                rank_metrics.append(accumulated)
            gathered = {}
            for metrics in rank_metrics:
                for key, value in metrics.items():
                    gathered.setdefault(key, []).append(value)
            reduced = {}
            for key, values in gathered.items():
                reduced[key] = (
                    Metric.aggregate_dp(values) if isinstance(values[0], Metric) else list(chain.from_iterable(values))
                )
            reduced = reduce_metrics(reduced)
            for key, value in reduced.items():
                self.assertTrue(torch.isfinite(torch.tensor(value)), key)
                if key != "distillation/loss":
                    self.assertAlmostEqual(value, valid_metrics[key])
            # Four real microbatches across four ranks: padding adds no loss.
            self.assertAlmostEqual(reduced["distillation/loss"], valid_loss.item())

    def test_range_excludes_masked_values(self):
        metrics = compute_distillation_loss_range(
            torch.tensor([[-100.0, 2.0, 5.0, 999.0]]), torch.tensor([[0, 1, 1, 0]])
        )
        self.assertEqual(metrics, {"distillation/loss_min": 2.0, "distillation/loss_max": 5.0})
        self.assertEqual(compute_distillation_loss_range(torch.empty(1, 0), torch.empty(1, 0)), {})


class PatchSizeTests(unittest.TestCase):
    def test_dataset_and_rollout_use_same_patch_size(self):
        config_path = Path(__file__).resolve().parents[1] / "verl-runtime/verl/trainer/config/data/legacy_data.yaml"
        for processor_size, override, expected in ((16, None, 16), (14, None, 14), (16, 14, 14)):
            with self.subTest(processor_size=processor_size, override=override):
                config = OmegaConf.load(config_path)
                if override is not None:
                    config.image_patch_size = override
                processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=processor_size))
                with patch.object(RLHFDataset, "_download"), patch.object(RLHFDataset, "_read_files_and_tokenize"):
                    dataset = RLHFDataset([], None, config, processor)
                extract = AsyncMock(return_value=(None, None, None))
                loop = SimpleNamespace(
                    processor=processor, data_config=config,
                    dataset_cls=SimpleNamespace(process_multi_modal_info=extract),
                )
                asyncio.run(AgentLoopBase.process_multi_modal_info(loop, []))
                self.assertEqual(dataset.image_patch_size, expected)
                self.assertEqual(extract.call_args.kwargs["image_patch_size"], dataset.image_patch_size)


if __name__ == "__main__":
    unittest.main()
