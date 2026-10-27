"""Exercise hybrid loss transport, independent clipping, masking and gradients."""
import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import compute_topk_loss, distillation_loss, distillation_ppo_loss
from verl.workers.config.distillation import DistillationLossConfig


class K3JSDTests(unittest.TestCase):
    def fixture(self, mode, coefficient=1., cap=10., masked=False, outer_coefficient=1.):
        torch.manual_seed(31)
        logits = torch.randn(1, 5, 6, requires_grad=True)
        teacher = torch.tensor([.45, .35, .1, .05, .03, .02]).log().repeat(1, 5, 1).requires_grad_()
        labels = torch.tensor([[5, 4, 3, 5, 4]])  # outside teacher loss top-2
        cfg = DistillationLossConfig(loss_mode=mode, use_policy_gradient=False, use_task_rewards=False,
                                     topk=2, entropy_topk=4, k3_jsd_lambda=coefficient,
                                     entropy_jsd_normalize=True, loss_max_clamp=cap,
                                     distillation_loss_coef=outer_coefficient)
        count = 4 if mode == 'k3_entropy_jsd' else 2
        top_logs, ids = teacher.topk(count)
        sampled_t = teacher.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        rows = torch.cat([top_logs[0], sampled_t[0, :, None]], dim=-1)
        row_ids = torch.cat([ids[0], torch.zeros(5, 1, dtype=torch.long)], dim=-1)
        teacher_data = {
            'teacher_logprobs': torch.nested.nested_tensor([rows], layout=torch.jagged),
            'teacher_ids': torch.nested.nested_tensor([row_ids], layout=torch.jagged),
        }
        raw = compute_topk_loss(SimpleNamespace(strategy='fsdp'), SimpleNamespace(distillation_loss=cfg),
                                teacher_data, logits, 'thd')
        sampled_s = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
        output = {k: v.squeeze(0) for k, v in raw.items()}
        output['log_probs'] = sampled_s
        mask = torch.zeros(1, 3) if masked else torch.tensor([[1., 0., 1.]])
        data = TensorDict({
            'prompts': torch.ones(1, 2, dtype=torch.long),
            'responses': torch.ones(1, 3, dtype=torch.long),
            'attention_mask': torch.ones(1, 5, dtype=torch.long), 'response_mask': mask,
            'teacher_logprobs': rows,
            'dp_size': 1, 'batch_num_tokens': 2, 'global_batch_size': 1,
        }, batch_size=[])
        actor = SimpleNamespace(loss_agg_mode='token-mean', global_batch_info={}, loss_scale_factor=None)
        return cfg, actor, data, output, logits, teacher, sampled_t[0]

    def test_weighted_sum_and_gradient_with_sampled_tokens_outside_topk(self):
        for mode in ('k3_jsd', 'k3_entropy_jsd'):
            for coefficient in (0., 1., 3.):
                with self.subTest(mode=mode, coefficient=coefficient):
                    cfg, actor, data, out, x, teacher, t = self.fixture(mode, coefficient)
                    loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
                    delta = (t.detach()-out['log_probs']).clamp(-20, 20)
                    k3 = (delta.exp()-delta-1).clamp(-10, 10)
                    jsd = out['distillation_losses'].clamp(-10, 10)
                    expected = (k3+coefficient*jsd)[[1, 3]].mean()
                    torch.testing.assert_close(loss, expected)
                    got = torch.autograd.grad(loss, (x, teacher), retain_graph=True, allow_unused=True)
                    want = torch.autograd.grad(expected, x, retain_graph=True)[0]
                    torch.testing.assert_close(got[0], want)
                    self.assertIsNone(got[1])
                    torch.testing.assert_close(got[0][0, [0, 2, 4]], torch.zeros_like(got[0][0, [0, 2, 4]]))
                    self.assertAlmostEqual(metrics['distillation/jsd_lambda'], coefficient)
                    self.assertAlmostEqual(metrics['distillation/k3_component'], k3[[1, 3]].mean().item())
                    self.assertAlmostEqual(metrics['distillation/weighted_jsd_component'],
                                           coefficient*jsd[[1, 3]].mean().item(), places=6)

    def test_zero_lambda_matches_standalone_k3_loss_and_gradient(self):
        for mode in ('k3_jsd', 'k3_entropy_jsd'):
            for cap in (None, .1, 10.):
                cfg, actor, data, out, x, _, t = self.fixture(mode, 0., cap)
                hybrid, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
                plain_cfg = DistillationLossConfig(loss_mode='k3', use_policy_gradient=False, loss_max_clamp=cap)
                plain_data = data.clone()
                plain_data['teacher_logprobs'] = t[:, None]
                plain, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=plain_cfg),
                                             {'log_probs': out['log_probs']}, plain_data)
                torch.testing.assert_close(hybrid, plain)
                torch.testing.assert_close(torch.autograd.grad(hybrid, x, retain_graph=True)[0],
                                           torch.autograd.grad(plain, x)[0])

    def test_lambda_scales_auxiliary_gradient_after_component_clipping(self):
        cfg, actor, data, out, _, _, _ = self.fixture('k3_jsd', 20., .5)
        # Isolate a JSD component below its cap but above the cap once weighted.
        jsd = torch.full((5,), .2, requires_grad=True)
        out['distillation_losses'] = jsd
        loss, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
        grad = torch.autograd.grad(loss, jsd)[0]
        torch.testing.assert_close(grad, torch.tensor([0., 10., 0., 10., 0.]))
        self.assertGreater(loss.item(), cfg.loss_max_clamp)

    def test_fully_masked_and_nested_padding(self):
        for mode in ('k3_jsd', 'k3_entropy_jsd'):
            for nested in (False, True):
                cfg, actor, data, out, x, _, _ = self.fixture(mode, masked=True)
                if nested:
                    for key in ('prompts', 'responses', 'response_mask'):
                        data[key] = torch.nested.nested_tensor([data[key][0]], layout=torch.jagged)
                loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
                self.assertEqual(loss.item(), 0.)
                self.assertEqual(metrics['distillation/k3_component'], 0.)
                self.assertEqual(metrics['distillation/jsd_component'], 0.)
                self.assertEqual(metrics['distillation/jsd_clipfrac'], 0.)
                torch.testing.assert_close(torch.autograd.grad(loss, x)[0], torch.zeros_like(x))

    def test_lambda_applies_without_task_rewards(self):
        # The outer coefficient is ignored in pure distillation; lambda still applies.
        cfg, actor, data, out, _, _, _ = self.fixture('k3_entropy_jsd', 2., outer_coefficient=99.)
        expected, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
        got, metrics = distillation_ppo_loss(actor, SimpleNamespace(distillation_loss=cfg), out, data)
        torch.testing.assert_close(got, expected)
        self.assertEqual(metrics['distillation/jsd_lambda'], 2.)

    def test_coefficient_validation_and_hydra_override(self):
        from hydra import compose, initialize_config_module
        from verl.utils.config import omega_conf_to_dataclass
        for mode in ('k3_jsd', 'k3_entropy_jsd'):
            default = DistillationLossConfig(loss_mode=mode, use_policy_gradient=False)
            self.assertEqual(default.k3_jsd_lambda, 1.)
            for value in (-.1, float('nan'), float('inf')):
                with self.assertRaisesRegex(ValueError, 'k3_jsd_lambda'):
                    DistillationLossConfig(loss_mode=mode, use_policy_gradient=False, k3_jsd_lambda=value)
            with initialize_config_module(config_module='verl.trainer.config', version_base=None):
                config = compose(config_name='ppo_trainer', overrides=[
                    f'distillation.distillation_loss.loss_mode={mode}',
                    'distillation.distillation_loss.use_policy_gradient=false',
                    'distillation.distillation_loss.k3_jsd_lambda=2.5',
                ])
            parsed = omega_conf_to_dataclass(config.distillation.distillation_loss)
            self.assertEqual(parsed.k3_jsd_lambda, 2.5)
            self.assertTrue(parsed.loss_settings.use_topk)


if __name__ == '__main__':
    unittest.main()
