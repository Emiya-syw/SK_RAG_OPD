"""CPU numerical and runtime wiring checks for the four distribution objectives."""
import unittest
from types import SimpleNamespace

import torch

from verl.trainer.distillation.distribution_losses import (
    DISTRIBUTION_LOSS_MODES, compute_distribution_from_logits, distribution_loss,
)
from verl.workers.config.distillation import DistillationLossConfig


def config(mode, **kwargs):
    return DistillationLossConfig(loss_mode=mode, use_policy_gradient=False, **kwargs)


def kl(p, q):
    return (p * (p.log() - q.log())).sum(-1)


class DistributionLossTests(unittest.TestCase):
    def test_formulas_and_gradients(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            with self.subTest(mode=mode):
                x = torch.tensor([[1., -1., .5], [-2., 1., 0.]], dtype=torch.double, requires_grad=True)
                y = torch.tensor([[0., 0., 0.], [2., -1., -2.]], dtype=torch.double, requires_grad=True)
                s, t = x.softmax(-1), y.softmax(-1).detach()
                c = config(mode)
                out = distribution_loss(x.log_softmax(-1), y.log_softmax(-1), c, include_tail=False)
                hs, ht = -(s * s.log()).sum(-1), -(t * t.log()).sum(-1)
                alpha = (.05 + .9 * torch.sigmoid(ht - hs)).detach()
                beta = 1 - alpha if mode == 'entropy_jsd' else torch.full_like(alpha, .5)
                m = beta[:, None] * t + (1-beta[:, None]) * s
                expected = {'forward_kl': kl(t, s), 'entropy_kl': alpha*kl(t, s)+(1-alpha)*kl(s, t),
                            'jsd': beta*kl(t, m)+(1-beta)*kl(s, m),
                            'entropy_jsd': beta*kl(t, m)+(1-beta)*kl(s, m)}[mode]
                torch.testing.assert_close(out['distillation_losses'], expected)
                got = torch.autograd.grad(out['distillation_losses'].sum(), (x, y), allow_unused=True, retain_graph=True)
                want = torch.autograd.grad(expected.sum(), x)[0]
                torch.testing.assert_close(got[0], want)
                self.assertIsNone(got[1])
                self.assertFalse(out['teacher_entropy'].requires_grad)
                self.assertFalse(out['student_entropy'].requires_grad)
                if mode == 'entropy_kl':
                    self.assertGreater(out['forward_weight'][0].item(), .5)
                    self.assertLess(out['forward_weight'][1].item(), .5)
                if mode == 'entropy_jsd':
                    self.assertLess(out['jsd_beta'][0].item(), .5)
                    self.assertGreater(out['jsd_beta'][1].item(), .5)

    def test_jsd_symmetry_bound_identity(self):
        a, b = torch.tensor([[.99, .01]]).log(), torch.tensor([[.02, .98]]).log()
        c = config('jsd')
        ab = distribution_loss(a, b, c, include_tail=False)['distillation_losses']
        ba = distribution_loss(b, a, c, include_tail=False)['distillation_losses']
        torch.testing.assert_close(ab, ba)
        self.assertTrue((ab <= torch.log(torch.tensor(2.))).all())
        for mode in DISTRIBUTION_LOSS_MODES:
            loss = distribution_loss(a, a, config(mode), include_tail=False)['distillation_losses']
            torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)

    def test_tail_bucket(self):
        s, t = torch.tensor([[.2, .3, .5]]), torch.tensor([[.6, .1, .3]])
        for mode in DISTRIBUTION_LOSS_MODES:
            c = config(mode)
            tail = distribution_loss(s[:, :2].log(), t[:, :2].log(), c)
            full = distribution_loss(s.log(), t.log(), c, include_tail=False)
            for key in ('distillation_losses', 'student_entropy', 'teacher_entropy'):
                torch.testing.assert_close(tail[key], full[key])
            torch.testing.assert_close(tail['teacher_mass'], torch.tensor([.7]))

    def test_chunked_gradients_and_empty(self):
        torch.manual_seed(17)
        for mode in DISTRIBUTION_LOSS_MODES:
            x = torch.randn(1, 7, 9, requires_grad=True)
            logs, ids = torch.randn_like(x).log_softmax(-1).topk(4)
            a = compute_distribution_from_logits(x, logs, ids, config(mode))
            b = compute_distribution_from_logits(x, logs, ids, config(mode, use_chunked_topk=True, chunked_topk_chunk_size=2))
            torch.testing.assert_close(a['distillation_losses'], b['distillation_losses'])
            ga = torch.autograd.grad(a['distillation_losses'].sum(), x, retain_graph=True)[0]
            gb = torch.autograd.grad(b['distillation_losses'].sum(), x)[0]
            torch.testing.assert_close(ga, gb)
            empty = compute_distribution_from_logits(x[:, :0], logs[:, :0], ids[:, :0], config(mode, use_chunked_topk=True))
            self.assertEqual(empty['distillation_losses'].shape, (1, 0))

    def test_extreme_probabilities(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            x = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
            out = distribution_loss(x.log_softmax(-1), torch.tensor([[-1000., 0., -1000.]]), config(mode))
            loss = out['distillation_losses'].sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_normalization(self):
        s, t = torch.tensor([[.9, .1]]).log(), torch.tensor([[.5, .5]]).log()
        a = distribution_loss(s, t, config('entropy_jsd'), include_tail=False)
        b = distribution_loss(s, t, config('entropy_jsd', entropy_jsd_normalize=True), include_tail=False)
        beta = a['jsd_beta']
        torch.testing.assert_close(b['distillation_losses'], a['distillation_losses'] / (beta*(1-beta)))

    def test_configuration_and_registration(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            self.assertTrue(config(mode).loss_settings.use_topk)
            for kwargs in ({'topk': 0}, {'jsd_beta': 1.}, {'jsd_beta': float('nan')},
                           {'entropy_gate_temperature': 0.}, {'entropy_gate_epsilon': .5}):
                with self.assertRaises(ValueError):
                    config(mode, **kwargs)
            with self.assertRaises(ValueError):
                DistillationLossConfig(loss_mode=mode, use_policy_gradient=True)
        self.assertEqual(DistillationLossConfig().loss_mode, 'k3')

    def test_fsdp_nested_dispatch(self):
        from verl.trainer.distillation.losses import compute_topk_loss
        torch.manual_seed(18)
        x = torch.randn(1, 5, 8, requires_grad=True)
        logs, ids = torch.randn_like(x).log_softmax(-1).topk(3)
        data = {'teacher_logprobs': torch.nested.nested_tensor([logs[0, :2], logs[0, 2:]], layout=torch.jagged),
                'teacher_ids': torch.nested.nested_tensor([ids[0, :2], ids[0, 2:]], layout=torch.jagged)}
        for mode in DISTRIBUTION_LOSS_MODES:
            c = config(mode)
            out = compute_topk_loss(SimpleNamespace(strategy='fsdp'), SimpleNamespace(distillation_loss=c), data, x, 'thd')
            expected = compute_distribution_from_logits(x, logs, ids, c)
            torch.testing.assert_close(out['distillation_losses'], expected['distillation_losses'])
            self.assertTrue(torch.isfinite(torch.autograd.grad(out['distillation_losses'].sum(), x)[0]).all())

    def test_response_mask_aggregation(self):
        from tensordict import TensorDict
        from verl.trainer.distillation.losses import distillation_loss
        x = torch.randn(1, 5, 8, requires_grad=True)
        logs, ids = torch.randn_like(x).log_softmax(-1).topk(3)
        data = TensorDict({
            'prompts': torch.ones(1, 2, dtype=torch.long),
            'responses': torch.ones(1, 3, dtype=torch.long),
            'attention_mask': torch.ones(1, 5, dtype=torch.long), 'response_mask': torch.tensor([[1., 0., 1.]]),
            'dp_size': 1, 'batch_num_tokens': 2, 'global_batch_size': 1,
        }, batch_size=[])
        actor = SimpleNamespace(loss_agg_mode='token-mean', global_batch_info={}, loss_scale_factor=None)
        for mode in DISTRIBUTION_LOSS_MODES:
            c = config(mode)
            raw = compute_distribution_from_logits(x, logs, ids, c)
            output = {k: v.squeeze(0) for k, v in raw.items()}
            loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=c), output, data)
            expected = raw['distillation_losses'][0, [1, 3]].mean()
            torch.testing.assert_close(loss, expected)
            self.assertAlmostEqual(metrics['distillation/teacher_entropy'], raw['teacher_entropy'][0, [1, 3]].mean().item())
            gradient = torch.autograd.grad(loss, x)[0]
            torch.testing.assert_close(gradient[0, [0, 2, 4]], torch.zeros_like(gradient[0, [0, 2, 4]]))

    def test_hydra_configuration(self):
        from hydra import compose, initialize_config_module
        with initialize_config_module(config_module='verl.trainer.config', version_base=None):
            cfg = compose(config_name='ppo_trainer', overrides=[
                'distillation.distillation_loss.loss_mode=entropy_jsd',
                'distillation.distillation_loss.use_policy_gradient=false',
                'distillation.distillation_loss.jsd_beta=0.5',
                'distillation.distillation_loss.entropy_gate_temperature=0.7',
                'distillation.distillation_loss.entropy_gate_epsilon=0.1',
                'distillation.distillation_loss.entropy_jsd_normalize=true',
            ])
        self.assertEqual(cfg.distillation.distillation_loss.entropy_gate_temperature, .7)
        self.assertTrue(cfg.distillation.distillation_loss.entropy_jsd_normalize)


if __name__ == '__main__':
    unittest.main()
