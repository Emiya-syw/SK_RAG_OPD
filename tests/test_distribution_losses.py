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


def loss_for_test(student, teacher, mode):
    return distribution_loss(student, teacher, config(mode))


def bucket_entropy(logprobs):
    p = logprobs.exp()
    r = (1 - p.sum(-1)).clamp_min(0)
    return -torch.special.xlogy(p, p).sum(-1) - torch.special.xlogy(r, r), r


class DistributionLossTests(unittest.TestCase):
    def test_entropy_kl_gate_uses_own_topk_and_tail_bucket(self):
        logits = torch.tensor([[[2., 1., 0., -1.]]], requires_grad=True)
        teacher_logits = torch.tensor([[[1., 0., -.5, -2.]]])
        teacher_logprobs = teacher_logits.log_softmax(-1)[..., :2]
        out = compute_distribution_from_logits(logits, teacher_logprobs,
                                               torch.tensor([[[0, 1]]]), config('entropy_kl'))
        student_topk = logits.log_softmax(-1).topk(2).values
        student_entropy, student_tail = bucket_entropy(student_topk)
        teacher_entropy, teacher_tail = bucket_entropy(teacher_logprobs)
        torch.testing.assert_close(out['student_entropy'], student_entropy)
        torch.testing.assert_close(out['teacher_entropy'], teacher_entropy)
        confidence = (1 - torch.maximum(student_tail, teacher_tail) / .2).clamp(0, 1)
        expected_alpha = .5 + confidence * (.05 + .9 * torch.sigmoid(teacher_entropy - student_entropy) - .5)
        torch.testing.assert_close(out['forward_weight'], expected_alpha)
        torch.testing.assert_close(out['entropy_gate_confidence'], confidence)
        self.assertFalse(out['forward_weight'].requires_grad)

    def test_entropy_jsd_gate_neutralizes_large_tail(self):
        logits = torch.tensor([[[2., 1., 0., -1.]]], requires_grad=True)
        teacher_logits = torch.tensor([[[1., 0., -.5, -2.]]])
        teacher_logprobs = teacher_logits.log_softmax(-1)[..., :2]
        out = compute_distribution_from_logits(logits, teacher_logprobs,
                                               torch.tensor([[[0, 1]]]), config('entropy_jsd', entropy_gate_tail_cutoff=.1))
        torch.testing.assert_close(out['jsd_beta'], torch.full_like(out['jsd_beta'], .5))
        torch.testing.assert_close(out['entropy_gate_confidence'], torch.zeros_like(out['jsd_beta']))
        self.assertFalse(out['jsd_beta'].requires_grad)

    def test_formulas_and_gradients(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            with self.subTest(mode=mode):
                x = torch.tensor([[1., -1., .5], [-2., 1., 0.]], dtype=torch.double, requires_grad=True)
                y = torch.tensor([[0., 0., 0.], [2., -1., -2.]], dtype=torch.double, requires_grad=True)
                s, t = x.softmax(-1), y.softmax(-1).detach()
                c = config(mode)
                out = loss_for_test(x.log_softmax(-1), y.log_softmax(-1), mode)
                hs, ht = -(s * s.log()).sum(-1), -(t * t.log()).sum(-1)
                alpha = (.05 + .9 * torch.sigmoid(ht - hs)).detach()
                beta = 1 - alpha if mode == 'entropy_jsd' else torch.full_like(alpha, .5)
                m = beta[:, None] * t + (1-beta[:, None]) * s
                expected = {'forward_kl': kl(t, s), 'entropy_kl': alpha*kl(t, s),
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
        ab = distribution_loss(a, b, c)['distillation_losses']
        ba = distribution_loss(b, a, c)['distillation_losses']
        torch.testing.assert_close(ab, ba)
        self.assertTrue((ab <= torch.log(torch.tensor(2.))).all())
        for mode in DISTRIBUTION_LOSS_MODES:
            loss = loss_for_test(a, a, mode)['distillation_losses']
            torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)

    def test_loss_support_stays_topk_while_entropy_uses_tail_bucket(self):
        s, t = torch.tensor([[.2, .3, .5]]), torch.tensor([[.6, .1, .3]])
        s_topk = s[:, :2] / s[:, :2].sum(-1, keepdim=True)
        t_topk = t[:, :2] / t[:, :2].sum(-1, keepdim=True)
        for mode in DISTRIBUTION_LOSS_MODES:
            c = config(mode)
            out = loss_for_test(s[:, :2].log(), t[:, :2].log(), mode)
            if mode in ('entropy_kl', 'entropy_jsd'):
                torch.testing.assert_close(out['student_entropy'], bucket_entropy(s[:, :2].log())[0])
                torch.testing.assert_close(out['teacher_entropy'], bucket_entropy(t[:, :2].log())[0])
            else:
                torch.testing.assert_close(out['student_entropy'], -(s_topk * s_topk.log()).sum(-1))
                torch.testing.assert_close(out['teacher_entropy'], -(t_topk * t_topk.log()).sum(-1))
            torch.testing.assert_close(out['student_mass'], torch.tensor([.5]))
            torch.testing.assert_close(out['teacher_mass'], torch.tensor([.7]))
            forward = (t_topk * (t_topk.log() - s[:, :2].log())).sum(-1)
            if mode == 'forward_kl':
                expected = forward
            elif mode == 'entropy_kl':
                alpha = out['forward_weight']
                expected = alpha * forward
            else:
                beta = out['jsd_beta'].unsqueeze(-1)
                mixture = beta * t_topk + (1 - beta) * s_topk
                expected = (beta.squeeze(-1) * kl(t_topk, mixture)
                            + (1 - beta.squeeze(-1)) * kl(s_topk, mixture))
            torch.testing.assert_close(out['distillation_losses'], expected)

        # Changing only the mass outside teacher top-k leaves conditional
        # JSD unchanged, while EOPD forward KL still sees it.
        less_topk_mass = torch.tensor([[.08, .12]])
        for mode in ('jsd',):
            original = loss_for_test(s[:, :2].log(), t[:, :2].log(), mode)
            changed = loss_for_test(less_topk_mass.log(), t[:, :2].log(), mode)
            torch.testing.assert_close(original['distillation_losses'], changed['distillation_losses'])
        original = distribution_loss(s[:, :2].log(), t[:, :2].log(), config('forward_kl'))
        changed = distribution_loss(less_topk_mass.log(), t[:, :2].log(), config('forward_kl'))
        self.assertGreater(changed['distillation_losses'].item(), original['distillation_losses'].item())
        original = distribution_loss(s[:, :2].log(), t[:, :2].log(), config('entropy_jsd', entropy_gate_tail_cutoff=1.0))
        changed = distribution_loss(less_topk_mass.log(), t[:, :2].log(), config('entropy_jsd', entropy_gate_tail_cutoff=1.0))
        self.assertNotAlmostEqual(changed['jsd_beta'].item(), original['jsd_beta'].item())

    def test_forward_kl_uses_normalized_teacher_topk_and_full_student_probability(self):
        logits = torch.tensor([[.3, -.4, 1.2]], dtype=torch.double, requires_grad=True)
        student = logits.log_softmax(-1)
        teacher = torch.tensor([[.6, .1]], dtype=torch.double).log()
        target = torch.tensor([[.6 / .7, .1 / .7, 0.]], dtype=torch.double)
        expected_forward = (target[:, :2] * (target[:, :2].log() - student[:, :2])).sum(-1)
        forward = distribution_loss(student[:, :2], teacher, config('forward_kl'))['distillation_losses']
        torch.testing.assert_close(forward, expected_forward)
        torch.testing.assert_close(torch.autograd.grad(forward.sum(), logits, retain_graph=True)[0],
                                   student.exp() - target)
        # The EOPD approximation must penalize student mass outside teacher top-k.
        self.assertGreater(forward.item(), 0)

        mixed = distribution_loss(student[:, :2], teacher, config('entropy_kl'), student.topk(2).values)
        alpha = mixed['forward_weight']
        torch.testing.assert_close(mixed['distillation_losses'], alpha * expected_forward)

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
            empty = compute_distribution_from_logits(x[:, :0], logs[:, :0], ids[:, :0],
                                                     config(mode, use_chunked_topk=True))
            self.assertEqual(empty['distillation_losses'].shape, (1, 0))

    def test_extreme_probabilities(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            x = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
            out = loss_for_test(x.log_softmax(-1), torch.tensor([[-1000., 0., -1000.]]), mode)
            loss = out['distillation_losses'].sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_normalization(self):
        s, t = torch.tensor([[.9, .1]]).log(), torch.tensor([[.5, .5]]).log()
        a = distribution_loss(s, t, config('entropy_jsd'))
        b = distribution_loss(s, t, config('entropy_jsd', entropy_jsd_normalize=True))
        beta = a['jsd_beta']
        torch.testing.assert_close(b['distillation_losses'], a['distillation_losses'] / (beta*(1-beta)))

    def test_configuration_and_registration(self):
        for mode in DISTRIBUTION_LOSS_MODES:
            self.assertTrue(config(mode).loss_settings.use_topk)
            for kwargs in ({'topk': 0}, {'jsd_beta': 1.}, {'jsd_beta': float('nan')},
                           {'entropy_gate_temperature': 0.}, {'entropy_gate_epsilon': .5},
                           {'entropy_gate_tail_cutoff': 0.}):
                with self.assertRaises(ValueError):
                    config(mode, **kwargs)
            with self.assertRaises(ValueError):
                DistillationLossConfig(loss_mode=mode, use_policy_gradient=True)
        self.assertEqual(DistillationLossConfig().loss_mode, 'k3')
        self.assertEqual(DistillationLossConfig().topk, 16)

    def test_fsdp_nested_dispatch(self):
        from verl.trainer.distillation.losses import compute_topk_loss
        torch.manual_seed(18)
        x = torch.randn(1, 5, 8, requires_grad=True)
        logs, ids = torch.randn_like(x).log_softmax(-1).topk(3)
        data = {'teacher_logprobs': torch.nested.nested_tensor([logs[0, :2], logs[0, 2:]], layout=torch.jagged),
                'teacher_ids': torch.nested.nested_tensor([ids[0, :2], ids[0, 2:]], layout=torch.jagged)}
        for mode in DISTRIBUTION_LOSS_MODES:
            c = config(mode)
            mode_data = data
            if mode == 'entropy_kl':
                sampled_logs = torch.randn(5).log_softmax(0)
                expanded_logs = torch.cat((logs[0], sampled_logs[:, None]), dim=-1)
                expanded_ids = torch.cat((ids[0], torch.zeros(5, 1, dtype=ids.dtype)), dim=-1)
                mode_data = {
                    'teacher_logprobs': torch.nested.nested_tensor(
                        [expanded_logs[:2], expanded_logs[2:]], layout=torch.jagged),
                    'teacher_ids': torch.nested.nested_tensor(
                        [expanded_ids[:2], expanded_ids[2:]], layout=torch.jagged),
                }
            out = compute_topk_loss(SimpleNamespace(strategy='fsdp'), SimpleNamespace(distillation_loss=c), mode_data, x, 'thd')
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
        for mode in ('forward_kl', 'jsd', 'entropy_jsd'):
            c = config(mode)
            raw = compute_distribution_from_logits(x, logs, ids, c)
            output = {k: v.squeeze(0) for k, v in raw.items()}
            loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=c), output, data)
            expected = raw['distillation_losses'][0, [1, 3]].mean()
            torch.testing.assert_close(loss, expected)
            self.assertAlmostEqual(metrics['distillation/teacher_entropy'], raw['teacher_entropy'][0, [1, 3]].mean().item())
            gradient = torch.autograd.grad(loss, x)[0]
            torch.testing.assert_close(gradient[0, [0, 2, 4]], torch.zeros_like(gradient[0, [0, 2, 4]]))

    def test_entropy_kl_reverse_uses_sampled_logprob_advantage(self):
        from verl.trainer.distillation.losses import distillation_loss
        from tensordict import TensorDict

        sampled_teacher = torch.tensor([0., -.5, -1., -5.5, 0.])
        teacher_rows = torch.zeros(5, 17)
        teacher_rows[:, -1] = sampled_teacher
        old = torch.tensor([[-.7, -.8, -5.]])
        current = torch.tensor([0., -.1, -.9, -2., 0.], requires_grad=True)
        forward = torch.tensor([0., .4, .3, .2, 0.], requires_grad=True)
        weight = torch.full((5,), .25)
        data = TensorDict({
            'prompts': torch.ones(1, 2, dtype=torch.long),
            'responses': torch.ones(1, 3, dtype=torch.long),
            'attention_mask': torch.ones(1, 5, dtype=torch.long),
            'response_mask': torch.tensor([[1., 0., 1.]]),
            'old_log_probs': old,
            'teacher_logprobs': torch.nested.nested_tensor([teacher_rows], layout=torch.jagged),
            'dp_size': 1, 'batch_num_tokens': 2, 'global_batch_size': 1,
        }, batch_size=[])
        output = {
            'distillation_losses': forward,
            'student_mass': torch.ones(5), 'teacher_mass': torch.ones(5),
            'student_entropy': torch.zeros(5), 'teacher_entropy': torch.zeros(5),
            'forward_weight': weight,
            'log_probs': current,
        }
        actor = SimpleNamespace(loss_agg_mode='token-mean', global_batch_info={}, loss_scale_factor=None)
        loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=config('entropy_kl')), output, data)
        sampled_reverse_kl = old - sampled_teacher[1:4].unsqueeze(0)
        advantage = -.75 * sampled_reverse_kl
        ratio = (current[1:4].unsqueeze(0) - old).exp()
        reverse = torch.maximum(-ratio * advantage, -ratio.clamp(.8, 1.2) * advantage)
        torch.testing.assert_close(loss, forward[[1, 3]].mean() + reverse[0, [0, 2]].mean())
        self.assertAlmostEqual(metrics['distillation/sampled_reverse_kl'],
                               (.75 * sampled_reverse_kl[0, [0, 2]]).mean().item())
        self.assertAlmostEqual(metrics['distillation/pg_clipfrac'], .5)
        self.assertGreater(reverse[0, 2].item(), 3 * abs(advantage[0, 2]).item())
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNotNone(forward.grad)

    def test_hydra_configuration(self):
        from hydra import compose, initialize_config_module
        with initialize_config_module(config_module='verl.trainer.config', version_base=None):
            cfg = compose(config_name='ppo_trainer', overrides=[
                'distillation.distillation_loss.loss_mode=entropy_jsd',
                'distillation.distillation_loss.use_policy_gradient=false',
                'distillation.distillation_loss.jsd_beta=0.5',
                'distillation.distillation_loss.entropy_gate_temperature=0.7',
                'distillation.distillation_loss.entropy_gate_epsilon=0.1',
                'distillation.distillation_loss.entropy_gate_tail_cutoff=0.2',
                'distillation.distillation_loss.entropy_jsd_normalize=true',
            ])
        self.assertEqual(cfg.distillation.distillation_loss.entropy_gate_temperature, .7)
        self.assertEqual(cfg.distillation.distillation_loss.entropy_gate_tail_cutoff, .2)
        self.assertTrue(cfg.distillation.distillation_loss.entropy_jsd_normalize)


if __name__ == '__main__':
    unittest.main()
