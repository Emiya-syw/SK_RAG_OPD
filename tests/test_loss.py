import torch
from opd_rag.trainer import OPDTrainer


def test_reverse_kl_is_zero_for_identical_logits():
    logits = torch.randn(2, 4, 17)
    student = logits.clone().requires_grad_(True)
    labels = torch.tensor([[1, 2, -100, 3], [4, 5, 6, -100]])
    loss = OPDTrainer.token_level_reverse_kl_loss(student, logits, labels)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
    loss.backward()
    assert student.grad is not None


def test_reverse_kl_matches_manual_masked_computation():
    student = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
    labels = torch.tensor([[1, -100]])
    student_logp = torch.log_softmax(student[:, :1], dim=-1)
    teacher_logp = torch.log_softmax(teacher[:, :1], dim=-1)
    expected = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1).mean()
    actual = OPDTrainer.token_level_reverse_kl_loss(student, teacher, labels)
    assert torch.allclose(actual, expected)

def test_jsd_is_finite_and_has_gradient():
    student = torch.randn(2, 4, 17, requires_grad=True)
    teacher = torch.randn(2, 4, 17)
    labels = torch.tensor([[1, 2, -100, 3], [4, 5, 6, -100]])
    loss = OPDTrainer.generalized_jsd_loss(student, teacher, labels, top_k=5)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None

def test_sampled_objective_is_finite():
    student = torch.randn(1, 3, 11, requires_grad=True)
    teacher = torch.randn(1, 3, 11)
    token_ids = torch.tensor([[2, 4, 7]])
    labels = torch.tensor([[0, -100, 1]])
    loss = OPDTrainer.sampled_policy_gradient_loss(student, teacher, token_ids, labels)
    assert torch.isfinite(loss)
