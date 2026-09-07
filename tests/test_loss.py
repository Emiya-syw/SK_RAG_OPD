import torch
from opd_rag.trainer import OPDTrainer

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
