"""Distribution losses on teacher top-k probabilities.

Forward KL follows EOPD: normalize the teacher's top-k probabilities, then
score those tokens against the student's full-vocabulary probabilities. JSD
and both entropy gates normalize each distribution over teacher top-k only.
The entropy_kl reverse component instead uses the student-sampled token's
teacher and student logprobs with a PPO surrogate in losses.py. No residual-mass
bucket is used.
"""

import math

import torch


DISTRIBUTION_LOSS_MODES = ("forward_kl", "jsd", "entropy_kl", "entropy_jsd")


def validate_distribution_settings(mode, topk, beta, gate_temperature, gate_epsilon, use_policy_gradient):
    if mode not in DISTRIBUTION_LOSS_MODES:
        return
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        raise ValueError("Distribution losses require topk >= 1.")
    if not math.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("jsd_beta must be finite and strictly between 0 and 1.")
    if not math.isfinite(gate_temperature) or gate_temperature <= 0:
        raise ValueError("entropy_gate_temperature must be finite and > 0.")
    if not math.isfinite(gate_epsilon) or not 0 < gate_epsilon < 0.5:
        raise ValueError("entropy_gate_epsilon must be finite and in (0, 0.5).")
    if use_policy_gradient:
        raise ValueError(
            "Distribution losses require use_policy_gradient=False "
            "(entropy_kl handles its sampled reverse term internally)."
        )


def _distribution(log_probs):
    # Keep double precision for numerical gradient checks; otherwise use fp32.
    log_probs = log_probs.double() if log_probs.dtype == torch.float64 else log_probs.float()
    mass = log_probs.exp().sum(-1)
    probabilities = log_probs.softmax(-1)
    # A tiny floor handles underflow/roundoff, not the legacy -10 logprob clamp.
    probabilities = probabilities.clamp_min(1e-12)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return probabilities, probabilities.log(), mass


def distribution_loss(student_log_probs, teacher_log_probs, config):
    """Return differentiable token losses and detached, per-token diagnostics.

    alpha is the FKL coefficient for entropy_kl. beta is the TEACHER
    coefficient in M=beta*T+(1-beta)*S for JSD, so beta=1-alpha in entropy_jsd.
    Only the teacher and gate are detached; the student's contribution to M
    MUST remain differentiable. jsd_beta=0.5 gives ordinary symmetric JSD.
    """
    mode = config.loss_mode
    if mode not in DISTRIBUTION_LOSS_MODES:
        raise ValueError(f"Unknown distribution loss: {mode}")
    s, log_s, student_mass = _distribution(student_log_probs)
    t, log_t, teacher_mass = _distribution(teacher_log_probs.detach())
    hs = -(s * log_s).sum(-1).detach()
    ht = -(t * log_t).sum(-1).detach()
    # EOPD Eq. (10): normalize only the teacher on its top-k support. Keep the
    # student's original full-softmax probabilities; normalizing them on the
    # same support would remove the penalty for mass outside the teacher top-k.
    teacher_topk_log_probs = teacher_log_probs.detach().to(log_t.dtype)
    student_topk_log_probs = student_log_probs.to(log_s.dtype)
    teacher_topk_log_probs = teacher_topk_log_probs.log_softmax(dim=-1)
    forward = (
        teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    ).sum(-1)
    eps = config.entropy_gate_epsilon
    alpha = (eps + (1 - 2 * eps) * torch.sigmoid((ht - hs) / config.entropy_gate_temperature)).detach()
    beta = 1 - alpha if mode == "entropy_jsd" else torch.full_like(hs, config.jsd_beta)
    if mode == "forward_kl":
        loss = forward
        alpha = torch.ones_like(hs)
    elif mode == "entropy_kl":
        # The sampled-token reverse component is added by distillation_loss,
        # where the sampled token's teacher logprob and PPO logprobs are available.
        loss = alpha * forward
    else:
        b = beta.unsqueeze(-1)
        log_m = torch.logaddexp(log_t + b.log(), log_s + torch.log1p(-b))
        loss = beta * (t * (log_t - log_m)).sum(-1) + (1 - beta) * (s * (log_s - log_m)).sum(-1)
        # Optional scale normalization for the adaptive mode only. Disabled by
        # default: otherwise even beta=.5 is 4 times ordinary symmetric JSD.
        if mode == "entropy_jsd" and config.entropy_jsd_normalize:
            loss = loss / (beta * (1 - beta))
    result = {
        "distillation_losses": loss.clamp_min(0),
        "student_mass": student_mass.detach(),
        "teacher_mass": teacher_mass.detach(),
        "student_entropy": hs,
        "teacher_entropy": ht,
    }
    if mode in ("forward_kl", "entropy_kl"):
        result["forward_weight"] = alpha
    else:
        result["jsd_beta"] = beta.detach()
    return result


def compute_distribution_from_logits(student_logits, teacher_log_probs, teacher_ids, config):
    """Gather student probabilities on the identical teacher support, in fp32."""
    vocab_size = student_logits.shape[-1]
    if teacher_ids.shape[-1] > vocab_size:
        raise ValueError("topk cannot exceed the student vocabulary size.")
    if config.use_chunked_topk:
        chunk_size = config.chunked_topk_chunk_size
        if chunk_size < 1:
            raise ValueError("chunked_topk_chunk_size must be positive.")
        logits = student_logits.reshape(-1, vocab_size)
        ids = teacher_ids.reshape(-1, teacher_ids.shape[-1])
        parts = []
        for start in range(0, logits.shape[0], chunk_size):
            chunk = logits[start : start + chunk_size].float()
            parts.append(chunk.gather(-1, ids[start : start + chunk_size]) - chunk.logsumexp(-1, keepdim=True))
        selected = torch.cat(parts).reshape(teacher_ids.shape) if parts else student_logits.float()[..., :0]
        if not parts:
            selected = selected.reshape(teacher_ids.shape)
    else:
        logits = student_logits.float()
        selected = logits.gather(-1, teacher_ids) - logits.logsumexp(-1, keepdim=True)
    return distribution_loss(selected, teacher_log_probs, config)
