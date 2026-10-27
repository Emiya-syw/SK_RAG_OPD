"""Distribution losses on teacher top-k probabilities.

Forward KL follows EOPD: normalize the teacher's top-k probabilities, then
score those tokens against the student's full-vocabulary probabilities. JSD
and entropy_jsd retain full-softmax mass on teacher top-k plus a residual
bucket for all other tokens. Both entropy modes approximate full-vocabulary
entropy from each model's own top-k probabilities and one bucket for the
remaining probability mass. Their shared gate signal uses teacher entropy,
student coverage, and teacher top-k mass.
The entropy_kl reverse component instead uses the student-sampled token's
teacher and student logprobs with a PPO surrogate in losses.py. Forward KL
does not use a residual-mass bucket.
"""

import math

import torch


K3_JSD_LOSS_MODES = ("k3_jsd", "k3_entropy_jsd")
ENTROPY_GATE_LOSS_MODES = ("entropy_kl", "entropy_jsd", "k3_entropy_jsd")
# These top-k objectives also need the teacher's probability of the rollout token.
SAMPLED_TOPK_LOSS_MODES = ("entropy_kl", *K3_JSD_LOSS_MODES)
DISTRIBUTION_LOSS_MODES = ("forward_kl", "jsd", "entropy_kl", "entropy_jsd", *K3_JSD_LOSS_MODES)


def validate_distribution_settings(
    mode, topk, beta, gate_temperature, gate_epsilon, use_policy_gradient, tail_cutoff=0.2, entropy_topk=64,
    teacher_entropy_threshold=0.8, coverage_deficit_scale=0.3, teacher_mass_floor=0.95,
    k3_jsd_lambda=1.0,
):
    if mode not in DISTRIBUTION_LOSS_MODES:
        return
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        raise ValueError("Distribution losses require topk >= 1.")
    if mode in ENTROPY_GATE_LOSS_MODES and (
        not isinstance(entropy_topk, int) or isinstance(entropy_topk, bool) or entropy_topk < 1
    ):
        raise ValueError("Entropy-gated losses require entropy_topk >= 1.")
    if not math.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("jsd_beta must be finite and strictly between 0 and 1.")
    if not math.isfinite(gate_temperature) or gate_temperature <= 0:
        raise ValueError("entropy_gate_temperature must be finite and > 0.")
    if not math.isfinite(gate_epsilon) or not 0 < gate_epsilon < 0.5:
        raise ValueError("entropy_gate_epsilon must be finite and in (0, 0.5).")
    if not math.isfinite(tail_cutoff) or not 0 < tail_cutoff <= 1:
        raise ValueError("entropy_gate_tail_cutoff must be finite and in (0, 1].")
    if not math.isfinite(teacher_entropy_threshold) or teacher_entropy_threshold < 0:
        raise ValueError("entropy_kl_teacher_entropy_threshold must be finite and >= 0.")
    if not math.isfinite(coverage_deficit_scale) or not 0 < coverage_deficit_scale <= 1:
        raise ValueError("entropy_kl_coverage_deficit_scale must be finite and in (0, 1].")
    if not math.isfinite(teacher_mass_floor) or not 0 < teacher_mass_floor < 1:
        raise ValueError("entropy_kl_teacher_mass_floor must be finite and in (0, 1).")
    if mode in K3_JSD_LOSS_MODES and (not math.isfinite(k3_jsd_lambda) or k3_jsd_lambda < 0):
        raise ValueError("k3_jsd_lambda must be finite and >= 0.")
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


def _tail_bucket_entropy(log_probs):
    """Coarse-grained entropy: top-k tokens plus one residual-mass bucket."""
    log_probs = log_probs.double() if log_probs.dtype == torch.float64 else log_probs.float()
    probabilities = log_probs.exp()
    tail_mass = (1 - probabilities.sum(-1)).clamp(0, 1)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1) - torch.special.xlogy(
        tail_mass, tail_mass
    )
    return entropy.detach(), tail_mass.detach()


def _distribution_with_tail(log_probs):
    """Full-softmax probabilities on teacher top-k plus one residual bucket."""
    log_probs = log_probs.double() if log_probs.dtype == torch.float64 else log_probs.float()
    probabilities = log_probs.exp()
    tail = (1 - probabilities.sum(-1, keepdim=True)).clamp_min(0)
    probabilities = torch.cat((probabilities, tail), dim=-1)
    return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)


def distribution_loss(student_log_probs, teacher_log_probs, config, student_gate_log_probs=None,
                      teacher_gate_log_probs=None):
    """Return differentiable token losses and detached, per-token diagnostics.

    alpha is the FKL coefficient for entropy_kl and the student mixture
    coefficient for entropy_jsd. beta is the TEACHER coefficient in
    M=beta*T+(1-beta)*S. Low teacher top-k mass sends entropy_jsd beta to 0.5.
    Only the teacher and gate are detached; the student's contribution to M
    MUST remain differentiable. jsd_beta=0.5 gives ordinary symmetric JSD.
    """
    mode = config.loss_mode
    if mode not in DISTRIBUTION_LOSS_MODES:
        raise ValueError(f"Unknown distribution loss: {mode}")
    # Only compute the JSD component here. The sampled K3 component and lambda
    # are applied in distillation_loss, where sampled-token logprobs are available.
    mode = {"k3_jsd": "jsd", "k3_entropy_jsd": "entropy_jsd"}.get(mode, mode)
    s, log_s, student_mass = _distribution(student_log_probs)
    t, log_t, teacher_mass = _distribution(teacher_log_probs.detach())
    hs = -(s * log_s).sum(-1).detach()
    ht = -(t * log_t).sum(-1).detach()
    tail_confidence = None
    if mode in ("entropy_kl", "entropy_jsd"):
        # Teacher and student use their own top-k. In particular, these are
        # full-softmax logprobs, not renormalized top-k probabilities.
        hs, student_tail = _tail_bucket_entropy(
            student_gate_log_probs if student_gate_log_probs is not None else student_log_probs
        )
        ht, teacher_tail = _tail_bucket_entropy(
            teacher_gate_log_probs.detach() if teacher_gate_log_probs is not None else teacher_log_probs.detach()
        )
        if hs.shape != ht.shape:
            raise ValueError("Student and teacher entropy rows must have matching token dimensions.")
    # EOPD Eq. (10): normalize only the teacher on its top-k support. Keep the
    # student's original full-softmax probabilities; normalizing them on the
    # same support would remove the penalty for mass outside the teacher top-k.
    teacher_topk_log_probs = teacher_log_probs.detach().to(log_t.dtype)
    student_topk_log_probs = student_log_probs.to(log_s.dtype)
    teacher_topk_log_probs = teacher_topk_log_probs.log_softmax(dim=-1)
    forward = (
        teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    ).sum(-1)
    if mode in ("entropy_kl", "entropy_jsd"):
        # Compare raw full-softmax probabilities on the identical teacher top-k
        # IDs. A student overallocating one teacher token cannot compensate for
        # missing another teacher mode.
        teacher_prob = teacher_log_probs.detach().to(student_topk_log_probs.dtype).exp()
        student_prob = student_topk_log_probs.detach().exp()
        coverage = torch.minimum(teacher_prob, student_prob).sum(-1) / teacher_mass.clamp_min(1e-12)
        coverage = coverage.clamp(0, 1).detach()
        deficit = ((1 - coverage) / config.entropy_kl_coverage_deficit_scale).clamp(0, 1)
        entropy_signal = torch.sigmoid(
            (ht - config.entropy_kl_teacher_entropy_threshold) / config.entropy_gate_temperature
        )
        teacher_mass_confidence = (
            (teacher_mass - config.entropy_kl_teacher_mass_floor) /
            (1 - config.entropy_kl_teacher_mass_floor)
        ).clamp(0, 1).detach()
        raw_alpha = entropy_signal * deficit
        if mode == "entropy_jsd":
            # Unreliable top-k mass removes the adaptive tilt, not the JSD
            # supervision. Entropy KL still gates its additional FKL term.
            alpha = (0.5 + teacher_mass_confidence * (raw_alpha - 0.5)).detach()
        else:
            alpha = (teacher_mass_confidence * raw_alpha).detach()
        tail_confidence = teacher_mass_confidence
    else:
        alpha = torch.zeros_like(hs)
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
        t_jsd = _distribution_with_tail(teacher_log_probs.detach())
        s_jsd = _distribution_with_tail(student_log_probs)
        log_m = (b * t_jsd + (1 - b) * s_jsd).clamp_min(1e-12).log()
        log_t_jsd = t_jsd.clamp_min(1e-12).log()
        log_s_jsd = s_jsd.clamp_min(1e-12).log()
        loss = beta * (t_jsd * (log_t_jsd - log_m)).sum(-1) + (
            1 - beta
        ) * (s_jsd * (log_s_jsd - log_m)).sum(-1)
        # Optional scale normalization for the adaptive mode only. Disabled by
        # default: otherwise even beta=.5 is 4 times ordinary symmetric JSD.
        if mode == "entropy_jsd" and config.entropy_jsd_normalize:
            # alpha can be exactly zero when student coverage is complete.
            # In that case beta=1 and the JSD term is exactly zero.
            loss = loss / (beta * (1 - beta)).clamp_min(1e-12)
    result = {
        "distillation_losses": loss.clamp_min(0),
        "student_mass": student_mass.detach(),
        "teacher_mass": teacher_mass.detach(),
        "student_entropy": hs,
        "teacher_entropy": ht,
    }
    if tail_confidence is not None:
        result["student_tail_mass"] = student_tail
        result["teacher_tail_mass"] = teacher_tail
        result["entropy_gate_confidence"] = tail_confidence
        result["teacher_topk_coverage"] = coverage
    if mode in ("forward_kl", "entropy_kl"):
        result["forward_weight"] = alpha
    else:
        result["jsd_beta"] = beta.detach()
    return result


def compute_distribution_from_logits(student_logits, teacher_log_probs, teacher_ids, config,
                                     teacher_gate_log_probs=None):
    """Gather student probabilities on the identical teacher support, in fp32."""
    vocab_size = student_logits.shape[-1]
    if teacher_ids.shape[-1] > vocab_size:
        raise ValueError("topk cannot exceed the student vocabulary size.")
    need_gate_entropy = config.loss_mode in ENTROPY_GATE_LOSS_MODES
    gate_topk = teacher_gate_log_probs.shape[-1] if teacher_gate_log_probs is not None else teacher_ids.shape[-1]
    if need_gate_entropy and gate_topk > vocab_size:
        raise ValueError("entropy_topk cannot exceed the student vocabulary size.")
    gate_logprob_parts = []
    if config.use_chunked_topk:
        chunk_size = config.chunked_topk_chunk_size
        if chunk_size < 1:
            raise ValueError("chunked_topk_chunk_size must be positive.")
        logits = student_logits.reshape(-1, vocab_size)
        ids = teacher_ids.reshape(-1, teacher_ids.shape[-1])
        parts = []
        for start in range(0, logits.shape[0], chunk_size):
            chunk = logits[start : start + chunk_size].float()
            log_z = chunk.logsumexp(-1, keepdim=True)
            parts.append(chunk.gather(-1, ids[start : start + chunk_size]) - log_z)
            if need_gate_entropy:
                with torch.no_grad():
                    gate_logprob_parts.append(chunk.topk(gate_topk, dim=-1).values - log_z)
        selected = torch.cat(parts).reshape(teacher_ids.shape) if parts else student_logits.float()[..., :0]
        if not parts:
            selected = selected.reshape(teacher_ids.shape)
    else:
        logits = student_logits.float()
        log_z = logits.logsumexp(-1, keepdim=True)
        selected = logits.gather(-1, teacher_ids) - log_z
        if need_gate_entropy:
            with torch.no_grad():
                gate_logprob_parts.append(logits.topk(gate_topk, dim=-1).values - log_z)
    student_gate_log_probs = None
    if need_gate_entropy:
        student_gate_log_probs = (
            torch.cat(gate_logprob_parts).reshape(*teacher_ids.shape[:-1], gate_topk)
            if gate_logprob_parts else student_logits.new_empty((*teacher_ids.shape[:-1], gate_topk), dtype=torch.float32)
        )
    return distribution_loss(selected, teacher_log_probs, config, student_gate_log_probs,
                             teacher_gate_log_probs)
