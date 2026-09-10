from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import GenerationConfig, Trainer


class OPDTrainer(Trainer):
    """On-policy distillation trainer for Qwen-VL.

    The student generates from normal multimodal inputs. The teacher receives a
    privileged prompt with demonstration/reference answer. Loss is computed
    on the same generated tokens.
    """

    def __init__(
        self,
        *args,
        processor=None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        fixed_teacher: bool = True,
        advantage_clip: float = 5.0,
        loss_type: str = "jsd",
        beta: float = 0.5,
        top_k_loss: int = 0,
        jsd_token_clip: float = 0.0,
        teacher_model=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.teacher_model = teacher_model
        self.fixed_teacher = fixed_teacher
        self.advantage_clip = advantage_clip
        self.loss_type = loss_type
        self.beta = beta
        self.top_k_loss = top_k_loss if top_k_loss > 0 else None
        self.jsd_token_clip = jsd_token_clip if jsd_token_clip > 0 else None
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
        if self.loss_type not in {"reverse_kl", "jsd", "sampled_pg"}:
            raise ValueError("--loss_type must be 'reverse_kl', 'jsd', or 'sampled_pg'")

    @staticmethod
    def token_level_reverse_kl_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Standard full-vocabulary token-level OPD reverse-KL objective.

        Rollout prefixes are sampled from the current student. At every valid
        completion position, minimize KL(p_student || p_teacher).
        """
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_probs = student_log_probs.exp()
        token_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        mask = labels != -100
        if not mask.any():
            return student_logits.sum() * 0.0
        return token_loss[mask].mean() * (temperature ** 2)

    @staticmethod
    def generalized_jsd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        beta: float = 0.5,
        temperature: float = 1.0,
        top_k: int | None = None,
        token_clip: float | None = None,
    ) -> torch.Tensor:
        # 在 student 生成的 completion 位置上，对齐 student 和 teacher 的
        # next-token 分布。这是 OPSD 风格的分布匹配损失；top-k 用来降低
        # 多模态模型大词表带来的显存和计算开销。
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            # Generalized JSD：先构造 beta 加权的混合分布，再同时惩罚
            # teacher 到混合分布、student 到混合分布的偏离。
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta_tensor),
                        teacher_log_probs + torch.log(beta_tensor),
                    ]
                ),
                dim=0,
            )
            teacher_kl = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            student_kl = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            loss = beta_tensor * teacher_kl + (1 - beta_tensor) * student_kl

        loss = loss.sum(dim=-1)
        if token_clip is not None:
            loss = loss.clamp(max=token_clip)

        mask = labels != -100
        if not mask.any():
            return student_logits.sum() * 0.0
        return loss[mask].mean()

    @staticmethod
    def sampled_policy_gradient_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
        advantage_clip: float = 5.0,
    ) -> torch.Tensor:
        """Low-memory OPD objective using teacher-student log-prob advantage."""
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_token_logp = torch.gather(
            student_log_probs, -1, sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        teacher_token_logp = torch.gather(
            teacher_log_probs, -1, sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        advantage = (teacher_token_logp - student_token_logp).detach()
        if advantage_clip > 0:
            advantage = advantage.clamp(-advantage_clip, advantage_clip)
        mask = labels != -100
        if not mask.any():
            return student_token_logp.sum() * 0.0
        return -(advantage[mask] * student_token_logp[mask]).mean()

    def _split_prefixed(self, inputs: dict[str, Any], prefix: str) -> dict[str, Any]:
        output = {}
        for key, value in inputs.items():
            if key.startswith(prefix):
                unprefixed = key[len(prefix) :]
                if unprefixed in {"prompt_length", "prompt_lengths", "prompt_lengths_per_example"}:
                    continue
                output[unprefixed] = value
        return output

    def _teacher_context(self, model):
        # fixed-teacher LoRA 模式下，teacher 是关闭 adapter 的冻结 base model，
        # student 则是带 LoRA adapter 的可训练模型。
        if self.fixed_teacher and isinstance(self.accelerator.unwrap_model(model), PeftModel):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _independent_teacher(self, model):
        if self.teacher_model is None:
            return nullcontext()
        # Trainer does not prepare auxiliary models; keep the frozen teacher on
        # the same device as the student replica in each process.
        device = next(model.parameters()).device
        self.teacher_model.to(device)
        self.teacher_model.eval()
        return nullcontext()

    def _generate_student(self, model, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        # 从当前 student policy 采样生成 completion。后续 loss 只监督这些
        # on-policy 生成出来的 token。
        generation_model = self.accelerator.unwrap_model(model)
        model_was_training = model.training
        model.eval()
        student_inputs = self._split_prefixed(inputs, "student_")
        original_use_cache = getattr(generation_model.config, "use_cache", None)
        generation_model.config.use_cache = True
        with torch.no_grad():
            generated = generation_model.generate(
                **student_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            ).sequences
        if original_use_cache is not None:
            generation_model.config.use_cache = original_use_cache
        if model_was_training:
            model.train()
        attention_mask = torch.ones_like(generated)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            attention_mask[generated == pad_token_id] = 0
        return generated, attention_mask

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1) 基于 student prompt 采样一个 on-policy answer。
        generated_ids, generated_attention_mask = self._generate_student(model, inputs)
        student_prompt_len = inputs["student_prompt_length"]
        completion_ids = generated_ids[:, student_prompt_len:]

        # 2) 将 student 的输入从纯 prompt 替换为 prompt + 采样答案。
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask
        if "student_mm_token_type_ids" in inputs:
            completion_token_types = torch.zeros_like(completion_ids)
            inputs["student_mm_token_type_ids"] = torch.cat(
                [inputs["student_mm_token_type_ids"], completion_token_types],
                dim=1,
            )

        # 3) 把同一段采样答案拼到 privileged teacher prompt 后面，确保
        # student 和 teacher forward 评分的是完全相同的 completion token。
        teacher_full_ids = torch.cat([inputs["teacher_input_ids"], completion_ids], dim=1)
        teacher_completion_mask = torch.ones_like(completion_ids)
        teacher_attention_mask = torch.cat([inputs["teacher_attention_mask"], teacher_completion_mask], dim=1)
        pad_token_id = self.processor.tokenizer.pad_token_id
        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask
        if "teacher_mm_token_type_ids" in inputs:
            completion_token_types = torch.zeros_like(completion_ids)
            inputs["teacher_mm_token_type_ids"] = torch.cat(
                [inputs["teacher_mm_token_type_ids"], completion_token_types],
                dim=1,
            )

        # 忽略 prompt 和 padding token。loss 只作用在采样出来的 completion
        # token 上，从而保持 on-policy 训练目标。
        labels = generated_ids.clone()
        labels[:, :student_prompt_len] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels

        return super().training_step(model, inputs, num_items_in_batch)

    def _model_forward_kwargs(self, inputs: dict[str, Any], prefix: str) -> dict[str, Any]:
        kwargs = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
        }
        for key, value in inputs.items():
            if key.startswith(f"{prefix}_") and key not in (
                f"{prefix}_input_ids",
                f"{prefix}_attention_mask",
                f"{prefix}_prompt_length",
                f"{prefix}_prompt_lengths",
            ):
                kwargs[key[len(prefix) + 1 :]] = value
        return kwargs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # student forward 保留梯度；teacher forward 是 no-grad 的 privileged
        # view，只用来提供训练信号。
        student_outputs = model(**self._model_forward_kwargs(inputs, "student"))
        student_logits = student_outputs.logits[:, student_prompt_len - 1 : -1, :]

        with torch.no_grad(), self._teacher_context(model), self._independent_teacher(model):
            teacher_model = self.teacher_model or model
            teacher_outputs = teacher_model(**self._model_forward_kwargs(inputs, "teacher"))
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_len - 1 : -1, :]

        if self.loss_type == "reverse_kl":
            loss = self.token_level_reverse_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=shifted_labels,
                temperature=self.generation_config.temperature,
            )
        elif self.loss_type == "jsd":
            loss = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.generation_config.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
        else:
            loss = self.sampled_policy_gradient_loss(
                student_logits,
                teacher_logits,
                sampled_token_ids,
                shifted_labels,
                temperature=self.generation_config.temperature,
                advantage_clip=self.advantage_clip,
            )

        # Track policy uncertainty and rollout length alongside the OPD loss.
        # These diagnostics are detached and do not affect the training graph.
        with torch.no_grad():
            completion_mask = shifted_labels != -100
            valid_tokens = completion_mask.sum().clamp_min(1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
            teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1)
            student_entropy = student_entropy.masked_select(completion_mask).sum() / valid_tokens
            teacher_entropy = teacher_entropy.masked_select(completion_mask).sum() / valid_tokens
            completion_tokens = completion_mask.sum(dim=1).float().mean()
            reached_max_tokens = (completion_mask.sum(dim=1) >= sampled_token_ids.shape[1]).float().mean()
            self.log({
                "student_entropy": student_entropy.item(),
                "teacher_entropy": teacher_entropy.item(),
                "completion_tokens": completion_tokens.item(),
                "reached_max_tokens_ratio": reached_max_tokens.item(),
            })

        if return_outputs:
            return loss, {"loss": loss}
        return loss
