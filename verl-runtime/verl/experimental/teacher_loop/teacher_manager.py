# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    params = {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
        "detokenize": False,
    }
    # Stock vLLM does not put prompt_entropies on RequestOutput. Request all
    # vocabulary logprobs for entropy-gated losses, then reduce them to entropy
    # and the requested top-k inside the teacher server before the Ray return.
    if (
        distillation_loss_config.loss_mode in ("entropy_kl", "entropy_jsd")
        and getattr(teacher_model_config.inference, "name", None) == "vllm"
    ):
        params["prompt_logprobs"] = -1
        params["entropy_topk"] = num_logprobs
    return params


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_entropy: Optional[torch.Tensor],
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    padded_entropy = None
    if teacher_entropy is not None:
        padded_entropy = F.pad(teacher_entropy, (left_pad_size, right_pad_size), value=0.0).unsqueeze(0)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
        padded_entropy,
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        response_length: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        sampled_logprobs = None
        teacher_entropy = None
        loss_mode = getattr(self.distillation_loss_config, "loss_mode", None)
        if loss_mode in ("entropy_kl", "entropy_jsd"):
            entropies = teacher_output.extra_fields.get("prompt_entropies")
            if entropies is None:
                raise ValueError(f"{loss_mode} requires prompt_entropies: full-vocabulary teacher entropy per token.")
            teacher_entropy = torch.as_tensor(entropies, dtype=torch.float32)
            if teacher_entropy.shape != (teacher_ids.shape[0],) or not torch.isfinite(teacher_entropy).all():
                raise ValueError("Teacher prompt_entropies must be a finite value for every top-k row.")
        if loss_mode == "entropy_kl":
            sampled = teacher_output.extra_fields.get("prompt_sampled_logprobs")
            if sampled is None:
                raise ValueError("entropy_kl requires the teacher logprob of each sampled token.")
            sampled_logprobs = torch.tensor(sampled, dtype=teacher_logprobs.dtype)
            if sampled_logprobs.shape != (teacher_ids.shape[0],):
                raise ValueError("Teacher sampled-token logprobs are not aligned with top-k rows.")
        returned_length = teacher_ids.shape[0]
        expected_length = len(sequence_ids)
        if returned_length != teacher_logprobs.shape[0]:
            raise ValueError(
                "Teacher returned different token-id and logprob lengths: "
                f"teacher_ids={returned_length}, teacher_logprobs={teacher_logprobs.shape[0]}"
            )
        if returned_length != expected_length:
            # vLLM may re-expand multimodal placeholder runs differently from the
            # Hugging Face processor used by the agent loop. Distillation only
            # consumes response positions, so preserve the response-aligned suffix
            # and add/drop prompt-only rows on the left. For estimator losses
            # (prompt_logprobs=0), prompt_ids contains the scored token at each row.
            # For top-k, verify against the actual processed prompt token ids:
            # the scored token may not appear among the top-k columns.
            if not multi_modal_data or teacher_ids.ndim != 2:
                raise ValueError(
                    "Teacher sequence length mismatch cannot be safely aligned: "
                    f"expected={expected_length}, teacher_ids={returned_length}, "
                    f"teacher_logprobs={teacher_logprobs.shape[0]}, multimodal={bool(multi_modal_data)}"
                )
            if returned_length < response_length + 1:
                raise ValueError(
                    "Teacher output is too short to contain the response: "
                    f"returned={returned_length}, response={response_length}, expected={expected_length}"
                )
            scored_prompt_ids = teacher_output.extra_fields.get("scored_prompt_token_ids")
            if scored_prompt_ids is not None:
                if len(scored_prompt_ids) != returned_length:
                    raise ValueError(
                        "Teacher scored prompt ids and logprob rows have different lengths: "
                        f"prompt_ids={len(scored_prompt_ids)}, teacher_logprobs={returned_length}"
                    )
                returned_response_ids = scored_prompt_ids[-response_length:]
            elif teacher_ids.shape[1] == 1:
                returned_response_ids = teacher_ids[-(response_length + 1) : -1, 0].tolist()
            else:
                raise ValueError(
                    "Top-k teacher alignment requires scored_prompt_token_ids from the teacher server."
                )
            expected_response_ids = sequence_ids[-response_length:]
            if returned_response_ids != expected_response_ids:
                raise ValueError(
                    "Teacher response-token suffix differs from the student sequence; "
                    f"expected_length={expected_length}, returned_length={returned_length}, "
                    f"response_length={response_length}"
                )
            logger.warning(
                "Aligning multimodal teacher output on the response suffix: "
                "sequence=%d, teacher_ids=%d, teacher_logprobs=%d, response=%d",
                expected_length,
                returned_length,
                teacher_logprobs.shape[0],
                response_length,
            )
            if returned_length > expected_length:
                teacher_ids = teacher_ids[-expected_length:]
                teacher_logprobs = teacher_logprobs[-expected_length:]
                if sampled_logprobs is not None:
                    sampled_logprobs = sampled_logprobs[-expected_length:]
                if teacher_entropy is not None:
                    teacher_entropy = teacher_entropy[-expected_length:]
            else:
                left_padding = expected_length - returned_length
                teacher_ids = F.pad(teacher_ids, (0, 0, left_padding, 0), value=0)
                teacher_logprobs = F.pad(teacher_logprobs, (0, 0, left_padding, 0), value=0.0)
                if sampled_logprobs is not None:
                    sampled_logprobs = F.pad(sampled_logprobs, (left_padding, 0), value=0.0)
                if teacher_entropy is not None:
                    teacher_entropy = F.pad(teacher_entropy, (left_padding, 0), value=0.0)
        if sampled_logprobs is not None:
            # Keep teacher top-k in the first K columns. The last logprob is
            # the scored student token even when it falls outside top-k.
            teacher_ids = F.pad(teacher_ids, (0, 1), value=0)
            teacher_logprobs = torch.cat((teacher_logprobs, sampled_logprobs.unsqueeze(-1)), dim=-1)
        return teacher_ids, teacher_logprobs, teacher_entropy
