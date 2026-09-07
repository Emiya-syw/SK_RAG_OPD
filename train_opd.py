"""Train a Qwen-VL model with on-policy distillation.

Example:
  python train_opd.py --train_file data/train.jsonl --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --output_dir outputs/opd --bf16 true --use_lora true
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, HfArgumentParser, TrainingArguments

from opd_rag.collator import OPDDataCollator
from opd_rag.trainer import OPDTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    teacher_model_name_or_path: str | None = None
    lora_init_path: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class DataArguments:
    train_file: str | None = None
    dataset_name: str | None = None
    dataset_split: str = "train"
    max_prompt_length: int = 4096
    max_image_pixels: int = 262144
    include_reference_answer: bool = True
    image_root: str | None = None


@dataclass
class OPDArguments:
    max_new_tokens: int = 128
    generation_temperature: float = 1.0
    generation_top_p: float = 0.95
    generation_top_k: int = 20
    fixed_teacher: bool = True
    loss_type: str = "jsd"
    beta: float = 0.5
    top_k_loss: int = 0
    jsd_token_clip: float = 0.0
    advantage_clip: float = 5.0


def _dtype(name: str):
    values = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
              "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported --torch_dtype: {name}") from exc


def load_train_dataset(args: DataArguments):
    if args.dataset_name:
        return load_dataset(args.dataset_name, split=args.dataset_split)
    if not args.train_file:
        raise ValueError("Pass --train_file or --dataset_name")
    path = Path(args.train_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_dataset("json", data_files=str(path), split="train")


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, OPDArguments, TrainingArguments))
    model_args, data_args, opd_args, training_args = parser.parse_args_into_dataclasses()
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )
    teacher_path = model_args.teacher_model_name_or_path or model_args.model_name_or_path
    if teacher_path == model_args.model_name_or_path:
        teacher_processor = processor
        teacher_model = None
    else:
        teacher_processor = AutoProcessor.from_pretrained(
            teacher_path, trust_remote_code=model_args.trust_remote_code
        )
        if teacher_processor.tokenizer.pad_token is None:
            teacher_processor.tokenizer.pad_token = teacher_processor.tokenizer.eos_token
        teacher_processor.tokenizer.padding_side = "left"
        teacher_model = AutoModelForImageTextToText.from_pretrained(
            teacher_path,
            torch_dtype=_dtype(model_args.torch_dtype),
            attn_implementation=model_args.attn_implementation,
            trust_remote_code=model_args.trust_remote_code,
        )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        student_vocab = processor.tokenizer.get_vocab()
        teacher_vocab = teacher_processor.tokenizer.get_vocab()
        if student_vocab != teacher_vocab:
            raise ValueError(
                "Independent teacher and student must use identical tokenizers for OPD token/logit alignment."
            )
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if model_args.use_lora:
        if model_args.lora_init_path:
            model = PeftModel.from_pretrained(model, model_args.lora_init_path, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=[x.strip() for x in model_args.lora_target_modules.split(",") if x.strip()],
                bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    trainer = OPDTrainer(
        model=model, args=training_args, train_dataset=load_train_dataset(data_args),
        data_collator=OPDDataCollator(processor, data_args.max_prompt_length,
                                       data_args.include_reference_answer, data_args.image_root,
                                       data_args.max_image_pixels, teacher_processor=teacher_processor), processor=processor,
        teacher_model=teacher_model,
        max_new_tokens=opd_args.max_new_tokens, temperature=opd_args.generation_temperature,
        top_p=opd_args.generation_top_p, top_k=opd_args.generation_top_k,
        fixed_teacher=opd_args.fixed_teacher, loss_type=opd_args.loss_type,
        beta=opd_args.beta, top_k_loss=opd_args.top_k_loss,
        jsd_token_clip=opd_args.jsd_token_clip, advantage_clip=opd_args.advantage_clip)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
