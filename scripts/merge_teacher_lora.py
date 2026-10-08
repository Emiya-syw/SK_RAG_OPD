#!/usr/bin/env python3
r"""Merge a Qwen3-VL teacher's PEFT LoRA into standalone BF16 weights.

Run in the training Python environment. CPU merging requires enough host RAM
for the base weights plus temporary merge buffers. Use the exact base model
used to train the adapter (including Thinking versus Instruct).

Example:
    python3 scripts/merge_teacher_lora.py \
        --base models/Qwen3-VL-8B-Thinking \
        --adapter /path/to/teacher_lora \
        --output models/Qwen3-VL-8B-Teacher-Merged

Then set TEACHER_MODEL_PATH to the output directory when starting training.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Base model directory or Hugging Face ID")
    parser.add_argument("--adapter", required=True, type=Path, help="Local PEFT adapter directory")
    parser.add_argument("--output", required=True, type=Path, help="New merged model directory")
    parser.add_argument(
        "--processor", help="Processor directory or ID; defaults to --base. Use the fine-tuning processor if modified."
    )
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"Missing adapter_config.json in {args.adapter}")
    if not any((args.adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        parser.error(f"Missing adapter_model.safetensors or adapter_model.bin in {args.adapter}")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new directory: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Loading processor: {args.processor or args.base}", flush=True)
    processor = AutoProcessor.from_pretrained(args.processor or args.base)
    print(f"Loading base model on CPU in BF16: {args.base}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("Merging LoRA weights with safe_merge=True", flush=True)
    model = model.merge_and_unload(safe_merge=True)

    # Reserve a fresh destination without overwriting any existing model.
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Saving merged model: {args.output}", flush=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output)
    print(f"Done. Set TEACHER_MODEL_PATH={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
