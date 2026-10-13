from __future__ import annotations

from pathlib import Path
from typing import Any
from time import perf_counter

from opd_rag.thinking import chat_template_kwargs
from opd_rag.verl_data import (
    BASIC_INSTRUCTION, BASIC_RAG_INSTRUCTION, INSTRUCTION, PROMPT_MODES, instruction_for,
)
from rag_eval.data import read_jsonl, write_jsonl

def rag_question_of(row: dict[str, Any]) -> str:
    """Use the raw user question when a benchmark prompt wraps it."""
    prompt = str(row.get("prompt") or "").strip()
    dataset = str(row.get("dataset") or "").lower()
    if dataset == "mssbench":
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and str(metadata.get("question") or "").strip():
            return str(metadata["question"]).strip()
        for marker in ("The user query is:", "Your task is:"):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
    return prompt


def build_rag_messages(row: dict[str, Any], prompt_mode: str = "guided") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Current Image\n\n"},
    ]
    for _ in row.get("images", []):
        content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                f"\n\nCurrent Question\n\n{rag_question_of(row)}"
                "\n\nRetrieved Examples\n\n"
            ),
        }
    )
    for index, example in enumerate(row.get("retrieval", []), start=1):
        for _ in example.get("images", []):
            content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Example {index} | score={float(example['score']):.4f}\n"
                    f"Question\n{rag_question_of(example)}\n\n"
                    f"Response\n{example['answer']}\n\n"
                ),
            }
        )
    content.append({"type": "text", "text": instruction_for(prompt_mode)})
    return [{"role": "user", "content": content}]


def build_direct_messages(row: dict[str, Any], prompt_mode: str = "guided") -> list[dict[str, Any]]:
    instruction_for(prompt_mode)
    content: list[dict[str, Any]] = []
    for _ in row.get("images", []):
        content.append({"type": "image"})
    content.append({"type": "text", "text": str(row.get("prompt") or "").strip()})
    if prompt_mode == "basic":
        content.append({"type": "text", "text": BASIC_INSTRUCTION})
    return [{"role": "user", "content": content}]


def limit_retrieval(row: dict[str, Any], top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    retrieval = list(row.get("retrieval", []))
    if len(retrieval) < top_k:
        raise ValueError(
            f"Row {row.get('id')} only has {len(retrieval)} retrieved cases; "
            f"cannot select top-{top_k}"
        )
    if len(retrieval) <= top_k:
        return row
    limited = dict(row)
    limited["retrieval"] = retrieval[:top_k]
    config = dict(limited.get("retrieval_config") or {})
    config["top_k"] = top_k
    source_top_k = (
        row.get("retrieval_config", {}).get("top_k")
        if isinstance(row.get("retrieval_config"), dict)
        else None
    )
    if source_top_k is not None:
        config["source_top_k"] = source_top_k
    limited["retrieval_config"] = config
    return limited


def _open_all_images(row: dict[str, Any], max_pixels: int) -> list[Any]:
    from PIL import Image

    paths = list(row.get("images", []))
    for example in row.get("retrieval", []):
        paths.extend(example.get("images", []))
    images: list[Any] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Image for {row.get('id')}: {path}")
        image = Image.open(path).convert("RGB")
        if max_pixels and image.width * image.height > max_pixels:
            scale = (max_pixels / float(image.width * image.height)) ** 0.5
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        images.append(image)
    return images


def _open_current_images(row: dict[str, Any], max_pixels: int) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for value in row.get("images", []):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Image for {row.get('id')}: {path}")
        image = Image.open(path).convert("RGB")
        if max_pixels and image.width * image.height > max_pixels:
            scale = (max_pixels / float(image.width * image.height)) ** 0.5
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        images.append(image)
    return images


def _dtype(name: str):
    import torch

    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_generator(
    base_model: Path,
    adapter: Path | None,
    processor_path: Path | None,
    dtype: str,
    attn_implementation: str,
    merge_lora: bool = False,
):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_started = perf_counter()
    print(f"Loading generator: attention={attn_implementation}, merge_lora={merge_lora}", flush=True)
    selected_processor = processor_path or (
        adapter if adapter is not None and (adapter / "tokenizer_config.json").is_file() else base_model
    )
    processor = AutoProcessor.from_pretrained(selected_processor, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=_dtype(dtype),
        attn_implementation=attn_implementation,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge_lora:
            model = model.merge_and_unload(safe_merge=True)
    print(f"Generator loaded in {perf_counter() - load_started:.1f}s", flush=True)
    return model.eval(), processor


def visible_answer(text: str) -> str:
    return text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def _completed_ids(path: Path, prompt_mode: str = "guided") -> set[str]:
    rows = read_jsonl(path) if path.is_file() else []
    if any(row.get("prompt_mode", "guided") != prompt_mode for row in rows):
        raise ValueError(
            f"Prompt mode differs from existing answers in {path}; "
            "use a different --output-dir or --no-resume."
        )
    return {str(row["id"]) for row in rows}


def generate_dataset(
    model,
    processor,
    retrieval_path: Path,
    output_path: Path,
    base_model: Path,
    adapter: Path | None,
    max_new_tokens: int,
    max_image_pixels: int,
    resume: bool,
    progress_every: int,
    batch_size: int = 1,
    shard_index: int = 0,
    num_shards: int = 1,
    enable_thinking: bool = False,
    top_k: int | None = None,
    direct_generation: bool = False,
    prompt_mode: str = "guided",
) -> None:
    import torch

    instruction_for(prompt_mode)
    if batch_size < 1:
        raise ValueError("generation batch_size must be >= 1")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")
    rows = read_jsonl(retrieval_path)
    if top_k is not None and not direct_generation:
        rows = [limit_retrieval(row, top_k) for row in rows]
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    if num_shards > 1:
        rows = rows[shard_index::num_shards]
    done = _completed_ids(output_path, prompt_mode) if resume else set()
    pending = [row for row in rows if str(row["id"]) not in done]
    if output_path.exists() and not resume:
        output_path.unlink()
    timings = {"prepare": 0.0, "generate_decode": 0.0, "write": 0.0}
    dataset_started = perf_counter()
    for start in range(0, len(pending), batch_size):
        batch_started = perf_counter()
        batch_rows = pending[start : start + batch_size]
        open_images = _open_current_images if direct_generation else _open_all_images
        build_messages = build_direct_messages if direct_generation else build_rag_messages
        batch_images = [open_images(row, max_image_pixels) for row in batch_rows]
        try:
            texts = [
                processor.apply_chat_template(
                    build_messages(row, prompt_mode), tokenize=False, add_generation_prompt=True,
                    **chat_template_kwargs(enable_thinking),
                )
                for row in batch_rows
            ]
            inputs = processor(
                text=texts,
                images=batch_images if any(batch_images) else None,
                text_kwargs={"padding": True, "return_tensors": "pt"},
                images_kwargs={},
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            generation_started = perf_counter()
            timings["prepare"] += generation_started - batch_started
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            raw_predictions = [
                value.strip()
                for value in processor.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
            ]
            # batch_decode transfers generated IDs to CPU, completing GPU work.
            timings["generate_decode"] += perf_counter() - generation_started
        finally:
            for images in batch_images:
                for image in images:
                    image.close()
        results = []
        for row, raw in zip(batch_rows, raw_predictions):
            generated_tokens = max(0, int(generated.shape[1] - inputs["input_ids"].shape[1]))
            results.append(
                {
                    **row,
                    "mode": "direct" if direct_generation else "rag",
                    "prompt_mode": prompt_mode,
                    "base_model": str(base_model),
                    "adapter": str(adapter) if adapter is not None else None,
                    "generation_batch_size": batch_size,
                    "raw_prediction": raw,
                    "prediction": visible_answer(raw),
                    "generated_tokens": generated_tokens,
                    "generation_truncated": generated_tokens >= max_new_tokens,
                }
            )
        write_started = perf_counter()
        write_jsonl(output_path, results, mode="a")
        timings["write"] += perf_counter() - write_started
        completed = min(start + batch_size, len(pending))
        if completed % progress_every < batch_size or completed == len(pending):
            elapsed = perf_counter() - dataset_started
            print(
                f"  generated {completed}/{len(pending)} pending rows | "
                f"{completed / max(elapsed, 1e-9):.3f} rows/s | "
                f"prepare={timings['prepare']:.1f}s "
                f"generate_decode={timings['generate_decode']:.1f}s write={timings['write']:.1f}s",
                flush=True,
            )
    print(f"saved {len(rows)} total rows to {output_path}", flush=True)
