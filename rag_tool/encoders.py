from __future__ import annotations

import sys
from pathlib import Path
from typing import Any



class Encoder:
    def __init__(self, backend: str, model_path: str, max_length: int, max_pixels: int, instruction: str | None = None, code_path: str | None = None):
        self.backend = backend
        self.model_path = model_path
        self.instruction = instruction
        self.max_length = max_length
        self.max_pixels = max_pixels
        if backend == "qwen3vl":
            repo = Path(code_path or model_path).resolve()
            if (repo / "src").is_dir():
                sys.path.insert(0, str(repo))
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
            import torch
            self.model = Qwen3VLEmbedder(model_name_or_path=model_path, max_length=max_length, max_pixels=max_pixels, default_instruction=instruction or "Represent the user's multimodal input.", torch_dtype=torch.bfloat16)
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            import torch
            self.model = SentenceTransformer(model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
        elif backend == "transformers":
            from transformers import AutoModel, AutoTokenizer
            import torch
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def encode(self, rows: list[dict[str, Any]], batch_size: int) -> np.ndarray:
        import numpy as np
        if self.backend == "qwen3vl":
            from PIL import Image
            import torch
            output = []
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                opened = []
                inputs = []
                try:
                    for row in batch:
                        images = []
                        for path in row.get("images", []):
                            image = Image.open(path).convert("RGB")
                            opened.append(image)
                            images.append(image)
                        inputs.append({"text": row.get("contents", ""), "image": images or None, "instruction": "Retrieve images or text relevant to the user's query."})
                    with torch.inference_mode():
                        output.append(self.model.process(inputs, normalize=True).detach().cpu().float().numpy())
                finally:
                    for image in opened:
                        image.close()
            return np.concatenate(output).astype("float32")
        texts = [row.get("contents", "") for row in rows]
        if self.backend == "sentence-transformers":
            return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True), dtype="float32")
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(self.torch.nn.functional.normalize(vector, dim=-1).cpu().numpy())
        return np.concatenate(vectors).astype("float32")
