"""Configuration handling for the standalone RAG workflow."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_name": "default",
    "split": ["test"],
    "seed": 2024,
    "device": None,
    "gpu_id": None,
    "retrieval_method": "dense",
    "retrieval_topk": 5,
    "retrieval_batch_size": 32,
    "retrieval_cache_path": None,
    "use_retrieval_cache": False,
    "save_retrieval_cache": False,
    "use_reranker": False,
    "rerank_topk": 5,
    "generator_model_path": None,
    "generation_params": {},
    "metrics": ["em", "f1", "retrieval_recall"],
}


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON or install pyyaml") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class Config(dict):
    """Dictionary-like config with deterministic defaults and device setup."""

    def __init__(self, config_file: str | Path | None = None, config_dict: dict[str, Any] | None = None):
        values = copy.deepcopy(DEFAULT_CONFIG)
        if config_file:
            values = _merge(values, _load(Path(config_file)))
        if config_dict:
            values = _merge(values, config_dict)
        if isinstance(values.get("split"), str):
            values["split"] = [values["split"]]
        if values.get("gpu_id") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(values["gpu_id"])
        if values.get("device") is None:
            try:
                import torch
                values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                values["gpu_num"] = torch.cuda.device_count()
            except ImportError:
                values["device"], values["gpu_num"] = "cpu", 0
        self._set_seed(int(values.get("seed", 2024)))
        super().__init__(values)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            path.write_text(yaml.safe_dump(dict(self), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self, ensure_ascii=False, indent=2), encoding="utf-8")
