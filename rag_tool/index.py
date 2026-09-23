from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class DenseIndex:
    """Small self-contained cosine-similarity index.

    FAISS is used when available; the NumPy path is an exact fallback and keeps
    the tool runnable on servers where installing native FAISS is inconvenient.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except ImportError:
            pass

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._faiss is not None:
            return self._faiss.search(np.ascontiguousarray(queries, dtype="float32"), top_k)
        scores = np.asarray(queries, dtype="float32") @ self.vectors.T
        indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        values = np.take_along_axis(scores, indices, axis=1)
        order = np.argsort(-values, axis=1)
        return np.take_along_axis(values, order, axis=1), np.take_along_axis(indices, order, axis=1)

    def save(self, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.vectors)
        if self._faiss is not None:
            import faiss
            faiss.write_index(self._faiss, str(directory / "index.faiss"))
            return "faiss"
        return "numpy"

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
        instance = cls.__new__(cls)
        instance.vectors = vectors
        instance._faiss = None
        faiss_path = directory / "index.faiss"
        try:
            import faiss
            if faiss_path.is_file():
                instance._faiss = faiss.read_index(str(faiss_path))
        except ImportError:
            pass
        return instance
