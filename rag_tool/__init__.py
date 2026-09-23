"""Standalone corpus indexing and retrieval utilities for SK-RAG."""

__version__ = "0.1.0"
from .config import Config
from .dataset import Dataset, Item
from .retriever import Retriever
from .sparse import BM25Index
from .multi_retriever import MultiRetriever

__all__ = ["Config", "Dataset", "Item", "Retriever", "BM25Index", "MultiRetriever"]
