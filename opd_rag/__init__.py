"""On-policy distillation (OPD) for retrieval-augmented vision-language models."""

from .trainer import OPDTrainer
from .collator import OPDDataCollator

__all__ = ["OPDTrainer", "OPDDataCollator"]
