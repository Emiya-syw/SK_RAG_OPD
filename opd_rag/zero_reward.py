"""Zero task reward for pure distillation runs.

veRL's rollout pipeline still materializes ``rm_scores`` before the actor update.
The OPD configuration discards task rewards, so returning zero avoids dispatching
the project-specific ``sk_rag_opd`` source to an unrelated built-in scorer.
"""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 0.0
