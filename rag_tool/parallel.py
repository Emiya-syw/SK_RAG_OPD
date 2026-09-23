from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def map_batches(fn: Callable[[list[T]], list[U]], values: Iterable[T], batch_size: int, workers: int = 1) -> list[U]:
    items = list(values)
    batches = [items[start:start + batch_size] for start in range(0, len(items), batch_size)]
    if workers <= 1:
        return sum((fn(batch) for batch in batches), [])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = pool.map(fn, batches)
        return sum(outputs, [])
