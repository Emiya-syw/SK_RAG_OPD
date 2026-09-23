"""Dataset and item containers matching the RAG pipeline data contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import prepare_rows, read_rows, write_jsonl


@dataclass
class Item:
    data: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.output:
            return self.output[name]
        if name in self.data:
            return self.data[name]
        raise AttributeError(name)

    def update_output(self, key: str, value: Any) -> None:
        self.output[key] = value

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.data)
        result.update(self.output)
        return result


class Dataset:
    def __init__(self, rows: Iterable[dict[str, Any]] | None = None, path: str | Path | None = None, config: dict[str, Any] | None = None):
        if path is not None:
            rows = read_rows(Path(path))
        self.config = config or {}
        self.data = [Item(row) for row in prepare_rows(list(rows or []))]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index: int) -> Item:
        return self.data[index]

    def get_attr_data(self, name: str) -> list[Any]:
        return [getattr(item, name, None) for item in self.data]

    @property
    def question(self) -> list[str]:
        return [item.contents for item in self.data]

    @property
    def id(self) -> list[str]:
        return [item.id for item in self.data]

    def update_output(self, key: str, values: list[Any]) -> None:
        if len(values) != len(self):
            raise ValueError(f"{key} has {len(values)} values for {len(self)} items")
        for item, value in zip(self.data, values):
            item.update_output(key, value)

    def save(self, path: str | Path) -> None:
        write_jsonl(Path(path), (item.to_dict() for item in self.data))

    def select(self, positions: list[int]) -> "Dataset":
        return Dataset([self.data[pos].to_dict() for pos in positions], config=self.config)


def split_dataset(dataset: Dataset, symbols: list[Any]) -> dict[Any, Dataset]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for item, symbol in zip(dataset, symbols):
        groups.setdefault(symbol, []).append(item.to_dict())
    return {key: Dataset(rows, config=dataset.config) for key, rows in groups.items()}


def merge_dataset(parts: dict[Any, Dataset], symbols: list[Any]) -> Dataset:
    cursors = {key: iter(value.data) for key, value in parts.items()}
    return Dataset([next(cursors[symbol]).to_dict() for symbol in symbols], config=next(iter(parts.values())).config)
