from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def map(self, fn: Any, fn_kwargs: dict[str, Any] | None = None) -> FakeDataset:
        kwargs = fn_kwargs or {}
        return FakeDataset([fn(row, **kwargs) for row in self.rows])


def build_fake_datasets_module(
    dataset_splits: Mapping[str, list[str]],
    candidate_subsets: Mapping[str, str] | None = None,
) -> Any:
    data: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    split_names: dict[tuple[str, str], list[str]] = {}
    candidate_subsets = candidate_subsets or {}

    def add_split(dataset_id: str, split_name: str) -> None:
        data[(dataset_id, "corpus", split_name)] = [
            {"_id": f"{split_name}-d1", "text": "Document 1"},
            {"_id": f"{split_name}-d2", "text": "Document 2"},
        ]
        data[(dataset_id, "queries", split_name)] = [{"_id": f"{split_name}-q1", "text": "Query 1"}]
        data[(dataset_id, "qrels", split_name)] = [{"query-id": f"{split_name}-q1", "corpus-id": f"{split_name}-d1"}]
        for subset_name, column_name in candidate_subsets.items():
            data[(dataset_id, subset_name, split_name)] = [
                {"query-id": f"{split_name}-q1", column_name: [f"{split_name}-d2", f"{split_name}-d1"]}
            ]

    for dataset_id, splits in dataset_splits.items():
        for split_name in splits:
            add_split(dataset_id, split_name)
        for subset_name in ["corpus", "queries", "qrels", *candidate_subsets]:
            split_names[(dataset_id, subset_name)] = list(splits)

    def load_dataset(dataset_id: str, subset: str, split: str) -> FakeDataset:
        return FakeDataset(data[(dataset_id, subset, split)])

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        return split_names[(dataset_id, subset)]

    return SimpleNamespace(load_dataset=load_dataset, get_dataset_split_names=get_dataset_split_names)
