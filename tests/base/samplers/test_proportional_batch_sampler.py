from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.base.sampler import NoDuplicatesBatchSampler, ProportionalBatchSampler

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
from datasets import Dataset


@pytest.mark.parametrize("seed", [0, 42])
def test_proportional_preserves_batches_beyond_estimated_length(seed: int) -> None:
    datasets = [
        Dataset.from_dict({"anchor": ["same query"] * 6, "positive": [f"answer {i}" for i in range(6)]}),
        Dataset.from_dict({"anchor": [f"query {i}" for i in range(4)]}),
    ]
    children = [
        NoDuplicatesBatchSampler(dataset, batch_size=2, drop_last=False, generator=torch.Generator(), seed=seed)
        for dataset in datasets
    ]
    sampler = ProportionalBatchSampler(ConcatDataset(datasets), children, generator=torch.Generator(), seed=seed)

    batches = list(sampler)

    assert len(sampler) == 5  # Child lengths are estimates; iteration must still exhaust them.
    assert len(batches) == 8
    assert sorted(index for batch in batches for index in batch) == list(range(10))
    assert all(all(index < 6 for index in batch) or all(index >= 6 for index in batch) for batch in batches)
    assert list(sampler) == batches


@pytest.mark.parametrize("drop_last", [False, True])
def test_proportional_exact_lengths(drop_last: bool) -> None:
    datasets = [list(range(5)), list(range(8))]
    children = [BatchSampler(SequentialSampler(dataset), 2, drop_last) for dataset in datasets]
    sampler = ProportionalBatchSampler(ConcatDataset(datasets), children, generator=torch.Generator(), seed=42)

    batches = list(sampler)

    assert len(batches) == len(sampler)
    expected = list(range(4 if drop_last else 5)) + list(range(5, 13))
    assert sorted(index for batch in batches for index in batch) == expected


@pytest.mark.parametrize("estimate", [0, 1, 10])
def test_proportional_exhausts_inaccurately_sized_children(estimate: int) -> None:
    class EstimatedBatchSampler(BatchSampler):
        def __len__(self) -> int:
            return estimate

    datasets = [list(range(6)), [], list(range(4))]
    children = [EstimatedBatchSampler(SequentialSampler(dataset), 2, False) for dataset in datasets]
    sampler = ProportionalBatchSampler(ConcatDataset(datasets), children, generator=torch.Generator(), seed=42)

    batches = list(sampler)

    assert len(batches) == 5
    assert sorted(index for batch in batches for index in batch) == list(range(10))
