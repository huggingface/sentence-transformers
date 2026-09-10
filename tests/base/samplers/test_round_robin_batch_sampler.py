from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.base.sampler import NoDuplicatesBatchSampler, RoundRobinBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )

DATASET_LENGTH = 25


@pytest.fixture
def dummy_concat_dataset() -> ConcatDataset:
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ... , 23, 24, 100, 101, ..., 123, 124],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """
    values_1 = list(range(DATASET_LENGTH))
    labels = [x % 2 for x in values_1]
    dataset_1 = Dataset.from_dict({"data": values_1, "label": labels})

    values_2 = [x + 100 for x in values_1] + [x + 200 for x in values_1]
    dataset_2 = Dataset.from_dict({"data": values_2, "label": labels + labels})

    return ConcatDataset([dataset_1, dataset_2])


def test_round_robin_batch_sampler(dummy_concat_dataset: ConcatDataset) -> None:
    batch_size = 4
    batch_sampler_1 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[0]))), batch_size=batch_size, drop_last=True
    )
    batch_sampler_2 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[1]))), batch_size=batch_size, drop_last=True
    )

    sampler = RoundRobinBatchSampler(dataset=dummy_concat_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2])
    batches = list(iter(sampler))

    # Despite the second dataset being larger (2 * DATASET_LENGTH), we still only sample DATASET_LENGTH // batch_size batches from each dataset
    # because the RoundRobinBatchSampler should stop sampling once it has sampled all elements from one dataset
    assert len(batches) == 2 * DATASET_LENGTH // batch_size
    assert len(sampler) == len(batches)

    # Assert that batches are produced in a round-robin fashion
    for i in range(0, len(batches), 2):
        # Batch from the first part of the dataset
        batch_1 = batches[i]
        assert all(dummy_concat_dataset[idx]["data"] < 100 for idx in batch_1), (
            f"Batch {i} contains data from the second part of the dataset: {[dummy_concat_dataset[idx]['data'] for idx in batch_1]}"
        )

        # Batch from the second part of the dataset
        batch_2 = batches[i + 1]
        assert all(dummy_concat_dataset[idx]["data"] >= 100 for idx in batch_2), (
            f"Batch {i + 1} contains data from the first part of the dataset: {[dummy_concat_dataset[idx]['data'] for idx in batch_2]}"
        )


def test_round_robin_batch_sampler_value_error(dummy_concat_dataset: ConcatDataset) -> None:
    batch_size = 4
    batch_sampler_1 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)
    batch_sampler_3 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)

    with pytest.raises(
        ValueError, match="The number of batch samplers must match the number of datasets in the ConcatDataset"
    ):
        RoundRobinBatchSampler(
            dataset=dummy_concat_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2, batch_sampler_3]
        )


@pytest.mark.parametrize(
    ("dataset_lengths", "drop_last", "expected_batches"),
    [
        ((20, 12), True, 6),
        ((12, 20), True, 6),
        ((20, 24, 12), True, 9),
        ((20, 0), True, 0),
        ((0, 20), False, 0),
        ((20, 3), True, 0),
        ((20, 3), False, 2),
        ((21, 13), False, 8),
    ],
)
def test_round_robin_batch_sampler_stops_at_advertised_length(
    dataset_lengths: tuple[int, ...], drop_last: bool, expected_batches: int
) -> None:
    datasets = [Dataset.from_dict({"data": list(range(length))}) for length in dataset_lengths]
    concat_dataset = ConcatDataset(datasets)
    batch_samplers = [
        BatchSampler(SequentialSampler(range(len(dataset))), batch_size=4, drop_last=drop_last) for dataset in datasets
    ]

    sampler = RoundRobinBatchSampler(dataset=concat_dataset, batch_samplers=batch_samplers)

    assert len(sampler) == expected_batches
    assert len(list(sampler)) == len(sampler)


@pytest.mark.parametrize(
    ("values", "drop_last", "expected_batches"),
    [
        (([0] * 4, [1] * 6), False, 8),
        ((list(range(8)), [0, 0, 0, 1]), True, 4),
        ((list(range(8)), [0] * 4), True, 0),
    ],
)
def test_round_robin_batch_sampler_with_no_duplicates(
    values: tuple[list[int], list[int]], drop_last: bool, expected_batches: int
) -> None:
    datasets = [Dataset.from_dict({"data": data}) for data in values]
    batch_samplers = [
        NoDuplicatesBatchSampler(
            dataset=dataset,
            batch_size=2,
            drop_last=drop_last,
            generator=torch.Generator(),
            seed=42,
        )
        for dataset in datasets
    ]
    sampler = RoundRobinBatchSampler(dataset=ConcatDataset(datasets), batch_samplers=batch_samplers)

    batches = list(sampler)

    assert len(batches) == expected_batches
    assert [int(batch[0] >= len(datasets[0])) for batch in batches] == [0, 1] * (expected_batches // 2)


@pytest.mark.parametrize(
    ("batch_counts", "estimated_counts", "expected_batches"),
    [
        ((4, 6), (2, 3), 8),
        ((4, 1), (4, 2), 2),
        ((4, 0), (4, 2), 0),
    ],
)
def test_round_robin_batch_sampler_with_estimated_lengths(
    batch_counts: tuple[int, int], estimated_counts: tuple[int, int], expected_batches: int
) -> None:
    class EstimatedBatchSampler(BatchSampler):
        def __init__(self, dataset: Dataset, estimated_count: int) -> None:
            super().__init__(SequentialSampler(dataset), batch_size=2, drop_last=True)
            self.estimated_count = estimated_count

        def __len__(self) -> int:
            return self.estimated_count

    datasets = [Dataset.from_dict({"data": list(range(count * 2))}) for count in batch_counts]
    batch_samplers = [
        EstimatedBatchSampler(dataset, estimated_count) for dataset, estimated_count in zip(datasets, estimated_counts)
    ]
    sampler = RoundRobinBatchSampler(dataset=ConcatDataset(datasets), batch_samplers=batch_samplers)

    batches = list(sampler)

    assert len(sampler) != expected_batches
    assert len(batches) == expected_batches
    assert [int(batch[0] >= len(datasets[0])) for batch in batches] == [0, 1] * (expected_batches // 2)


def test_multi_dataset_batch_sampler_propagates_epoch(dummy_concat_dataset: ConcatDataset) -> None:
    batch_samplers = [
        NoDuplicatesBatchSampler(
            dataset=dataset,
            batch_size=4,
            drop_last=True,
            generator=torch.Generator(),
            seed=42,
        )
        for dataset in dummy_concat_dataset.datasets
    ]
    sampler = RoundRobinBatchSampler(
        dataset=dummy_concat_dataset,
        batch_samplers=batch_samplers,
        generator=torch.Generator(),
        seed=42,
    )

    sampler.set_epoch(3)

    assert sampler.epoch == 3
    assert all(batch_sampler.epoch == 3 for batch_sampler in batch_samplers)
