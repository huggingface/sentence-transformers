from __future__ import annotations

import pytest
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.sampler import RoundRobinBatchSampler
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
