from __future__ import annotations

from collections import Counter

import pytest

from sentence_transformers.sampler import GroupByLabelBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture
def dummy_dataset():
    """

    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ..., 99],
        "label_a": [0, 1, 0, 1, ..., 0, 1],
        "label_b": [0, 1, 2, 3, 4, 0, ..., 4]
    }
    """
    data = {"data": list(range(100)), "label_a": [i % 2 for i in range(100)], "label_b": [i % 5 for i in range(100)]}
    return Dataset.from_dict(data)


@pytest.fixture
def dummy_uneven_dataset():
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": ["a"] * 51,
        "label": [0] * 17 + [1] * 17 + [2] * 17,
    }
    """
    data = {"data": ["a"] * 51, "label": [0] * 17 + [1] * 17 + [2] * 17}
    return Dataset.from_dict(data)


def test_group_by_label_batch_sampler_label_a(dummy_dataset: Dataset) -> None:
    batch_size = 10

    sampler = GroupByLabelBatchSampler(
        dataset=dummy_dataset, batch_size=batch_size, drop_last=False, valid_label_columns=["label_a", "label_b"]
    )

    batches = list(iter(sampler))
    assert all(len(batch) == batch_size for batch in batches)

    # Check if all labels within each batch are identical
    # In this case, label_a has 50 0's and 50 1's, so with a batch size of 10 we expect each batch to
    # have only 0's or only 1's.
    for batch in batches:
        labels = [dummy_dataset[int(idx)]["label_a"] for idx in batch]
        assert len(set(labels)) == 1, f"Batch {batch} does not have identical labels: {labels}"


def test_group_by_label_batch_sampler_label_b(dummy_dataset: Dataset) -> None:
    batch_size = 8

    sampler = GroupByLabelBatchSampler(
        dataset=dummy_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["label_b"]
    )

    # drop_last=True, so each batch should be the same length and the last batch is dropped.
    batches = list(iter(sampler))
    assert all(len(batch) == batch_size for batch in batches), (
        "Not all batches are the same size, while drop_last was True."
    )

    # Assert that we have the expected number of total samples in the batches.
    assert sum(len(batch) for batch in batches) == 100 // batch_size * batch_size

    # Since we have 20 occurrences each of label_b values 0, 1, 2, 3 and 4 and a batch_size of 8, we expect each batch
    # to have either 4 or 8 samples with the same label. (The first two batches are 16 samples of the first label,
    # leaving 4 for the third batch. There 4 of the next label are added, leaving 16 for the next two batches, and so on.)
    for batch in batches:
        labels = [dummy_dataset[int(idx)]["label_b"] for idx in batch]
        counts = list(Counter(labels).values())
        assert counts == [8] or counts == [4, 4]


def test_group_by_label_batch_sampler_uneven_dataset(dummy_uneven_dataset: Dataset) -> None:
    batch_size = 8

    sampler = GroupByLabelBatchSampler(
        dataset=dummy_uneven_dataset, batch_size=batch_size, drop_last=False, valid_label_columns=["label"]
    )

    # With a batch_size of 8 and 17 samples per label; verify that every label in a batch occurs at least twice.
    # We accept some tiny data loss (1 sample per label) due to the uneven number of samples per label.
    batches = list(iter(sampler))
    for batch in batches:
        labels = [dummy_uneven_dataset[int(idx)]["label"] for idx in batch]
        counts = list(Counter(labels).values())
        assert [count > 1 for count in counts]
