from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import ConcatDataset

from sentence_transformers import sampler as sampler_module
from sentence_transformers.sampler import NoDuplicatesBatchSampler, ProportionalBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture
def dummy_dataset() -> Dataset:
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 47, 3, 30, 3, ... 2],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """

    # Create a list of two 0's, two 1's, two 2's, ... two 49's. Then shuffle.
    values = [j for i in range(50) for j in (i, i)]
    random.shuffle(values)
    data = {"data": values, "label": [i % 2 for i in range(100)]}
    return Dataset.from_dict(data)


@pytest.fixture
def dummy_duplicates_dataset() -> Dataset:
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "anchor": ["anchor_1", "anchor_1", "anchor_1", ... "anchor_2", "anchor_2"],
        "positive": ["positive_1", "positive_1", "positive_1", ... "positive_2", "positive_2"],
    }
    """
    values = [{"anchor": "anchor_1", "positive": "positive_1"}] * 10 + [
        {"anchor": "anchor_2", "positive": "positive_2"}
    ] * 8
    return Dataset.from_list(values)


@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_group_by_label_batch_sampler_label_a(dummy_dataset: Dataset, precompute_hashes: bool) -> None:
    batch_size = 10

    sampler_kwargs = {}
    if precompute_hashes:
        if sampler_module.xxhash is None:
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": 1,
            "precompute_batch_size": 10,
        }

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=True,
        valid_label_columns=["label"],
        **sampler_kwargs,
    )

    batches = list(iter(sampler))

    # Assert all batch sizes are correct
    assert all(len(batch) == batch_size for batch in batches)

    # Assert batches contain no duplicate values
    for batch in batches:
        batch_values = [dummy_dataset[i]["data"] for i in batch]
        assert len(batch_values) == len(set(batch_values)), f"Batch {batch} contains duplicate values: {batch_values}"


@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_proportional_no_duplicates(
    dummy_duplicates_dataset: Dataset, drop_last: bool, precompute_hashes: bool
) -> None:
    batch_size = 2
    sampler_kwargs = {}
    if precompute_hashes:
        if sampler_module.xxhash is None:
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": 1,
            "precompute_batch_size": 10,
        }
    sampler_1 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=["anchor"],
        **sampler_kwargs,
    )
    sampler_2 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=["positive"],
        **sampler_kwargs,
    )

    concat_dataset = ConcatDataset([dummy_duplicates_dataset, dummy_duplicates_dataset])

    batch_sampler = ProportionalBatchSampler(
        concat_dataset, [sampler_1, sampler_2], generator=torch.Generator(), seed=12
    )
    batches = list(iter(batch_sampler))

    if drop_last:
        # If we drop the last batch (i.e. incomplete batches), we should have 16 batches out of the 18 possible,
        # because of the duplicates being skipped by the NoDuplicatesBatchSampler.
        # Notably, we should not crash like reported in #2816.
        assert len(batches) == 16
        # All batches are the same size: 2
        assert all(len(batch) == batch_size for batch in batches)
        assert len(sum(batches, [])) == 32
    else:
        # If we don't drop incomplete batches, we should be able to do 18 batches, and get more data.
        # Note: we don't get all data, because the NoDuplicatesBatchSampler will estimate the number of batches
        # and it would require more (non-complete) batches to get all data.
        assert len(batches) == 18
        assert len(sum(batches, [])) == 34
