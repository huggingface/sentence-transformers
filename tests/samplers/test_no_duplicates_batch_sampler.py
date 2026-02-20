from __future__ import annotations

import random

import numpy as np
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


def _reference_no_duplicates_batches(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    seed: int,
    valid_label_columns: list[str] | None = None,
) -> list[list[int]]:
    """Reference implementation of the historical dict-based iteration logic."""
    if label_columns := set(dataset.column_names) & set(valid_label_columns or []):
        dataset = dataset.remove_columns(list(label_columns))

    generator = torch.Generator()
    generator.manual_seed(seed)

    def get_sample_values(index: int) -> set[str]:
        return {str(value) for key, value in dataset[index].items() if key != "dataset_name"}

    def has_overlap(sample_values: set[str], batch_values: set[str]) -> bool:
        return not sample_values.isdisjoint(batch_values)

    remaining_indices = dict.fromkeys(torch.randperm(len(dataset), generator=generator).tolist())
    batches: list[list[int]] = []
    while remaining_indices:
        batch_values: set[str] = set()
        batch_indices: list[int] = []
        for index in remaining_indices:
            sample_values = get_sample_values(index)
            if has_overlap(sample_values, batch_values):
                continue
            batch_indices.append(index)
            if len(batch_indices) == batch_size:
                batches.append(batch_indices)
                break
            batch_values.update(sample_values)
        else:
            if not drop_last:
                batches.append(batch_indices)

        for index in batch_indices:
            del remaining_indices[index]
    return batches


@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_group_by_label_batch_sampler_label_a(dummy_dataset: Dataset, precompute_hashes: bool) -> None:
    batch_size = 10

    sampler_kwargs = {}
    if precompute_hashes:
        if sampler_module.xxhash is None:
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": 0,
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
            "precompute_num_proc": 0,
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


@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_no_duplicates_batch_sampler_matches_reference_algorithm(
    dummy_dataset: Dataset, drop_last: bool, precompute_hashes: bool
) -> None:
    batch_size = 10
    seed = 1337
    valid_label_columns = ["label"]
    sampler_kwargs = {}
    if precompute_hashes:
        if sampler_module.xxhash is None:
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": 0,
            "precompute_batch_size": 10,
        }

    expected_batches = _reference_no_duplicates_batches(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        seed=seed,
        valid_label_columns=valid_label_columns,
    )

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=valid_label_columns,
        generator=torch.Generator(),
        seed=seed,
        **sampler_kwargs,
    )

    actual_batches = list(iter(sampler))
    assert actual_batches == expected_batches
    assert len(sum(actual_batches, [])) == len(sum(expected_batches, []))


def test_no_duplicates_batch_sampler_precomputed_hashes_are_int64(dummy_dataset: Dataset) -> None:
    if sampler_module.xxhash is None:
        pytest.skip("xxhash not installed")

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=10,
        drop_last=True,
        valid_label_columns=["label"],
        precompute_hashes=True,
        precompute_num_proc=0,
        precompute_batch_size=10,
    )
    sampler._build_hashes()

    assert sampler._row_hashes is not None
    assert sampler._row_hashes.dtype == np.int64


def test_no_duplicates_batch_sampler_list_values_match_between_hash_and_non_hash() -> None:
    if sampler_module.xxhash is None:
        pytest.skip("xxhash not installed")

    # This shape intentionally creates cross-column overlap:
    # row0 has list value ["a", "b"], row1 has scalar "a" in another column.
    # Hash/non-hash modes must still produce identical batching decisions.
    dataset = Dataset.from_list(
        [
            {"anchor": ["a", "b"], "positive": "p0"},
            {"anchor": ["c", "d"], "positive": "a"},
        ]
    )

    sampler_kwargs = {
        "dataset": dataset,
        "batch_size": 2,
        "drop_last": False,
        "generator": torch.Generator(),
        "seed": 1337,
    }
    default_sampler = NoDuplicatesBatchSampler(**sampler_kwargs)
    hashed_sampler = NoDuplicatesBatchSampler(
        **sampler_kwargs,
        precompute_hashes=True,
        precompute_num_proc=0,
        precompute_batch_size=2,
    )

    default_batches = list(iter(default_sampler))
    hashed_batches = list(iter(hashed_sampler))

    assert hashed_batches == default_batches
    assert len(default_batches) == 1
    assert len(default_batches[0]) == 2


def test_no_duplicates_batch_sampler_uses_int64_indices_when_int32_range_is_exceeded(
    dummy_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate "too many rows for int32" without allocating a huge dataset.
    real_iinfo = sampler_module.np.iinfo

    class _FakeIInfo:
        def __init__(self, max_value: int) -> None:
            self.max = max_value

    def fake_iinfo(dtype):
        if dtype is np.int32:
            return _FakeIInfo(max_value=8)
        return real_iinfo(dtype)

    captured_dtypes: dict[str, object] = {}
    real_randperm = sampler_module.torch.randperm

    def fake_randperm(*args, **kwargs):
        # Capture dtype selected for shuffled row indices.
        captured_dtypes["index_dtype"] = kwargs.get("dtype")
        return real_randperm(*args, **kwargs)

    monkeypatch.setattr(sampler_module.np, "iinfo", fake_iinfo)
    monkeypatch.setattr(sampler_module.torch, "randperm", fake_randperm)

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=10,
        drop_last=True,
        valid_label_columns=["label"],
        generator=torch.Generator(),
        seed=123,
    )
    _ = list(iter(sampler))

    assert captured_dtypes["index_dtype"] == torch.int64
