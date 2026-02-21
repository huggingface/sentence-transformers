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
def balanced_dataset():
    """100 samples, 5 labels with 20 samples each."""
    data = {"data": list(range(100)), "label": [i % 5 for i in range(100)]}
    return Dataset.from_dict(data)


@pytest.fixture
def two_label_dataset():
    """100 samples, 2 labels with 50 samples each."""
    data = {"data": list(range(100)), "label": [i % 2 for i in range(100)]}
    return Dataset.from_dict(data)


@pytest.fixture
def imbalanced_dataset():
    """140 samples: label 0 has 90, label 1 has 30, label 2 has 20."""
    labels = [0] * 90 + [1] * 30 + [2] * 20
    data = {"data": list(range(140)), "label": labels}
    return Dataset.from_dict(data)


def test_every_label_appears_at_least_twice_per_batch(balanced_dataset: Dataset) -> None:
    sampler = GroupByLabelBatchSampler(
        dataset=balanced_dataset, batch_size=16, drop_last=True, valid_label_columns=["label"]
    )
    labels_col = balanced_dataset["label"]
    for batch in sampler:
        counts = Counter(labels_col[i] for i in batch)
        for label, count in counts.items():
            assert count >= 2, f"Label {label} appears only {count} time(s) in batch"


def test_drop_last_true_no_short_batches(balanced_dataset: Dataset) -> None:
    batch_size = 16
    sampler = GroupByLabelBatchSampler(
        dataset=balanced_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["label"]
    )
    batches = list(sampler)
    p = sampler.labels_per_batch
    k = sampler.samples_per_label
    expected_size = p * k
    for batch in batches:
        assert len(batch) == expected_size


def test_drop_last_false_yields_remainder(two_label_dataset: Dataset) -> None:
    batch_size = 32
    sampler_drop = GroupByLabelBatchSampler(
        dataset=two_label_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["label"]
    )
    sampler_keep = GroupByLabelBatchSampler(
        dataset=two_label_dataset, batch_size=batch_size, drop_last=False, valid_label_columns=["label"]
    )
    batches_drop = list(sampler_drop)
    batches_keep = list(sampler_keep)
    assert len(batches_keep) >= len(batches_drop)
    total_samples_keep = sum(len(b) for b in batches_keep)
    total_samples_drop = sum(len(b) for b in batches_drop)
    assert total_samples_keep >= total_samples_drop


def test_sample_coverage(balanced_dataset: Dataset) -> None:
    """Nearly all samples should be used exactly once per epoch."""
    sampler = GroupByLabelBatchSampler(
        dataset=balanced_dataset, batch_size=16, drop_last=False, valid_label_columns=["label"]
    )
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
    assert len(all_indices) == len(set(all_indices)), "Some samples appear more than once"
    assert len(all_indices) >= len(balanced_dataset) * 0.8, "Too many samples dropped"


def test_len_matches_iteration(balanced_dataset: Dataset) -> None:
    for drop_last in [True, False]:
        sampler = GroupByLabelBatchSampler(
            dataset=balanced_dataset, batch_size=16, drop_last=drop_last, valid_label_columns=["label"]
        )
        batches = list(sampler)
        assert len(sampler) == len(batches), f"drop_last={drop_last}: __len__={len(sampler)} != actual={len(batches)}"


def test_raises_on_single_label() -> None:
    data = {"data": list(range(20)), "label": [0] * 20}
    ds = Dataset.from_dict(data)
    with pytest.raises(ValueError, match="at least 2"):
        GroupByLabelBatchSampler(dataset=ds, batch_size=8, drop_last=False, valid_label_columns=["label"])


def test_raises_on_invalid_batch_size(two_label_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        GroupByLabelBatchSampler(
            dataset=two_label_dataset, batch_size=7, drop_last=False, valid_label_columns=["label"]
        )
    with pytest.raises(ValueError):
        GroupByLabelBatchSampler(
            dataset=two_label_dataset, batch_size=2, drop_last=False, valid_label_columns=["label"]
        )


def test_imbalanced_dataset_multi_class(imbalanced_dataset: Dataset) -> None:
    sampler = GroupByLabelBatchSampler(
        dataset=imbalanced_dataset, batch_size=16, drop_last=True, valid_label_columns=["label"]
    )
    labels_col = imbalanced_dataset["label"]
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        batch_labels = {labels_col[i] for i in batch}
        assert len(batch_labels) >= 2


def test_minority_labels_in_pk_batches() -> None:
    """Labels too small for K full draws should still be scheduled into a proper PK batch."""
    labels = [0] * 1000 + [1] * 500 + [2] * 3
    ds = Dataset.from_dict({"data": list(range(len(labels))), "label": labels})
    sampler = GroupByLabelBatchSampler(dataset=ds, batch_size=32, drop_last=True, valid_label_columns=["label"])
    all_indices = set()
    for batch in sampler:
        all_indices.update(batch)
    minority_indices = set(range(1500, 1503))
    assert minority_indices.issubset(all_indices), "Minority label samples should appear in scheduled PK batches"


def test_two_labels_with_large_batch(two_label_dataset: Dataset) -> None:
    """With 2 labels and batch_size=32, P=2, K=16."""
    sampler = GroupByLabelBatchSampler(
        dataset=two_label_dataset, batch_size=32, drop_last=True, valid_label_columns=["label"]
    )
    assert sampler.labels_per_batch == 2
    assert sampler.samples_per_label == 16
    labels_col = two_label_dataset["label"]
    for batch in sampler:
        counts = Counter(labels_col[i] for i in batch)
        assert len(counts) == 2
        for count in counts.values():
            assert count == 16


def _compute_scheduled_efficiency(label_sizes: list[int], batch_size: int) -> float:
    """Helper: return fraction of samples that appear in properly PK-balanced batches (not the remainder)."""
    labels = []
    for i, size in enumerate(label_sizes):
        labels.extend([i] * size)
    data = {"data": list(range(len(labels))), "label": labels}
    ds = Dataset.from_dict(data)
    sampler = GroupByLabelBatchSampler(dataset=ds, batch_size=batch_size, drop_last=True)
    scheduled = sum(len(b) for b in sampler)
    return scheduled / len(labels)


def test_efficiency_few_imbalanced_labels() -> None:
    """6 imbalanced labels, with fixed P=6, only ~9.4% of samples would land in PK-balanced batches."""
    label_sizes = [1250, 1223, 1162, 896, 835, 86]
    efficiency = _compute_scheduled_efficiency(label_sizes, batch_size=32)
    assert efficiency >= 0.90, f"Efficiency {efficiency:.1%} is too low -- most training data is being ignored"


def test_efficiency_many_imbalanced_labels() -> None:
    """Many imbalanced labels, with fixed P, only ~52% of samples would land in PK-balanced batches."""
    label_sizes = [962, 464, 421, 363, 276, 274, 218, 217, 207, 191]
    label_sizes += [80 - i * 3 for i in range(10)]
    label_sizes += [20 - i for i in range(10)]
    label_sizes += [4, 4, 6, 8, 9, 9, 9, 10, 11, 11]
    efficiency = _compute_scheduled_efficiency(label_sizes, batch_size=32)
    assert efficiency >= 0.85, f"Efficiency {efficiency:.1%} is too low -- dominant labels are underused"
