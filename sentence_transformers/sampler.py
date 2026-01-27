from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from itertools import accumulate, cycle
from typing import Any

import torch
from torch.utils.data import BatchSampler, ConcatDataset, SubsetRandomSampler

from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)


class SetEpochMixin:
    """
    Required for a BatchSampler as the Trainer will call set_epoch on the BatchSampler at the beginning of each epoch.
    The BatchSampler can then set the generator seed accordingly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DefaultBatchSampler(SetEpochMixin, BatchSampler):
    """
    This sampler is the default batch sampler used in the SentenceTransformer library.
    It is equivalent to the PyTorch BatchSampler.

    Args:
        sampler (Sampler or Iterable): The sampler used for sampling elements from the dataset,
            such as SubsetRandomSampler.
        batch_size (int): Number of samples per batch.
        drop_last (bool): If True, drop the last incomplete batch if the dataset size
            is not divisible by the batch size.
        valid_label_columns (List[str], optional): List of column names to check for labels.
            The first column name from ``valid_label_columns`` found in the dataset will
            be used as the label column.
        generator (torch.Generator, optional): Optional random number generator for shuffling
            the indices.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last)
        self.valid_label_columns = valid_label_columns
        self.generator = generator
        self.seed = seed


class GroupByLabelBatchSampler(DefaultBatchSampler):
    """
    This sampler groups samples by their labels and aims to create batches such that
    each batch contains samples where the labels are as homogeneous as possible.
    This sampler is meant to be used alongside the ``Batch...TripletLoss`` classes, which
    require that each batch contains at least 2 examples per label class.

    Recommended for:
        - :class:`~sentence_transformers.losses.BatchAllTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardSoftMarginTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardTripletLoss`
        - :class:`~sentence_transformers.losses.BatchSemiHardTripletLoss`

    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch. Must be divisible by 2.
        drop_last (bool): If True, drop the last incomplete batch if the dataset size
            is not divisible by the batch size.
        valid_label_columns (List[str], optional): List of column names to check for labels.
            The first column name from ``valid_label_columns`` found in the dataset will
            be used as the label column.
        generator (torch.Generator, optional): Optional random number generator for shuffling
            the indices.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        self.dataset = dataset

        if self.batch_size % 2 == 1:
            raise ValueError("The batch size for `GroupByLabelBatchSampler` must be divisible by 2.")

        labels = self._determine_labels_to_use(dataset, self.valid_label_columns)
        groups = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            groups[label].append(sample_idx)

        self.groups = {
            label: sample_indices[:num_samples]
            for label, sample_indices in groups.items()
            if (num_samples := len(sample_indices) // 2 * 2)
        }

    @staticmethod
    def _determine_labels_to_use(dataset: Dataset, valid_label_columns: list[str] | None) -> list[Any]:
        for column_name in valid_label_columns or []:
            if column_name in dataset.column_names:
                return dataset[column_name]
        raise ValueError(
            f"None of the valid_label_columns {valid_label_columns} are in the dataset, "
            f"which only has these columns: {dataset.column_names}."
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        partial_batch = []
        unique_labels = list(self.groups.keys())
        for label_idx in torch.randperm(len(self.groups), generator=self.generator):
            label = unique_labels[label_idx]
            samples = self.groups[label]
            partial_batch.extend(samples)
            while len(partial_batch) >= self.batch_size:
                yield partial_batch[: self.batch_size]
                partial_batch = partial_batch[self.batch_size :]

        if not self.drop_last and partial_batch:
            yield partial_batch


class NoDuplicatesBatchSampler(DefaultBatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        """
        This sampler creates batches such that each batch contains samples where the values are unique,
        even across columns. This is useful when losses consider other samples in a batch to be in-batch
        negatives, and you want to ensure that the negatives are not duplicates of the anchor/positive sample.

        Recommended for:
            - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.losses.MultipleNegativesBidirectionalRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesBidirectionalRankingLoss`
            - :class:`~sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.losses.MegaBatchMarginLoss`
            - :class:`~sentence_transformers.losses.GISTEmbedLoss`
            - :class:`~sentence_transformers.losses.CachedGISTEmbedLoss`

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch if the dataset size
                is not divisible by the batch size.
            valid_label_columns (List[str], optional): List of column names to check for labels.
                The first column name from ``valid_label_columns`` found in the dataset will
                be used as the label column.
            generator (torch.Generator, optional): Optional random number generator for shuffling
                the indices.
            seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        if label_columns := set(dataset.column_names) & set(self.valid_label_columns or []):
            dataset = dataset.remove_columns(list(label_columns))
        self.dataset = dataset

    def __iter__(self) -> Iterator[list[int]]:
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        # We create a dictionary to None because we need a data structure that:
        # 1. Allows for cheap removal of elements
        # 2. Preserves the order of elements, i.e. remains random
        remaining_indices = dict.fromkeys(torch.randperm(len(self.dataset), generator=self.generator).tolist())
        while remaining_indices:
            batch_values = set()
            batch_indices = []
            for index in remaining_indices:
                sample_values = {str(value) for key, value in self.dataset[index].items() if key != "dataset_name"}
                if sample_values & batch_values:
                    continue

                batch_indices.append(index)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break

                batch_values.update(sample_values)

            else:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

            for index in batch_indices:
                del remaining_indices[index]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MultiDatasetDefaultBatchSampler(SetEpochMixin, BatchSampler, ABC):
    """
    Abstract base batch sampler that yields batches from multiple batch samplers.
    This class must be subclassed to implement specific sampling strategies, and
    cannot be used directly.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        if len(dataset.datasets) != len(batch_samplers):
            raise ValueError("The number of batch samplers must match the number of datasets in the ConcatDataset.")
        super().__init__(dataset, batch_size=batch_samplers[0].batch_size, drop_last=batch_samplers[0].drop_last)
        self.dataset = dataset
        self.batch_samplers = batch_samplers
        self.generator = generator
        self.seed = seed

    @abstractmethod
    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches from the underlying datasets in a specific order."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches in the sampler."""
        pass


class RoundRobinBatchSampler(MultiDatasetDefaultBatchSampler):
    """
    Batch sampler that yields batches in a round-robin fashion from multiple batch samplers, until one is exhausted.
    With this sampler, it's unlikely that all samples from each dataset are used, but we do ensure that each dataset
    is sampled from equally.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in cycle(range(len(batch_samplers))):
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]
            except StopIteration:
                # current iterator is apparently exhausted
                break

    def __len__(self) -> int:
        return min(len(sampler) for sampler in self.batch_samplers) * len(self.batch_samplers)


class ProportionalBatchSampler(MultiDatasetDefaultBatchSampler):
    """
    Batch sampler that samples from each dataset in proportion to its size, until all are exhausted simultaneously.
    With this sampler, all samples from each dataset are used and larger datasets are sampled from more frequently.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __iter__(self) -> Iterator[list[int]]:
        self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        num_batches = [len(sampler) for sampler in self.batch_samplers]
        dataset_indices = [idx for idx, length in enumerate(num_batches) for _ in range(length)]
        dataset_idx_sampler = SubsetRandomSampler(dataset_indices, generator=self.generator)

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in dataset_idx_sampler:
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]
            except StopIteration:
                continue

    def __len__(self) -> int:
        return sum([len(sampler) for sampler in self.batch_samplers])
