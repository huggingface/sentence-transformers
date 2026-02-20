from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from itertools import accumulate, cycle

import numpy as np
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SubsetRandomSampler

try:
    import xxhash
except ImportError:  # pragma: no cover - optional dependency
    xxhash = None

from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)

_XXHASH_INT64_MAX = 1 << 63
_XXHASH_UINT64_MAX = 1 << 64
_EXCLUDE_DATASET_COLUMNS = {"dataset_name"}


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
    def _determine_labels_to_use(dataset: Dataset, valid_label_columns: list[str] | None) -> list[object]:
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


def _xxhash_int64(value: str) -> int:
    # Convert uint64 -> int64 to keep values compatible with Arrow int64 storage.
    hashed = xxhash.xxh64_intdigest(value)
    if hashed >= _XXHASH_INT64_MAX:
        hashed -= _XXHASH_UINT64_MAX
    return hashed


def _hash_batch(
    batch: dict[str, list[object]],
    columns: list[str],
    exclude_columns: set[str],
) -> dict[str, list[list[int]]]:
    # Must be defined at module scope because datasets.map with num_proc pickles this function.
    # Build per-row hash lists so we can later do fast overlap checks without re-reading the dataset.
    active_columns = [column for column in columns if column not in exclude_columns]
    batch_size = len(batch[active_columns[0]]) if active_columns else len(next(iter(batch.values()), []))
    if not active_columns:
        return {"__hashes": [[] for _ in range(batch_size)]}
    hashes: list[list[int]] = []
    for row_idx in range(batch_size):
        row_hashes: list[int] = []
        for column in active_columns:
            value = batch[column][row_idx]
            # Keep semantics aligned with the non-hash path, which compares
            # stringified per-column values (including list values as a whole).
            row_hashes.append(_xxhash_int64(str(value)))
        hashes.append(row_hashes)
    return {"__hashes": hashes}


class NoDuplicatesBatchSampler(DefaultBatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        precompute_hashes: bool = False,
        precompute_num_proc: int | None = None,
        precompute_batch_size: int = 1000,
    ) -> None:
        """
        This sampler creates batches such that each batch contains samples where the values are unique,
        even across columns. This is useful when losses consider other samples in a batch to be in-batch
        negatives, and you want to ensure that the negatives are not duplicates of the anchor/positive sample.

        Recommended for:
            - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss`
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
            precompute_hashes (bool, optional): If True, precompute xxhash 64-bit values for dataset
                fields using ``datasets.map`` to speed up duplicate checks. Requires ``xxhash`` to
                be installed and uses additional memory: in theory roughly
                ``len(dataset) * num_columns * 8`` bytes for the dense int64 hash matrix,
                although actual memory usage may therefore differ in practice. Defaults to False.
            precompute_num_proc (int, optional): Number of processes for hashing with ``datasets.map``.
                If set to ``None``, defaults to ``min(8, cpu_count - 1)`` when ``precompute_hashes``
                is True.
            precompute_batch_size (int, optional): Batch size for ``datasets.map`` hashing.
                Defaults to 1000.
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
        self.precompute_hashes = precompute_hashes
        self.precompute_num_proc = precompute_num_proc
        self.precompute_batch_size = precompute_batch_size
        self._row_hashes: np.ndarray | None = None
        if self.precompute_hashes:
            if xxhash is None:
                raise ImportError(
                    "NoDuplicatesBatchSampler with precompute_hashes=True requires `xxhash`. "
                    "Install `xxhash` to use this option."
                )
            if self.precompute_num_proc is None:
                cpu_count = os.cpu_count() or 1
                # Leave one core free to avoid saturating the system when hashing.
                default_workers = max(1, min(8, cpu_count - 1))
                self.precompute_num_proc = default_workers

    def _build_hashes(self) -> None:
        # Build once lazily on first iteration, then reuse across epochs.
        # Hashes depend on dataset content, not epoch seed/order.
        if not self.precompute_hashes or self._row_hashes is not None:
            return
        columns = list(self.dataset.column_names)
        # Precompute hash values once to avoid repeated string processing per batch.
        # Use num_proc to parallelize hashing across CPU cores.
        hash_ds = self.dataset.map(
            _hash_batch,
            batched=True,
            batch_size=self.precompute_batch_size,
            num_proc=self.precompute_num_proc,
            remove_columns=columns,
            fn_kwargs={"columns": columns, "exclude_columns": _EXCLUDE_DATASET_COLUMNS},
            desc="Hashing dataset values",
        )
        import pyarrow as pa

        try:
            column = hash_ds.data.column("__hashes")
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            if not isinstance(column, (pa.ListArray, pa.LargeListArray)):
                raise ValueError("Expected a list column for hashed values.")

            row_count = len(column)
            if row_count == 0:
                self._row_hashes = np.zeros((0, 0), dtype=np.int64)
                return

            offsets = column.offsets.to_numpy(zero_copy_only=False)
            row_size = int(offsets[1] - offsets[0])
            # Dense ndarray storage below requires a fixed number of hashed
            # values per row to allow safe reshape(row_count, row_size).
            if row_size < 0 or not np.all(np.diff(offsets) == row_size):
                raise ValueError("Hashed rows have varying lengths.")
            # If every row has the same length, store as a dense ndarray to reduce overhead.
            values = column.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if values.size != row_count * row_size:
                raise ValueError("Unexpected hashed value buffer size.")
            self._row_hashes = values.reshape((row_count, row_size))
        finally:
            # Drop the temporary dataset to release Arrow buffers promptly.
            del hash_ds

    def __iter__(self) -> Iterator[list[int]]:
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        if self.precompute_hashes:
            self._build_hashes()
            row_hashes: np.ndarray = self._row_hashes

            def get_sample_values(index: int) -> set[str] | np.ndarray:
                return row_hashes[index]

        else:

            def get_sample_values(index: int) -> set[str] | np.ndarray:
                return {
                    str(value) for key, value in self.dataset[index].items() if key not in _EXCLUDE_DATASET_COLUMNS
                }

        def _has_overlap(sample_values: set[str] | np.ndarray, batch_values: set[str | np.int64]) -> bool:
            # Non-hash path with set[str] allows for disjoint overlap checks
            if isinstance(sample_values, set):
                return not sample_values.isdisjoint(batch_values)
            # Hash path with ndarray does set instance checks
            return any(value in batch_values for value in sample_values)

        num_rows = len(self.dataset)
        if num_rows == 0:
            return

        # Create a random numpy permutation using int32 (or int64 if necessary)
        index_dtype = torch.int32 if num_rows <= np.iinfo(np.int32).max else torch.int64
        remaining_indices = torch.randperm(num_rows, generator=self.generator, dtype=index_dtype).numpy()

        # Plus a singly linked list over shuffled positions, where the last position is marked with -1
        # for simple termination
        position_dtype = np.int32 if num_rows <= np.iinfo(np.int32).max else np.int64
        next_positions = np.arange(1, num_rows + 1, dtype=position_dtype)
        next_positions[-1] = -1
        head_position = 0

        while head_position != -1:
            batch_values: set[str | np.int64] = set()
            batch_indices: list[int] = []
            current_position = head_position
            previous_position = -1
            full_batch = False
            while current_position != -1:
                next_position = int(next_positions[current_position])
                index = int(remaining_indices[current_position])
                sample_values = get_sample_values(index)
                if _has_overlap(sample_values, batch_values):
                    # Defer conflicting samples to later batches instead of reordering them.
                    previous_position = current_position
                    current_position = next_position
                    continue

                batch_indices.append(index)
                if previous_position == -1:
                    head_position = next_position
                else:
                    next_positions[previous_position] = next_position

                if len(batch_indices) == self.batch_size:
                    full_batch = True
                    yield batch_indices
                    break

                batch_values.update(sample_values)
                current_position = next_position

            if not full_batch:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

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
