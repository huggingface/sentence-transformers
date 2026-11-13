"""Contract tests for synchronized multi-dataset batches.

These tests derive task identities and row coverage from yielded dataset indices,
then apply the real installed Accelerate BatchSamplerShard. Coverage and alignment are checked from the emitted row indices.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.base.sampler import RoundRobinBatchSampler, SynchronizedMultiDatasetBatchSampler

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
pytest.importorskip("accelerate", exc_type=ModuleNotFoundError)

from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset


def fixture_dataset(lengths):
    return ConcatDataset(
        [
            Dataset.from_dict(
                {
                    "anchor": [f"task{task}:anchor{i}" for i in range(length)],
                    "positive": [f"task{task}:positive{i}" for i in range(length)],
                }
            )
            for task, length in enumerate(lengths)
        ]
    )


def make_plan(
    lengths=(49, 21), batch_sizes=(8, 2), world_size=2, strategy="round_robin", seed=123, accumulation=1, strict=False
):
    return SynchronizedMultiDatasetBatchSampler(
        fixture_dataset(lengths),
        batch_sizes=list(batch_sizes),
        world_size=world_size,
        strategy=strategy,
        seed=seed,
        gradient_accumulation_steps=accumulation,
        preserve_batch_frequencies=strict,
    )


def task_of_index(index, lengths):
    offset = 0
    for task, length in enumerate(lengths):
        if offset <= index < offset + length:
            return task
        offset += length
    raise AssertionError(f"index {index} is outside dataset of lengths {lengths}")


def global_groups(plan, lengths, world_size):
    raw = [list(batch) for batch in plan]
    assert len(raw) == len(plan)
    assert len(raw) % world_size == 0
    groups = []
    for start in range(0, len(raw), world_size):
        shards = raw[start : start + world_size]
        indices = [i for batch in shards for i in batch]
        task_ids = {task_of_index(i, lengths) for i in indices}
        assert len(task_ids) == 1
        groups.append((task_ids.pop(), shards, indices))
    return raw, groups


@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("strategy", ["round_robin", "proportional"])
@pytest.mark.parametrize("batch_sizes", [(8, 2), (2, 8)])
def test_exact_count_coverage_and_real_accelerate_sharding(world_size, strategy, batch_sizes):
    lengths = [world_size * batch_sizes[0] * 3 + 1, world_size * batch_sizes[1] * 5 + 1]
    plan = make_plan(lengths, batch_sizes, world_size, strategy)
    raw, groups = global_groups(plan, lengths, world_size)
    capacities = [n // (world_size * b) for n, b in zip(lengths, batch_sizes)]
    expected_counts = [min(capacities)] * 2 if strategy == "round_robin" else capacities
    expected_microsteps = sum(expected_counts)
    assert plan.drop_last is True
    assert len(plan) == world_size * expected_microsteps
    assert Counter(task for task, _, _ in groups) == Counter(dict(enumerate(expected_counts)))
    assert list(plan.retained_batch_counts) == expected_counts
    if strategy == "round_robin":
        assert [task for task, _, _ in groups] == [0, 1] * min(capacities)
    emitted = []
    for task, shards, indices in groups:
        assert all(len(shard) == batch_sizes[task] for shard in shards)
        assert len(indices) == world_size * batch_sizes[task]
        assert len(set(indices)) == len(indices)
        emitted.extend(indices)
    assert len(set(emitted)) == len(emitted), "global drop must not pad/resample rows"

    rank_batches = []
    for rank in range(world_size):
        shard = BatchSamplerShard(
            plan, num_processes=world_size, process_index=rank, split_batches=False, even_batches=False
        )
        yielded = [list(batch) for batch in shard]
        assert len(shard) == len(yielded) == expected_microsteps
        assert yielded == raw[rank::world_size]
        rank_batches.append(yielded)
    for step in range(expected_microsteps):
        tasks = {task_of_index(batch[0], lengths) for batch in [x[step] for x in rank_batches]}
        assert len(tasks) == 1


@pytest.mark.parametrize("strategy", ["round_robin", "proportional"])
def test_epoch_seed_and_len_reads_are_reproducible(strategy):
    plan = make_plan(strategy=strategy)
    initial = [list(x) for x in plan]
    for _ in range(10):
        assert len(plan) == len(initial)
    assert [list(x) for x in plan] == initial
    assert [list(x) for x in make_plan(strategy=strategy)] == initial
    plan.set_epoch(1)
    second = [list(x) for x in plan]
    assert second != initial
    assert len(second) == len(initial)
    plan.set_epoch(1)
    assert [list(x) for x in plan] == second
    plan.set_epoch(0)
    assert [list(x) for x in plan] == initial
    assert [list(x) for x in make_plan(strategy=strategy, seed=987)] != initial


def test_no_global_rng_mutation_and_schedule_does_not_change_sample_streams():
    before = torch.random.get_rng_state().clone()
    lengths = (49, 21)
    _, rr = global_groups(make_plan(lengths=lengths, strategy="round_robin"), lengths, 2)
    _, prop = global_groups(make_plan(lengths=lengths, strategy="proportional"), lengths, 2)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    for task in [0, 1]:
        rr_rows = [indices for selected, _, indices in rr if selected == task]
        prop_rows = [indices for selected, _, indices in prop if selected == task]
        assert rr_rows == prop_rows[: len(rr_rows)]


def test_changing_another_dataset_does_not_reshuffle_first_dataset():
    lengths1, lengths2 = (49, 21), (49, 41)
    _, plan1 = global_groups(make_plan(lengths=lengths1, strategy="proportional"), lengths1, 2)
    _, plan2 = global_groups(make_plan(lengths=lengths2, strategy="proportional"), lengths2, 2)
    rows1 = [indices for task, _, indices in plan1 if task == 0]
    rows2 = [indices for task, _, indices in plan2 if task == 0]
    assert rows1 == rows2


def test_proportional_uses_retained_global_counts_not_raw_rows_or_local_counts():
    lengths, batch_sizes = (40, 8), (8, 2)
    plan = make_plan(lengths, batch_sizes, strategy="proportional")
    _, groups = global_groups(plan, lengths, 2)
    # Raw sample ratio is 5:1; local batch counts are 5:4; retained global counts are 2:2.
    assert Counter(task for task, _, _ in groups) == {0: 2, 1: 2}
    assert list(plan.retained_batch_counts) == [2, 2]


@pytest.mark.parametrize("strategy", ["round_robin", "proportional"])
def test_strict_frequency_rejects_nondivisible_local_batch_count(strategy):
    with pytest.raises(ValueError):
        make_plan(lengths=(40, 8), batch_sizes=(8, 2), strategy=strategy, strict=True)


def test_strict_frequency_allows_a_tail_already_dropped_by_local_drop_last():
    # floor(17/4)=4 and floor(9/2)=4: both local batch counts divide by W=2.
    # The one-row tails were also dropped by the old local drop_last behavior.
    plan = make_plan(lengths=(17, 9), batch_sizes=(4, 2), strategy="proportional", strict=True)
    assert list(plan.retained_batch_counts) == [2, 2]
    assert len(list(plan)) == len(plan) == 8


def test_complete_accumulation_window_and_partial_window_rejection():
    plan = make_plan(lengths=(48, 12), accumulation=2)
    assert len(plan) // 2 == 6
    assert len(plan) // (2 * 2) == 3
    with pytest.raises(ValueError):
        make_plan(lengths=(32, 12), strategy="proportional", accumulation=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 0},
        {"world_size": -1},
        {"batch_sizes": (0, 2)},
        {"batch_sizes": (-1, 2)},
        {"batch_sizes": (8,)},
        {"batch_sizes": (8, 2, 1)},
        {"accumulation": 0},
        {"accumulation": -1},
        {"strategy": "unrecognized"},
        {"lengths": (15, 8)},
        {"lengths": (16, 3)},
    ],
)
def test_invalid_plan_is_rejected_at_construction(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_plan(**kwargs)


def test_three_tasks_round_robin_has_no_partial_final_task_round():
    lengths, sizes, world = (13, 25, 9), (2, 3, 1), 2
    plan = make_plan(lengths, sizes, world)
    _, groups = global_groups(plan, lengths, world)
    assert [task for task, _, _ in groups] == [0, 1, 2] * 3
    assert list(plan.retained_batch_counts) == [3, 3, 3]


def test_existing_round_robin_order_is_not_silently_rewritten():
    dataset = fixture_dataset([8, 8])
    children = [BatchSampler(SequentialSampler(child), batch_size=2, drop_last=True) for child in dataset.datasets]
    original = RoundRobinBatchSampler(dataset, children)
    expected = [[0, 1], [8, 9], [2, 3], [10, 11], [4, 5], [12, 13], [6, 7], [14, 15]]
    assert list(original) == expected
