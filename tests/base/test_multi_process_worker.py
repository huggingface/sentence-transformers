from __future__ import annotations

import queue

import pytest
import torch
from torch import Tensor

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.model import _move_tensors_to_cpu


class _OffDeviceTensor(torch.Tensor):
    """A real tensor that reports a non-CPU device, so the move can be exercised without one.

    Only ``device`` is faked. ``cpu()`` hands back a plain tensor whose device is the real one,
    which is what a tensor coming off an accelerator would do.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", 0)

    def cpu(self) -> Tensor:
        return self.as_subclass(Tensor)


def _off_device(*values: float) -> _OffDeviceTensor:
    return torch.tensor(values).as_subclass(_OffDeviceTensor)


def _on_cpu(value) -> bool:
    return isinstance(value, Tensor) and value.device.type == "cpu"


class _FakeInputQueue:
    """Hands the worker one chunk and then behaves like a drained queue."""

    def __init__(self, item) -> None:
        self._items = [item]

    def get(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


class _FakeResultsQueue(list):
    def put(self, item) -> None:
        self.append(item)


class _LeavesResultsOnDevice:
    """Stands in for a model whose encode/predict returns tensors still on the accelerator.

    ``encode`` returns the ``output_value=None`` shape (one feature dict per input) and
    ``predict`` the list-of-scores shape, which are the shapes the bare-tensor check misses.
    """

    def encode(self, inputs, device=None, **kwargs):
        return [{"token_embeddings": _off_device(1.0, 2.0), "attention_mask": _off_device(1.0, 1.0)} for _ in inputs]

    def predict(self, pairs, device=None, **kwargs):
        return [_off_device(1.0) for _ in pairs]


def test_move_tensors_to_cpu_walks_every_shape_encode_can_return():
    assert _on_cpu(_move_tensors_to_cpu(_off_device(1.0)))
    assert all(_on_cpu(item) for item in _move_tensors_to_cpu([_off_device(1.0), _off_device(2.0)]))

    moved = _move_tensors_to_cpu([{"token_embeddings": _off_device(1.0), "attention_mask": _off_device(1.0)}])
    assert all(_on_cpu(value) for value in moved[0].values())

    # values are preserved, and anything that is not a tensor is passed through untouched
    assert _move_tensors_to_cpu(_off_device(1.0, 2.0)).tolist() == [1.0, 2.0]
    assert _move_tensors_to_cpu(["a", 3, None]) == ["a", 3, None]

    already_on_cpu = torch.tensor([1.0])
    assert _move_tensors_to_cpu(already_on_cpu) is already_on_cpu


@pytest.mark.parametrize(
    "model_class, chunk",
    [
        (SentenceTransformer, ["a", "b"]),
        (SparseEncoder, ["a", "b"]),
        (MultiVectorEncoder, ["a", "b"]),
        (CrossEncoder, [("a", "b")]),
    ],
)
def test_multi_process_worker_returns_cpu_tensors(model_class, chunk):
    """Results leave the worker through a queue, so nothing may still be on the accelerator: the
    shared handle a device tensor is sent as is only readable while its worker is alive, and
    ``stop_multi_process_pool`` tears the workers down before the caller reads the embeddings."""
    input_queue = _FakeInputQueue([0, chunk, {}])
    results_queue = _FakeResultsQueue()

    model_class._multi_process_worker("cpu", _LeavesResultsOnDevice(), input_queue, results_queue)

    assert len(results_queue) == 1
    chunk_id, result = results_queue[0]
    assert chunk_id == 0
    for entry in result:
        values = list(entry.values()) if isinstance(entry, dict) else [entry]
        assert values and all(_on_cpu(value) for value in values)
