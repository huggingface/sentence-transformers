from __future__ import annotations

import pickle

import pytest
import torch
from torch import Tensor

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.util import _move_tensors_to_cpu
from tests.utils import CrashingModel


class _StopWorker(BaseException):
    """Breaks the worker's endless loop once its single chunk has been handed over.

    Not an ``Exception``, because a worker that caught it would report it as a chunk failure and
    then loop on it forever.
    """


class _OneShotQueue:
    """Hands out a single chunk, then stops the worker loop."""

    def __init__(self, item) -> None:
        self._items = [item]

    def get(self, *args, **kwargs):
        if not self._items:
            raise _StopWorker
        return self._items.pop(0)


class _RecordingQueue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


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


class _LeavesResultsOnDevice:
    """Stands in for a model whose encode/predict returns tensors still on the accelerator.

    ``encode`` returns the ``output_value=None`` shape (one feature dict per input) and
    ``predict`` the list-of-scores shape, which are the shapes the bare-tensor check misses.
    """

    def encode(self, inputs, device=None, **kwargs):
        return [{"token_embeddings": _off_device(1.0, 2.0), "attention_mask": _off_device(1.0, 1.0)} for _ in inputs]

    def predict(self, pairs, device=None, **kwargs):
        return [_off_device(1.0) for _ in pairs]


@pytest.mark.parametrize("unpicklable", (False, True))
@pytest.mark.parametrize(
    "model_class", (SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder), ids=lambda cls: cls.__name__
)
def test_multi_process_worker_reports_inference_failure(model_class, unpicklable: bool) -> None:
    results_queue = _RecordingQueue()
    with pytest.raises(_StopWorker):
        model_class._multi_process_worker(
            "cpu", CrashingModel(unpicklable=unpicklable), _OneShotQueue([0, ["text"], {}]), results_queue
        )

    # _multi_process blocks for exactly one result per submitted chunk
    assert len(results_queue.items) == 1
    chunk_id, result = results_queue.items[0]
    assert chunk_id == 0
    assert isinstance(result, Exception)
    assert "simulated worker crash" in str(result)
    # A payload that does not survive the queue's pickling is dropped by its feeder thread
    pickle.loads(pickle.dumps(result))
    # The replacement for an exception that could not be pickled carries the worker-side frames
    if unpicklable:
        assert "in _crash" in str(result)


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
    ids=lambda value: value.__name__ if isinstance(value, type) else "",
)
def test_multi_process_worker_returns_cpu_tensors(model_class, chunk):
    """Results leave the worker through a queue, so nothing may still be on the accelerator.

    A device tensor crosses as a shared handle that is only readable while its worker is alive, and
    ``stop_multi_process_pool`` tears the workers down before the caller reads the embeddings.
    """
    results_queue = _RecordingQueue()
    with pytest.raises(_StopWorker):
        model_class._multi_process_worker(
            "cpu", _LeavesResultsOnDevice(), _OneShotQueue([0, chunk, {}]), results_queue
        )

    assert len(results_queue.items) == 1
    chunk_id, result = results_queue.items[0]
    assert chunk_id == 0
    for entry in result:
        values = list(entry.values()) if isinstance(entry, dict) else [entry]
        assert values and all(_on_cpu(value) for value in values)
