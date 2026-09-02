from __future__ import annotations

import pickle

import pytest

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
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
