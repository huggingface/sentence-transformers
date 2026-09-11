from __future__ import annotations

import pickle
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from transformers.utils import logging

from .environment import is_dist_initialized
from .tensor import _move_tensors_to_cpu, _move_tensors_to_device

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = logging.get_logger(__name__)


def get_rank() -> int:
    """The rank of the current process in the distributed group, or ``0`` when not distributed."""
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """The number of processes in the distributed group, or ``1`` when not distributed."""
    if is_dist_initialized():
        return dist.get_world_size()
    return 1


def all_gather(tensor: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
    """
    Gathers a tensor from each distributed rank into a list. Always retains gradients for the local rank's tensor,
    and optionally retains gradients for the gathered tensors if `with_grad` is True.

    Args:
        tensor (torch.Tensor): The tensor to gather from each rank.
        with_grad (bool, optional): If True, the local rank's tensor retains its gradients. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the gathered tensors from all ranks, concatenated along the first dimension.
        If torch.distributed is not available or not initialized, returns the original tensor.
    """

    if is_dist_initialized():
        if with_grad:
            gathered_tensors = torch.distributed.nn.all_gather(tensor)
        else:
            world_size = dist.get_world_size()
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

            # Perform all_gather.
            dist.all_gather(gathered_tensors, tensor)

            # Replace local rank's tensor with the original (retaining gradients).
            local_rank = dist.get_rank()
            gathered_tensors[local_rank] = tensor
        return torch.cat(gathered_tensors, dim=0)

    # Warn once about uninitialized or single-GPU usage.
    warning = (
        "Trying to gather while torch.distributed is not available or has not been initialized, "
        "returning the original (local) tensor. This is expected if you are "
        "only using one GPU; consider not using gathering to remove this warning."
    )
    logger.warning_once(warning)
    return tensor


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gathers a tensor from each distributed rank into a list, retaining gradients for the local rank's tensor.

    Args:
        tensor (torch.Tensor): The tensor to gather from each rank.

    Returns:
        torch.Tensor: A tensor containing the gathered tensors from all ranks, concatenated along the first dimension.
        If torch.distributed is not available or not initialized, returns the original tensor.
    """
    return all_gather(tensor, with_grad=True)


def all_gather_padded(
    tensor: torch.Tensor, mask: torch.Tensor, with_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather a ``(B, T, D)`` token-embedding tensor and its ``(B, T)`` mask across ranks, padding
    the token axis to the cross-rank max ``T`` first.

    ``all_gather`` requires every rank to contribute an identically-shaped tensor, but multi-vector
    batches pad each column to its own per-rank batch-longest ``T``, which differs across ranks. This
    reduces the global max ``T``, pads both the embeddings and the mask up to it (keeping them
    aligned), then gathers. Embeddings are gathered with ``with_grad``. The mask never carries a
    gradient.

    Args:
        tensor (torch.Tensor): ``(B, T, D)`` token embeddings to gather.
        mask (torch.Tensor): ``(B, T)`` mask to gather, padded with ``False`` to match ``tensor``.
        with_grad (bool, optional): Retain gradients for the embeddings (see :func:`all_gather`).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gathered ``(sum(B), T_max, D)`` embeddings and
        ``(sum(B), T_max)`` mask. Without an initialised process group this only forwards to
        :func:`all_gather` (no padding needed, all_gather returns the local tensor).
    """
    if is_dist_initialized():
        local_max = torch.tensor(tensor.size(1), device=tensor.device)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        T_max = int(local_max.item())
        if tensor.size(1) < T_max:
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, T_max - tensor.size(1)))
            mask = torch.nn.functional.pad(mask, (0, T_max - mask.size(1)))
    return all_gather(tensor, with_grad=with_grad), all_gather(mask)


class _DistributedInference:
    def __init__(self, model: BaseModel) -> None:
        self.model = model
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __call__(self, inputs: list, **kwargs) -> Any:
        kwargs["show_progress_bar"] = False
        if not inputs:
            return self.model._inference(inputs, **kwargs)

        size, remainder = divmod(len(inputs), self.world_size)
        shards = []
        for rank in range(self.world_size):
            start = rank * size + min(rank, remainder)
            end = start + size + (rank < remainder)
            shards.append(("infer", (inputs[start:end], kwargs)))
        try:
            pickle.dumps(shards)
        except Exception:
            return self.model._inference(inputs, **kwargs)
        _, payload = self._scatter(shards)
        results = self._infer_and_gather(payload)
        outputs, errors = [], []
        for rank, (output, error) in enumerate(results):
            if error is not None:
                errors.append(f"Rank {rank}:\n{error}")
            elif output is not None:
                outputs.append(output)
        if errors:
            raise RuntimeError("Distributed evaluator inference failed:\n" + "\n".join(errors))

        if isinstance(outputs[0], torch.Tensor):
            output = torch.cat(outputs)
        else:
            output = [item for shard in outputs for item in shard]
        if self.model.device.type != "cpu" and not kwargs.get("save_to_cpu"):
            output = _move_tensors_to_device(output, self.model.device)
        return output

    def _scatter(self, messages: list | None = None) -> tuple[str, Any]:
        payload = [None]
        with torch.inference_mode(False):
            dist.scatter_object_list(payload, messages, src=0)
        return payload[0]

    def _infer_and_gather(self, payload: tuple[list, dict[str, Any]]) -> list | None:
        inputs, kwargs = payload
        try:
            output = _move_tensors_to_cpu(self.model._inference(inputs, **kwargs)) if inputs else None
            result = (output, None)
            pickle.dumps(result)
        except Exception:
            result = (None, traceback.format_exc())
        results = [None] * self.world_size if self.rank == 0 else None
        with torch.inference_mode(False):
            dist.gather_object(result, results, dst=0)
        return results

    def serve(self) -> None:
        while True:
            command, payload = self._scatter()
            if command == "stop":
                if payload is not None:
                    raise RuntimeError(f"Distributed evaluator failed on rank 0:\n{payload}")
                return
            self._infer_and_gather(payload)

    def stop(self, error: str | None) -> None:
        self._scatter([("stop", error)] * self.world_size)


@contextmanager
def distributed_evaluation(model: BaseModel, enabled: bool = True) -> Iterator[None]:
    """Share evaluator inference across existing DDP ranks while rank zero computes the metrics."""
    if not enabled or get_world_size() == 1:
        yield
        return

    inference = _DistributedInference(model)
    if get_rank() != 0:
        inference.serve()
        yield
        return

    previous = model._distributed_inference
    model._distributed_inference = inference
    error = None
    try:
        yield
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        model._distributed_inference = previous
        inference.stop(error)
