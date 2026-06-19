from __future__ import annotations

import torch
import torch.distributed as dist
from transformers.utils import logging

from .environment import is_dist_initialized

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
