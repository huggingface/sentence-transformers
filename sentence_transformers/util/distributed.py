from __future__ import annotations

import torch
import torch.distributed as dist
from transformers.utils import logging

from .environment import is_dist_initialized

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = logging.get_logger(__name__)


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
