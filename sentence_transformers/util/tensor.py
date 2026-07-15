from __future__ import annotations

from typing import Any, overload

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch import Tensor, device


def _convert_to_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.
    Handles lists of sparse tensors by stacking them.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if isinstance(a, list):
        # Check if list contains sparse tensors
        if all(isinstance(x, Tensor) and x.is_sparse for x in a):
            # Stack sparse tensors while preserving sparsity
            return torch.stack([x.coalesce().to(dtype=torch.float32) for x in a])
        else:
            a = torch.tensor(a)
    elif not isinstance(a, Tensor):
        a = torch.tensor(a)
    if a.is_sparse:
        return a.to(dtype=torch.float32)
    return a


def _convert_to_batch(a: Tensor) -> Tensor:
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input data to a tensor with a batch dimension.
    Handles lists of sparse tensors by stacking them.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    if not embeddings.is_sparse:
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    embeddings = embeddings.coalesce()
    indices, values = embeddings.indices(), embeddings.values()

    # Compute row norms efficiently
    row_norms = torch.zeros(embeddings.size(0), device=embeddings.device)
    row_norms.index_add_(0, indices[0], values**2)
    row_norms = torch.sqrt(row_norms).index_select(0, indices[0])

    # Normalize values where norm > 0
    mask = row_norms > 0
    normalized_values = values.clone()
    normalized_values[mask] /= row_norms[mask]

    return torch.sparse_coo_tensor(indices, normalized_values, embeddings.size())


@overload
def truncate_embeddings(embeddings: np.ndarray, truncate_dim: int | None) -> np.ndarray: ...


@overload
def truncate_embeddings(embeddings: torch.Tensor, truncate_dim: int | None) -> torch.Tensor: ...


def truncate_embeddings(embeddings: np.ndarray | torch.Tensor, truncate_dim: int | None) -> np.ndarray | torch.Tensor:
    """
    Truncates the embeddings matrix.

    Args:
        embeddings (Union[np.ndarray, torch.Tensor]): Embeddings to truncate.
        truncate_dim (Optional[int]): The dimension to truncate sentence embeddings to. `None` does no truncation.

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> from sentence_transformers.util import truncate_embeddings
        >>> model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
        >>> embeddings = model.encode(["It's so nice outside!", "Today is a beautiful day.", "He drove to work earlier"])
        >>> embeddings.shape
        (3, 768)
        >>> model.similarity(embeddings, embeddings)
        tensor([[1.0000, 0.8100, 0.1426],
                [0.8100, 1.0000, 0.2121],
                [0.1426, 0.2121, 1.0000]])
        >>> truncated_embeddings = truncate_embeddings(embeddings, 128)
        >>> truncated_embeddings.shape
        >>> model.similarity(truncated_embeddings, truncated_embeddings)
        tensor([[1.0000, 0.8092, 0.1987],
                [0.8092, 1.0000, 0.2716],
                [0.1987, 0.2716, 1.0000]])

    Returns:
        Union[np.ndarray, torch.Tensor]: Truncated embeddings.
    """
    return embeddings[..., :truncate_dim]


@overload
def select_max_active_dims(embeddings: np.ndarray | torch.Tensor, max_active_dims: int) -> torch.Tensor: ...


@overload
def select_max_active_dims(embeddings: np.ndarray, max_active_dims: None) -> np.ndarray: ...


@overload
def select_max_active_dims(embeddings: torch.Tensor, max_active_dims: None) -> torch.Tensor: ...


def select_max_active_dims(
    embeddings: np.ndarray | torch.Tensor, max_active_dims: int | None
) -> np.ndarray | torch.Tensor:
    """
    Returns a new tensor with only the top-k values (in absolute terms) of each embedding, all others set to zero.

    The input embeddings are never modified in place.

    Args:
        embeddings (Union[np.ndarray, torch.Tensor]): Embeddings to sparsify by keeping only the largest values.
        max_active_dims (Optional[int]): Number of values to keep as non-zeros per embedding. `None` keeps all
            values, returning the embeddings as-is.

    Raises:
        ValueError: If `max_active_dims` is 0 or negative.

    Returns:
        Union[np.ndarray, torch.Tensor]: A new dense tensor of the same shape, with all but the top-k values per
            embedding set to zero. If `max_active_dims` is `None`, the embeddings are returned unchanged.
    """
    if max_active_dims is None:
        return embeddings
    if max_active_dims <= 0:
        raise ValueError(f"max_active_dims must be a positive integer, got {max_active_dims}.")

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)

    embedding_dim = embeddings.shape[-1]

    # Get the top-k indices for each embedding (by absolute value)
    _, top_indices = torch.topk(torch.abs(embeddings), k=min(max_active_dims, embedding_dim), dim=-1)

    # topk ran on absolute values, so gather the original (signed) values at those indices
    selected = torch.zeros_like(embeddings)
    selected.scatter_(-1, top_indices, embeddings.gather(-1, top_indices))

    return selected


def batch_to_device(batch: dict[str, Any], target_device: device) -> dict[str, Any]:
    """
    Send a PyTorch batch (i.e., a dictionary of string keys to Tensors) to a device (e.g. "cpu", "cuda", "mps").

    Args:
        batch (Dict[str, Tensor]): The batch to send to the device.
        target_device (torch.device): The target device (e.g. "cpu", "cuda", "mps").

    Returns:
        Dict[str, Tensor]: The batch with tensors sent to the target device.
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def to_scipy_coo(x: Tensor) -> coo_matrix:
    x = x.coalesce()
    indices = x.indices().cpu().numpy()
    values = x.values().cpu().numpy()
    return coo_matrix((values, (indices[0], indices[1])), shape=x.shape)


def compute_count_vector(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute count vector from sparse embeddings indicating how many samples have non-zero values in each dimension.

    Args:
        embeddings: Sparse tensor of shape (batch_size, vocab_size) or (vocab_size,)

    Returns:
        Count vector of shape (vocab_size,)
    """
    if not embeddings.is_sparse:
        embeddings = embeddings.to_sparse()

    # Coalesce to ensure indices are sorted and unique
    embeddings = embeddings.coalesce()

    count_vector = torch.zeros(embeddings.size(-1), device=embeddings.device, dtype=torch.int32)
    if embeddings.dim() == 1:
        # Single embedding case
        count_vector[embeddings.indices().squeeze()] = 1
        return count_vector
    elif embeddings.dim() == 2:
        # Batch case
        if embeddings.values().numel() > 0:
            indices = embeddings.indices()
            # Count how many samples have non-zero values in each dimension
            unique_dims, counts = torch.unique(indices[1], return_counts=True)
            count_vector[unique_dims] = counts.int()

        return count_vector
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {embeddings.dim()}D")
