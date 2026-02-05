from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics import pairwise_distances
from torch import Tensor
from transformers.utils import logging

from .tensor import _convert_to_batch_tensor, _convert_to_tensor, normalize_embeddings, to_scipy_coo

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = logging.get_logger(__name__)


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1)).to_dense()


def pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    # Handle sparse tensors
    if a.is_sparse or b.is_sparse:
        a_norm = normalize_embeddings(a)
        b_norm = normalize_embeddings(b)
        return (a_norm * b_norm).sum(dim=-1).to_dense()
    else:
        return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b)).to_dense()


def dot_score(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = dot_prod(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return torch.mm(a, b.transpose(0, 1)).to_dense()


def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return (a * b).sum(dim=-1).to_dense()


def manhattan_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the manhattan similarity (i.e., negative distance) between two tensors.
    Handles sparse tensors without converting to dense when possible.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -manhattan_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    if a.is_sparse or b.is_sparse:
        logger.warning_once("Using scipy for sparse Manhattan similarity computation.")

        a_coo = to_scipy_coo(a)
        b_coo = to_scipy_coo(b)
        dist = pairwise_distances(a_coo, b_coo, metric="manhattan")
        return torch.from_numpy(-dist).float().to(a.device).to_dense()

    else:
        return -torch.cdist(a, b, p=1.0).to_dense()


def pairwise_manhattan_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    """
    Computes the manhattan similarity (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -manhattan_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sum(torch.abs(a - b), dim=-1).to_dense()


def euclidean_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the euclidean similarity (i.e., negative distance) between two tensors.
    Handles sparse tensors without converting to dense when possible.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -euclidean_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    if a.is_sparse:
        a_norm_sq = torch.sparse.sum(a * a, dim=1).to_dense().unsqueeze(1)  # Shape (N, 1)
        b_norm_sq = torch.sparse.sum(b * b, dim=1).to_dense().unsqueeze(0)  # Shape (1, M)
        dot_product = torch.matmul(a, b.t()).to_dense()  # Shape (N, M)

        # Calculate squared distance
        squared_dist = a_norm_sq - 2 * dot_product + b_norm_sq

        # Ensure no negative values before square root (due to numerical precision)
        squared_dist = torch.clamp(squared_dist, min=0.0)

        return -torch.sqrt(squared_dist).to_dense()
    else:
        return -torch.cdist(a, b, p=2.0)


def pairwise_euclidean_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    """
    Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1)).to_dense()


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance. See :class:`~sentence_transformers.sentence_transformer.losses.AnglELoss`
    or https://huggingface.co/papers/2309.12871 for more information.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.

    Returns:
        Tensor: Vector with res[i] = angle_sim(a[i], b[i])
    """
    if x.is_sparse:
        logger.warning_once("Pairwise angle similarity does not support sparse tensors. Converting to dense.")
        x = x.coalesce().to_dense()
        y = y.coalesce().to_dense()

    x = _convert_to_tensor(x)
    y = _convert_to_tensor(y)

    # Pad tensors if the embedding dimension is odd, as torch.chunk requires even dimensions
    if x.shape[1] % 2 != 0:
        x = torch.nn.functional.pad(x, (0, 1), mode="constant", value=0)
        y = torch.nn.functional.pad(y, (0, 1), mode="constant", value=0)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


class SimilarityFunction(Enum):
    """
    Enum class for supported similarity functions. The following functions are supported:

    - ``SimilarityFunction.COSINE`` (``"cosine"``): Cosine similarity
    - ``SimilarityFunction.DOT_PRODUCT`` (``"dot"``, ``dot_product``): Dot product similarity
    - ``SimilarityFunction.EUCLIDEAN`` (``"euclidean"``): Euclidean distance
    - ``SimilarityFunction.MANHATTAN`` (``"manhattan"``): Manhattan distance
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    DOT = "dot"  # Alias for DOT_PRODUCT
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @staticmethod
    def to_similarity_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function name or enum value to the corresponding similarity function.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]: The corresponding similarity function.

        Raises:
            ValueError: If the provided function is not supported.

        Example:
            >>> similarity_fn = SimilarityFunction.to_similarity_fn("cosine")
            >>> similarity_scores = similarity_fn(embeddings1, embeddings2)
            >>> similarity_scores
            tensor([[0.3952, 0.0554],
                    [0.0992, 0.1570]])
        """
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return cos_sim
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return dot_score
        if similarity_function == SimilarityFunction.MANHATTAN:
            return manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return euclidean_sim

        raise ValueError(
            f"The provided function {similarity_function} is not supported. Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function into a pairwise similarity function.

        The pairwise similarity function returns the diagonal vector from the similarity matrix, i.e. it only
        computes the similarity(a[i], b[i]) for each i in the range of the input tensors, rather than
        computing the similarity between all pairs of a and b.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]: The pairwise similarity function.

        Raises:
            ValueError: If the provided similarity function is not supported.

        Example:
            >>> pairwise_fn = SimilarityFunction.to_similarity_pairwise_fn("cosine")
            >>> similarity_scores = pairwise_fn(embeddings1, embeddings2)
            >>> similarity_scores
            tensor([0.3952, 0.1570])
        """
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return pairwise_cos_sim
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return pairwise_dot_score
        if similarity_function == SimilarityFunction.MANHATTAN:
            return pairwise_manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return pairwise_euclidean_sim

        raise ValueError(
            f"The provided function {similarity_function} is not supported. Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def possible_values() -> list[str]:
        """
        Returns a list of possible values for the SimilarityFunction enum.

        Returns:
            list: A list of possible values for the SimilarityFunction enum.

        Example:
            >>> possible_values = SimilarityFunction.possible_values()
            >>> possible_values
            ['cosine', 'dot', 'euclidean', 'manhattan']
        """
        return [m.value for m in SimilarityFunction]
