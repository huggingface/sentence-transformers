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


def pairwise_manhattan_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
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


def pairwise_euclidean_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
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


def maxsim(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
    document_chunk_size: int | None = None,
) -> Tensor:
    """
    Computes the MaxSim (late-interaction) score between two collections of multi-vector embeddings.

    For each query in ``a`` and document in ``b``, the score is the sum over query tokens of the maximum
    similarity to any document token: ``sum_i max_j (a_i . b_j)``. This is the scoring function used by
    ColBERT-style models.

    Args:
        a (Union[list, np.ndarray, Tensor]): Query embeddings. Either a 3D tensor of shape
            ``(batch_a, num_query_tokens, embedding_dim)`` (pre-padded) or a list of 2D tensors of shape
            ``(num_query_tokens_i, embedding_dim)`` (variable-length, padded internally).
        b (Union[list, np.ndarray, Tensor]): Document embeddings. Either a 3D tensor of shape
            ``(batch_b, num_doc_tokens, embedding_dim)`` or a list of 2D tensors of shape
            ``(num_doc_tokens_i, embedding_dim)``.
        a_mask (Tensor, optional): Boolean or float mask for query tokens, shape ``(batch_a, num_query_tokens)``.
            Tokens with a 0 / False entry are excluded from the sum. Defaults to None (use all tokens).
        b_mask (Tensor, optional): Boolean or float mask for document tokens, shape ``(batch_b, num_doc_tokens)``.
            Tokens with a 0 / False entry are excluded from the max. Defaults to None (use all tokens).
        document_chunk_size (int, optional): If set, iterate the einsum + max-reduction over document
            chunks of this size along the ``b`` axis. Keeps the full ``(a, b, s, t)`` 4D intermediate
            from being materialized at once. Useful for evaluation against large corpora. Defaults to
            None (single einsum over the full ``b`` axis).

    Returns:
        Tensor: Matrix with ``res[i][j]`` = MaxSim(a[i], b[j]), shape ``(batch_a, batch_b)``.
    """
    a, a_mask_padded = _pad_multi_vector_inputs(a, a_mask)
    b, b_mask_padded = _pad_multi_vector_inputs(b, b_mask)

    if document_chunk_size is None or document_chunk_size >= b.size(0):
        reduced = _maxsim_reduce_documents(a, b, b_mask_padded)
        # Query tokens are reduced with sum, so masked tokens simply contribute 0.
        if a_mask_padded is not None:
            reduced = reduced * a_mask_padded.unsqueeze(1)
        return reduced.sum(dim=-1)

    score_chunks = []
    for d_start in range(0, b.size(0), document_chunk_size):
        d_end = d_start + document_chunk_size
        chunk_b_mask = b_mask_padded[d_start:d_end] if b_mask_padded is not None else None
        reduced = _maxsim_reduce_documents(a, b[d_start:d_end], chunk_b_mask)
        # Mask and sum per chunk: concatenating the (batch_a, chunk, q_tokens) reductions first would
        # rebuild a full-corpus-width intermediate and defeat the chunking.
        if a_mask_padded is not None:
            reduced = reduced * a_mask_padded.unsqueeze(1)
        score_chunks.append(reduced.sum(dim=-1))
    return torch.cat(score_chunks, dim=1)


def _maxsim_reduce_documents(a: Tensor, b: Tensor, b_mask: Tensor | None) -> Tensor:
    """Compute the ``(a_batch, b_batch, q_tokens)`` per-query-token max over document tokens for one
    document chunk. Pulled out of :func:`maxsim` so the chunked and unchunked paths share the
    reduction kernel verbatim.
    """
    scores = torch.einsum("ash,bth->abst", a, b)
    # Document tokens are reduced with max, so masked (padding) tokens are sent to the dtype minimum:
    # multiplying by 0 would let a padding token win the max over a genuinely negative similarity.
    # Note: PyLate's colbert_scores masks by multiplying by 0. We exclude masked tokens from the max
    # instead (matching xtr_scores). Parity tests pass against PyLate because real tokens dominate the
    # max in practice, but the dtype-min approach is the more correct one and the one we keep.
    if b_mask is not None:
        scores = scores.masked_fill(~b_mask.bool().unsqueeze(0).unsqueeze(2), torch.finfo(scores.dtype).min)
    return scores.max(dim=-1).values


def maxsim_pairwise(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
) -> Tensor:
    """
    Computes the pairwise MaxSim (late-interaction) score between each query-document pair.

    For each ``i``, computes the MaxSim score between ``a[i]`` and ``b[i]``: the sum over query tokens of the
    maximum similarity to any document token. This is the pairwise analogue of :func:`maxsim`.

    Args:
        a (Union[list, np.ndarray, Tensor]): Query embeddings. Either a 3D tensor of shape
            ``(batch, num_query_tokens, embedding_dim)`` (pre-padded) or a list of 2D tensors with
            shape ``(num_query_tokens_i, embedding_dim)``.
        b (Union[list, np.ndarray, Tensor]): Document embeddings. Either a 3D tensor of shape
            ``(batch, num_doc_tokens, embedding_dim)`` or a list of 2D tensors with shape
            ``(num_doc_tokens_i, embedding_dim)``.
        a_mask (Tensor, optional): Boolean or float mask for query tokens, shape ``(batch, num_query_tokens)``.
            Tokens with a 0 / False entry are excluded from the sum. Defaults to None (use all tokens).
        b_mask (Tensor, optional): Boolean or float mask for document tokens, shape ``(batch, num_doc_tokens)``.
            Tokens with a 0 / False entry are excluded from the max. Defaults to None (use all tokens).

    Returns:
        Tensor: Vector with ``res[i]`` = MaxSim(a[i], b[i]), shape ``(batch,)``.
    """
    if isinstance(a, list) or isinstance(b, list):
        scores = []
        for i in range(len(a)):
            qi = _convert_to_tensor(a[i])
            di = _convert_to_tensor(b[i])
            score = torch.einsum("sh,th->st", qi, di)
            if b_mask is not None:
                score = score.masked_fill(
                    ~_convert_to_tensor(b_mask[i]).bool().unsqueeze(0), torch.finfo(score.dtype).min
                )
            maxed = score.max(dim=-1).values
            if a_mask is not None:
                maxed = maxed * _convert_to_tensor(a_mask[i])
            scores.append(maxed.sum())
        return torch.stack(scores, dim=0)

    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    # Mirror maxsim: derive a mask from all-zero (pad) rows so pad cannot win the max.
    if a_mask is None and a.dim() == 3:
        a_mask = _zero_row_mask(a)
    if b_mask is None and b.dim() == 3:
        b_mask = _zero_row_mask(b)
    scores = torch.einsum("bsh,bth->bst", a, b)
    # Documents reduced with max (mask to dtype min so padding never wins). Queries reduced with sum (mask to 0).
    if b_mask is not None:
        scores = scores.masked_fill(~b_mask.bool().unsqueeze(1), torch.finfo(scores.dtype).min)
    scores = scores.max(dim=-1).values
    if a_mask is not None:
        scores = scores * a_mask
    return scores.sum(dim=-1)


def _zero_row_mask(padded: Tensor) -> Tensor:
    """Derive a ``(B, T)`` mask from a pre-padded ``(B, T, D)`` token-embedding tensor by treating
    all-zero rows as padding.

    Real token embeddings normally carry information in at least one dim, so a fully-zero row is the
    all-but-impossible-except-by-pad signal. This matches the padding pattern emitted by
    ``encode(convert_to_padded_tensor=True)``. Returned in the input dtype (1.0 for real tokens, 0.0 for pad).

    Boundary: a real all-zero row is indistinguishable from padding here and is masked out of
    scoring. L2-normalized pipelines cannot produce one, but token pooling can (a cluster mean of
    antipodal vectors cancels to zero). Masking such a row scores it as 0 rather than dtype-min
    garbage, the preferable failure mode. Any future module that intentionally emits zero rows
    (e.g. learned pruning) must pass explicit masks instead of relying on this detection.
    """
    return padded.any(dim=-1).to(padded.dtype)


def _pad_multi_vector_inputs(
    inputs: list | np.ndarray | Tensor,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Pad a list of variable-length multi-vector tensors into a single 3D tensor with a mask.

    Returns the padded tensor and either the user-provided mask or one derived from the input. For a
    list input the mask comes from the per-row lengths. For a 3D tensor input without a mask the
    mask comes from all-zero rows (see :func:`_zero_row_mask`).
    """
    if isinstance(inputs, list):
        tensors = [_convert_to_tensor(t) for t in inputs]
        lengths = torch.tensor([t.shape[0] for t in tensors])
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
        if mask is None:
            max_len = padded.shape[1]
            mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).to(padded.device, dtype=padded.dtype)
        return padded, mask
    padded = _convert_to_tensor(inputs)
    if mask is None and padded.dim() == 3:
        mask = _zero_row_mask(padded)
    return padded, mask


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


# TODO: Should we fully separate dense and multi-vector similarity functions? They don't overlap
class SimilarityFunction(Enum):
    """
    Enum class for supported similarity functions. The following functions are supported:

    - ``SimilarityFunction.COSINE`` (``"cosine"``): Cosine similarity
    - ``SimilarityFunction.DOT_PRODUCT`` (``"dot"``, ``dot_product``): Dot product similarity
    - ``SimilarityFunction.EUCLIDEAN`` (``"euclidean"``): Euclidean distance
    - ``SimilarityFunction.MANHATTAN`` (``"manhattan"``): Manhattan distance
    - ``SimilarityFunction.MAXSIM`` (``"maxsim"``): Late-interaction MaxSim, used by
      :class:`~sentence_transformers.MultiVectorEncoder` (ColBERT-style) models.
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    DOT = "dot"  # Alias for DOT_PRODUCT
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MAXSIM = "maxsim"

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
        if similarity_function == SimilarityFunction.MAXSIM:
            return maxsim

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
        if similarity_function == SimilarityFunction.MAXSIM:
            return maxsim_pairwise

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
            ['cosine', 'dot', 'euclidean', 'manhattan', 'maxsim']
        """
        return [m.value for m in SimilarityFunction]
