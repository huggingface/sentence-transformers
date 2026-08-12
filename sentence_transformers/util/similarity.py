from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics import pairwise_distances
from torch import Tensor
from transformers.utils import is_torchdynamo_compiling, logging

from .tensor import (
    _convert_to_batch_tensor,
    _convert_to_float_tensor,
    _convert_to_tensor,
    normalize_embeddings,
    to_scipy_coo,
)

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = logging.get_logger(__name__)


def _match_layouts(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """
    Converts the dense tensor to sparse COO if exactly one of the two inputs is sparse, so that mixed inputs can
    reuse the all-sparse computations. The lone dense operand in such a mix typically still holds mostly zeros
    (e.g. sparse embeddings encoded with ``convert_to_sparse_tensor=False``), making the conversion cheap, whereas
    densifying the sparse operand would allocate a second full-size dense tensor.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
        tuple[Tensor, Tensor]: The two tensors, using the same layout.
    """
    if a.is_sparse and not b.is_sparse:
        return a, b.to_sparse()
    if b.is_sparse and not a.is_sparse:
        return a.to_sparse(), b
    return a, b


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


def pairwise_cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = _convert_to_float_tensor(a)
    b = _convert_to_float_tensor(b)

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


def pairwise_dot_score(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    a = _convert_to_float_tensor(a)
    b = _convert_to_float_tensor(b)

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

        a, b = _match_layouts(a, b)
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
    a = _convert_to_float_tensor(a)
    b = _convert_to_float_tensor(b)
    a, b = _match_layouts(a, b)

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

    if a.is_sparse or b.is_sparse:
        a, b = _match_layouts(a, b)
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
    a = _convert_to_float_tensor(a)
    b = _convert_to_float_tensor(b)
    a, b = _match_layouts(a, b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1)).to_dense()


# Element budget for one chunk's padded embeddings plus its scoring intermediate when the caller gives
# no explicit budget: 100M elements allocates at most ~400 MB (fp32, half in bf16).
_MAXSIM_CHUNK_ELEMENT_BUDGET = 100_000_000

# Ranks below any real MaxSim score, and unlike the float32 minimum it survives a loss scaling and squaring it.
_EMPTY_DOCUMENT_SCORE = -1e9


def _document_chunk_ranges(
    document_widths: list[int], per_document_token: int, element_budget: int, per_document_fixed: int = 0
) -> list[tuple[int, int]]:
    """Greedily pack documents into ``(start, end)`` chunks whose padded cost of
    ``(per_document_token * chunk_width + per_document_fixed) * chunk_size`` elements stays under
    ``element_budget``, with a floor of one document per chunk. The cost uses each chunk's local max
    width, matching the per-chunk padding in :func:`maxsim`, so one long outlier document only sizes
    its own chunk.

    ``per_document_token`` must count every allocation that scales with a document's padded width: the
    scoring intermediate and the padded embeddings themselves. Omitting the embeddings overshoots the
    budget several-fold whenever the query side is small, since the padding then dominates.
    ``per_document_fixed`` covers what grows with the chunk size but not with the document width, such
    as the padded queries in :func:`maxsim_pairwise`.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    width = 0
    for index, document_width in enumerate(document_widths):
        candidate_width = max(width, document_width)
        count = index - start + 1
        if count > 1 and (per_document_token * candidate_width + per_document_fixed) * count > element_budget:
            ranges.append((start, index))
            start = index
            width = document_width
        else:
            width = candidate_width
    ranges.append((start, len(document_widths)))
    return ranges


def _batch_singular_input(
    embeddings: list | np.ndarray | Tensor, mask: Tensor | None
) -> tuple[list | np.ndarray | Tensor, Tensor | None]:
    # Mirror the dense family's _convert_to_batch_tensor: a bare 2D (tokens, dim) input is one
    # multi-vector embedding (e.g. from a singular encode call), scored as a batch of one.
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        embeddings = np.expand_dims(embeddings, 0)
    elif isinstance(embeddings, Tensor) and embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
    else:
        return embeddings, mask
    if mask is not None and mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return embeddings, mask


def _canonicalize_side(
    embeddings: list | np.ndarray | Tensor,
    mask: Tensor | np.ndarray | list | None,
    name: str,
) -> tuple[list | np.ndarray | Tensor, Tensor | None]:
    """Shared input front door for one side of a scoring call: masks become a single tensor (a list
    of per-item masks is padded to the batch), and a bare 2D input batches as a single item."""
    if isinstance(mask, (list, tuple)):
        mask = torch.nn.utils.rnn.pad_sequence([_convert_to_tensor(m) for m in mask], batch_first=True)
    elif isinstance(mask, np.ndarray):
        mask = _convert_to_tensor(mask)
    elif mask is not None and not isinstance(mask, Tensor):
        raise TypeError(
            f"`{name}` must be a tensor, an ndarray, or a list of per-item masks, not {type(mask).__name__}."
        )
    return _batch_singular_input(embeddings, mask)


def _fit_mask_width(mask: Tensor, width: int, name: str, truncate: bool = False) -> Tensor:
    """Match a mask's token span to the padded embeddings. The chunk loops pass ``truncate=True``,
    since a batch-wide mask legitimately spans more than one chunk's local padding. Caller-facing
    inputs must match exactly: a mismatched mask is a construction bug, not a layout to guess at."""
    if mask.shape[-1] == width:
        return mask
    if truncate and mask.shape[-1] > width:
        return mask[..., :width]
    raise ValueError(f"`{name}` covers {mask.shape[-1]} tokens, but the padded embeddings have {width}.")


def _embeddings_device(embeddings: list | np.ndarray | Tensor) -> torch.device | None:
    """The device multi-vector embeddings live on, or None for numpy and empty inputs."""
    if isinstance(embeddings, Tensor):
        return embeddings.device
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], Tensor):
        return embeddings[0].device
    return None


def _to_device(embeddings: Tensor, mask: Tensor | None, device: torch.device) -> tuple[Tensor, Tensor | None]:
    """Move one side's padded embeddings and mask onto the scoring device. The mask is checked
    separately, since a caller can hand in a cpu mask alongside gpu embeddings and the masked_fill
    kernel rejects that."""
    if embeddings.device != device:
        embeddings = embeddings.to(device)
    if mask is not None and mask.device != device:
        mask = mask.to(device)
    return embeddings, mask


def _pad_chunk(
    embeddings: list | np.ndarray | Tensor,
    mask: Tensor | None,
    start: int,
    end: int,
    device: torch.device,
    name: str,
) -> tuple[Tensor, Tensor | None]:
    """Pad one chunk of a side onto the scoring device, with its mask narrowed to the chunk's local
    width. Padding the whole batch upfront instead would let one long outlier size the full
    ``(batch, max_len, dim)`` tensor: multi-GB of mostly padding on ragged input."""
    chunk_mask = mask[start:end] if mask is not None else None
    chunk, chunk_mask = _pad_multi_vector_inputs(embeddings[start:end], chunk_mask)
    chunk, chunk_mask = _to_device(chunk, chunk_mask, device)
    if chunk_mask is not None:
        # A caller-provided mask spans the global width, the chunk only its local width.
        chunk_mask = _fit_mask_width(chunk_mask, chunk.shape[1], name, truncate=True)
    return chunk, chunk_mask


def _embedding_dim(embeddings: list | np.ndarray | Tensor) -> int:
    """The trailing embedding dimension, read without materializing a padded batch."""
    if not isinstance(embeddings, list):
        return embeddings.shape[-1]
    return next((len(item[0]) for item in embeddings if len(item)), 1)


def _query_token_counts(a: list | np.ndarray | Tensor, a_mask: Tensor | None) -> Tensor:
    """Per-item count of real query tokens for MeanMaxSim scoring: the mask sum when a mask is
    given, the item lengths for ragged lists, and the non-zero rows for padded tensors."""
    if a_mask is not None:
        return a_mask.bool().sum(dim=-1).clamp(min=1)
    if isinstance(a, list):
        return torch.tensor([max(len(item), 1) for item in a])
    return _zero_row_mask(_convert_to_tensor(a)).bool().sum(dim=-1).clamp(min=1)


def maxsim(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
    document_chunk_elements: int | None = None,
    length_normalize: bool = False,
) -> Tensor:
    """
    Computes the MaxSim (late-interaction) score between two collections of multi-vector embeddings.

    For each query in ``a`` and document in ``b``, the score is the sum over query tokens of the maximum
    similarity to any document token: ``sum_i max_j (a_i . b_j)``. This is the scoring function used by
    ColBERT-style models.

    Args:
        a (Union[list, np.ndarray, Tensor]): Query embeddings. Either a 3D tensor of shape
            ``(batch_a, num_query_tokens, embedding_dim)`` (pre-padded), a list of 2D tensors of shape
            ``(num_query_tokens_i, embedding_dim)`` (variable-length, padded internally), or a single
            2D tensor or array (one query, e.g. from a singular encode call, scored as a batch of one).
        b (Union[list, np.ndarray, Tensor]): Document embeddings. Either a 3D tensor of shape
            ``(batch_b, num_doc_tokens, embedding_dim)``, a list of 2D tensors of shape
            ``(num_doc_tokens_i, embedding_dim)``, or a single 2D tensor or array (one document,
            scored as a batch of one).
        a_mask (Tensor, optional): Boolean or float mask for query tokens, shape ``(batch_a, num_query_tokens)``,
            ``(num_query_tokens,)`` alongside a single 2D ``a``, or a list of per-item 1D masks (padded to
            the batch). Tokens with a 0 / False entry are excluded from the sum. Defaults to None (use all tokens).
        b_mask (Tensor, optional): Boolean or float mask for document tokens, shape ``(batch_b, num_doc_tokens)``,
            ``(num_doc_tokens,)`` alongside a single 2D ``b``, or a list of per-item 1D masks. Tokens with a
            0 / False entry are excluded from the max. Defaults to None (use all tokens).
        document_chunk_elements (int, optional): Element budget for the padded ``(chunk, d_tokens, dim)``
            documents plus the 4D ``(batch_a, chunk, q_tokens, d_tokens)`` scoring intermediate. Documents
            are greedily packed into chunks (padded per chunk, so a long outlier only sizes its own chunk)
            that stay under the budget, with a floor of one document per chunk. Defaults to None, which
            applies a 100M-element budget (at most ~400 MB, half that in bf16 / fp16). With very large
            query batches that floor can still be big: shard queries externally if a single document
            exceeds memory.
        length_normalize (bool, optional): Divide each score by the number of real (unmasked) query
            tokens, yielding MeanMaxSim: scores land in the per-token similarity range (about
            ``[-1, 1]`` for normalized embeddings) independent of the query length. Defaults to False.

    Returns:
        Tensor: Matrix with ``res[i][j]`` = MaxSim(a[i], b[j]), shape ``(batch_a, batch_b)``, always
        float32 (half precision inputs are accumulated in float32 to keep nearby scores distinct), on
        the documents' device: CPU documents score on the CPU even against GPU queries.
    """
    a, a_mask = _canonicalize_side(a, a_mask, "a_mask")
    b, b_mask = _canonicalize_side(b, b_mask, "b_mask")
    if len(a) == 0 or len(b) == 0:
        # An empty query or document set: nothing to score, and pad_sequence rejects an empty list.
        device = _embeddings_device(b) or _embeddings_device(a)
        return torch.zeros(len(a), len(b), dtype=torch.float32, device=device)
    query_token_counts = _query_token_counts(a, a_mask) if length_normalize else None
    a, a_mask_padded = _pad_multi_vector_inputs(a, a_mask)
    if a_mask_padded is not None:
        a_mask_padded = _fit_mask_width(a_mask_padded, a.shape[1], "a_mask")

    num_documents = len(b)
    budget = document_chunk_elements if document_chunk_elements is not None else _MAXSIM_CHUNK_ELEMENT_BUDGET
    if isinstance(b, list):
        document_widths = [len(document) for document in b]
    else:
        document_widths = [b.shape[1]] * num_documents
    if b_mask is not None:
        b_mask = _fit_mask_width(b_mask, max(document_widths), "b_mask")

    # Scoring runs on the documents' device: they are the big side, and the queries are cheap to move.
    # numpy documents carry no device of their own, so those follow the queries.
    device = _embeddings_device(b) or a.device
    a, a_mask_padded = _to_device(a, a_mask_padded, device)

    # The budget covers the padded documents (width x dim) as well as the (batch_a, q_tokens, width)
    # score intermediate: with a small query side the padding, not the score matrix, is the bigger half.
    ranges = _document_chunk_ranges(document_widths, a.shape[0] * a.shape[1] + _embedding_dim(b), budget)

    score_chunks = []
    for d_start, d_end in ranges:
        chunk_b, chunk_b_mask = _pad_chunk(b, b_mask, d_start, d_end, device, "b_mask")
        score_chunks.append(_maxsim_score_documents(a, chunk_b, a_mask_padded, chunk_b_mask))
    # cat copies even a single chunk, which is the common case of a batch that fits the budget.
    scores = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks, dim=1)
    if query_token_counts is not None:
        scores = scores / query_token_counts.to(device=scores.device, dtype=scores.dtype).unsqueeze(1)
    return scores


def _maxsim_score_documents(a: Tensor, b: Tensor, a_mask: Tensor | None, b_mask: Tensor | None) -> Tensor:
    """Score one chunk of documents against every query, returned as ``(a_batch, b_batch)`` float32.
    Pulled out of :func:`maxsim` so the chunk loop stays readable. Reducing the chunk all the way down
    to final scores here is what makes the chunking pay off: concatenating the
    ``(a_batch, b_batch, q_tokens)`` reductions first would rebuild a full-corpus-width intermediate.

    The float32 cast ahead of the query-token sum keeps that accumulation in float32: MaxSim scores
    reach O(num_query_tokens) magnitudes where the bf16 grid is 0.125 wide, coarse enough to collapse
    thousands of documents onto a handful of distinct scores.
    """
    if b.shape[1] == 0:
        # Every document is zero-width, nothing to take the max over.
        empty = torch.ones(a.shape[0], b.shape[0], dtype=torch.bool, device=b.device)
        return _fill_empty_document_scores(
            torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32, device=b.device), empty
        )
    scores = torch.einsum("ash,bth->abst", a, b)
    # Documents reduced with max, so masked (padding) tokens are sent to the dtype minimum: multiplying
    # by 0 would let a padding token win the max over a genuinely negative similarity, and a finite fill
    # (e.g. _EMPTY_DOCUMENT_SCORE, a final-score sentinel) could still be undercut by unnormalized real
    # scores and win the max the same way. Queries reduced with sum (mask to 0). Filled in place: a copy
    # here doubles the largest live tensor.
    # Note: PyLate's colbert_scores masks by multiplying by 0. We exclude masked tokens from the max
    # instead (matching xtr_scores). Parity tests pass against PyLate because real tokens dominate the
    # max in practice, but the dtype-min approach is the more correct one and the one we keep.
    if b_mask is not None:
        scores.masked_fill_(~b_mask.bool().unsqueeze(0).unsqueeze(2), torch.finfo(scores.dtype).min)
    # Query-token maxima in float32, so the sum does not accumulate on the coarse bf16 grid.
    scores = scores.max(dim=-1).values.float()
    if a_mask is not None:
        scores = scores * a_mask.unsqueeze(1)
    scores = scores.sum(dim=-1)
    if b_mask is not None:
        scores = _fill_empty_document_scores(scores, ~b_mask.bool().any(dim=-1))
    return scores


def _fill_empty_document_scores(scores: Tensor, empty: Tensor) -> Tensor:
    """Overwrite the scores of documents flagged in ``empty`` (no unmasked token), whose every token
    sat at the masked-fill dtype minimum and whose sum therefore poisons downstream losses.

    The fill is unconditional: reading ``empty`` on the host costs a device synchronization per
    scoring call and is a data-dependent branch that :func:`torch.compile` cannot capture, so only
    the warning reads it, and only where that is free.
    """
    # transformers' shim over torch.compiler.is_compiling(), which torch only added in 2.3.
    if not is_torchdynamo_compiling() and empty.device.type == "cpu" and empty.any():
        logger.warning_once(
            "Encountered documents without any scoreable token during multi-vector similarity scoring, "
            "e.g. because MultiVectorMask (via keep_only_token_ids or skiplist_words) masked out every "
            "token, or because a document embedding has zero token vectors. These documents score "
            f"about {_EMPTY_DOCUMENT_SCORE:.0e} (before any normalization), ranking them below every "
            "real document."
        )
    return scores.masked_fill(empty, _EMPTY_DOCUMENT_SCORE)


def maxsim_pairwise(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
    pair_chunk_elements: int | None = None,
    length_normalize: bool = False,
) -> Tensor:
    """
    Computes the pairwise MaxSim (late-interaction) score between each query-document pair.

    For each ``i``, computes the MaxSim score between ``a[i]`` and ``b[i]``: the sum over query tokens of the
    maximum similarity to any document token. This is the pairwise analogue of :func:`maxsim`.

    Args:
        a (Union[list, np.ndarray, Tensor]): Query embeddings. Either a 3D tensor of shape
            ``(batch, num_query_tokens, embedding_dim)`` (pre-padded), a list of 2D tensors with
            shape ``(num_query_tokens_i, embedding_dim)``, or a single 2D tensor or array (one query,
            e.g. from a singular encode call, scored as a batch of one).
        b (Union[list, np.ndarray, Tensor]): Document embeddings. Either a 3D tensor of shape
            ``(batch, num_doc_tokens, embedding_dim)``, a list of 2D tensors with shape
            ``(num_doc_tokens_i, embedding_dim)``, or a single 2D tensor or array (one document,
            scored as a batch of one).
        a_mask (Tensor, optional): Boolean or float mask for query tokens, shape ``(batch, num_query_tokens)``,
            ``(num_query_tokens,)`` alongside a single 2D ``a``, or a list of per-item 1D masks (padded to
            the batch). Tokens with a 0 / False entry are excluded from the sum. Defaults to None (use all tokens).
        b_mask (Tensor, optional): Boolean or float mask for document tokens, shape ``(batch, num_doc_tokens)``,
            ``(num_doc_tokens,)`` alongside a single 2D ``b``, or a list of per-item 1D masks. Tokens with a
            0 / False entry are excluded from the max. Defaults to None (use all tokens).
        pair_chunk_elements (int, optional): Element budget for the padded ``(chunk, q_tokens, dim)``
            queries and ``(chunk, d_tokens, dim)`` documents plus the ``(chunk, q_tokens, d_tokens)``
            scoring intermediate. Pairs are greedily packed into chunks (padded per chunk, so a long
            outlier only sizes its own chunk) that stay under the budget, with a floor of one pair per
            chunk. Defaults to None, which applies a 100M-element budget (at most ~400 MB, half that
            in bf16 / fp16).
        length_normalize (bool, optional): Divide each score by the number of real (unmasked) query
            tokens, yielding MeanMaxSim: scores land in the per-token similarity range (about
            ``[-1, 1]`` for normalized embeddings) independent of the query length. Defaults to False.

    Returns:
        Tensor: Vector with ``res[i]`` = MaxSim(a[i], b[i]), shape ``(batch,)``, always float32
        (half precision inputs are accumulated in float32 to keep nearby scores distinct), on the
        documents' device: CPU documents score on the CPU even against GPU queries.
    """
    a, a_mask = _canonicalize_side(a, a_mask, "a_mask")
    b, b_mask = _canonicalize_side(b, b_mask, "b_mask")
    if len(a) != len(b):
        raise ValueError(
            f"`a` and `b` must hold the same number of items to score as pairs, got {len(a)} and {len(b)}."
        )
    if len(a) == 0:
        # No pairs to score, and pad_sequence rejects an empty list.
        device = _embeddings_device(b) or _embeddings_device(a)
        return torch.zeros(0, dtype=torch.float32, device=device)
    query_token_counts = _query_token_counts(a, a_mask) if length_normalize else None

    budget = pair_chunk_elements if pair_chunk_elements is not None else _MAXSIM_CHUNK_ELEMENT_BUDGET
    query_widths = [len(query) for query in a] if isinstance(a, list) else [a.shape[1]] * len(a)
    document_widths = [len(document) for document in b] if isinstance(b, list) else [b.shape[1]] * len(b)
    if a_mask is not None:
        a_mask = _fit_mask_width(a_mask, max(query_widths), "a_mask")
    if b_mask is not None:
        b_mask = _fit_mask_width(b_mask, max(document_widths), "b_mask")
    # Scoring runs on the documents' device, or the queries' when the documents carry none (numpy).
    device = _embeddings_device(b) or _embeddings_device(a) or torch.device("cpu")
    # The budget covers the padded queries and documents (width x dim each) as well as the
    # (q_tokens x width) score intermediate: with a single query per pair the padding, not the score
    # matrix, is the bigger half. The query padding grows with the chunk size but not with the
    # document width, so it goes in as a fixed per-pair cost.
    query_width = max(query_widths)
    dim = _embedding_dim(b)
    ranges = _document_chunk_ranges(document_widths, query_width + dim, budget, per_document_fixed=query_width * dim)

    score_chunks = []
    for start, end in ranges:
        chunk_a, chunk_a_mask = _pad_chunk(a, a_mask, start, end, device, "a_mask")
        chunk_b, chunk_b_mask = _pad_chunk(b, b_mask, start, end, device, "b_mask")
        score_chunks.append(_maxsim_score_pairs(chunk_a, chunk_b, chunk_a_mask, chunk_b_mask))
    scores = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks, dim=0)
    if query_token_counts is not None:
        scores = scores / query_token_counts.to(device=scores.device, dtype=scores.dtype)
    return scores


def mean_maxsim(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
    document_chunk_elements: int | None = None,
) -> Tensor:
    """
    Computes the MeanMaxSim score between two collections of multi-vector embeddings: :func:`maxsim`
    divided by each query's real token count, with :func:`maxsim`'s arguments and return shape.

    MaxSim sums one similarity per query token, so its scale grows with query length and scores are
    not comparable across queries. MeanMaxSim averages instead, landing in the per-token similarity
    range (about ``[-1, 1]`` for normalized embeddings). Rankings within a query are unchanged, since
    the divisor is constant per row.
    """
    return maxsim(
        a,
        b,
        a_mask=a_mask,
        b_mask=b_mask,
        document_chunk_elements=document_chunk_elements,
        length_normalize=True,
    )


def mean_maxsim_pairwise(
    a: list | np.ndarray | Tensor,
    b: list | np.ndarray | Tensor,
    a_mask: Tensor | None = None,
    b_mask: Tensor | None = None,
    pair_chunk_elements: int | None = None,
) -> Tensor:
    """
    Computes the pairwise MeanMaxSim score for each query-document pair: :func:`maxsim_pairwise`
    divided by each query's real token count, with its arguments and return shape. See
    :func:`mean_maxsim` for why.
    """
    return maxsim_pairwise(
        a,
        b,
        a_mask=a_mask,
        b_mask=b_mask,
        pair_chunk_elements=pair_chunk_elements,
        length_normalize=True,
    )


def _maxsim_score_pairs(a: Tensor, b: Tensor, a_mask: Tensor | None, b_mask: Tensor | None) -> Tensor:
    """Score one chunk of aligned query-document pairs, returned as ``(batch,)`` float32. Pulled out of
    :func:`maxsim_pairwise` so the chunk loop stays readable. The mirror of
    :func:`_maxsim_score_documents`, differing only in the einsum and the mask broadcast axes."""
    if b.shape[1] == 0:
        # Every document is zero-width, nothing to take the max over.
        empty = torch.ones(b.shape[0], dtype=torch.bool, device=b.device)
        return _fill_empty_document_scores(torch.zeros(b.shape[0], dtype=torch.float32, device=b.device), empty)
    scores = torch.einsum("bsh,bth->bst", a, b)
    # Documents reduced with max (mask to dtype min so padding never wins, see
    # :func:`_maxsim_score_documents`). Queries reduced with sum (mask to 0). Filled in place: a copy
    # here doubles the largest live tensor.
    if b_mask is not None:
        scores.masked_fill_(~b_mask.bool().unsqueeze(1), torch.finfo(scores.dtype).min)
    # Query-token maxima in float32, so the sum does not accumulate on the coarse bf16 grid.
    scores = scores.max(dim=-1).values.float()
    if a_mask is not None:
        scores = scores * a_mask
    scores = scores.sum(dim=-1)
    if b_mask is not None:
        scores = _fill_empty_document_scores(scores, ~b_mask.bool().any(dim=-1))
    return scores


def _zero_row_mask(padded: Tensor) -> Tensor:
    """Derive a ``(B, T)`` mask from a pre-padded ``(B, T, D)`` token-embedding tensor by treating
    all-zero rows as padding.

    Real token embeddings normally carry information in at least one dim, so a fully-zero row is the
    all-but-impossible-except-by-pad signal. This matches the zero padding of
    :func:`torch.nn.utils.rnn.pad_sequence`. Returned in the input dtype (1.0 for real tokens, 0.0 for pad).

    Boundary: a real all-zero row is indistinguishable from padding here and is masked out of
    scoring. L2-normalized pipelines cannot produce one, but token pooling can (a cluster mean of
    antipodal vectors cancels to zero). Masking such a row scores it as 0 rather than dtype-min
    garbage, the preferable failure mode. A document whose rows are *all* zero instead reads as empty
    and takes the :func:`_fill_empty_document_scores` sentinel. Any future module that intentionally
    emits zero rows (e.g. learned pruning) must pass explicit masks instead of relying on this detection.
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
        # Quantized (integer) embeddings are upcast so einsum and the finfo-based masking work.
        tensors = [_convert_to_tensor(t) for t in inputs]
        tensors = [t if t.is_floating_point() else t.float() for t in tensors]
        lengths = torch.tensor([t.shape[0] for t in tensors])
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
        if mask is None:
            max_len = padded.shape[1]
            mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).to(padded.device, dtype=padded.dtype)
        return padded, mask
    padded = _convert_to_tensor(inputs)
    if not padded.is_floating_point():
        padded = padded.float()
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
    if x.is_sparse or y.is_sparse:
        logger.warning_once("Pairwise angle similarity does not support sparse tensors. Converting to dense.")
        if x.is_sparse:
            x = x.to_dense()
        if y.is_sparse:
            y = y.to_dense()

    x = _convert_to_float_tensor(x)
    y = _convert_to_float_tensor(y)

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
    - ``SimilarityFunction.MAXSIM`` (``"maxsim"``): Late-interaction MaxSim, used by
      :class:`~sentence_transformers.MultiVectorEncoder` (ColBERT-style) models.
    - ``SimilarityFunction.MEAN_MAXSIM`` (``"meanmaxsim"``): MaxSim divided by the query's token
      count, so scores are comparable across query lengths. Same rankings within a query.
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    DOT = "dot"  # Alias for DOT_PRODUCT
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MAXSIM = "maxsim"
    MEAN_MAXSIM = "meanmaxsim"

    @staticmethod
    def to_similarity_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[list | ndarray | Tensor, list | ndarray | Tensor], Tensor]:
        """
        Converts a similarity function name or enum value to the corresponding similarity function.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[list, ndarray, Tensor], Union[list, ndarray, Tensor]], Tensor]: The corresponding similarity function.

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
        if similarity_function == SimilarityFunction.MEAN_MAXSIM:
            return mean_maxsim

        raise ValueError(
            f"The provided function {similarity_function} is not supported. Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[list | ndarray | Tensor, list | ndarray | Tensor], Tensor]:
        """
        Converts a similarity function into a pairwise similarity function.

        The pairwise similarity function returns the diagonal vector from the similarity matrix, i.e. it only
        computes the similarity(a[i], b[i]) for each i in the range of the input tensors, rather than
        computing the similarity between all pairs of a and b.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[list, ndarray, Tensor], Union[list, ndarray, Tensor]], Tensor]: The pairwise similarity function.

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
        if similarity_function == SimilarityFunction.MEAN_MAXSIM:
            return mean_maxsim_pairwise

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
            ['cosine', 'dot', 'euclidean', 'manhattan', 'maxsim', 'meanmaxsim']
        """
        return [m.value for m in SimilarityFunction]
