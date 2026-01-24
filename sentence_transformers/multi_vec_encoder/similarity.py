from __future__ import annotations

import torch
from torch import Tensor


def maxsim(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_mask: Tensor | None = None,
    document_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute MaxSim similarity between queries and documents.

    MaxSim computes, for each query token, the maximum cosine similarity to any document token,
    then sums these maximum similarities across all query tokens:

        score(q, d) = Σᵢ maxⱼ(qᵢ · dⱼ)

    where qᵢ are query token embeddings and dⱼ are document token embeddings.

    This is the core similarity function used in late interaction models.

    Args:
        query_embeddings: Query token embeddings of shape [batch_q, num_query_tokens, dim].
            Should be L2-normalized for cosine similarity.
        document_embeddings: Document token embeddings of shape [batch_d, num_doc_tokens, dim].
            Should be L2-normalized for cosine similarity.
        query_mask: Optional attention mask for queries of shape [batch_q, num_query_tokens].
            1 for valid tokens, 0 for padding. If None, all tokens are considered valid.
        document_mask: Optional attention mask for documents of shape [batch_d, num_doc_tokens].
            1 for valid tokens, 0 for padding. If None, all tokens are considered valid.

    Returns:
        Similarity scores of shape [batch_q, batch_d].
    """
    batch_q, num_query_tokens, dim = query_embeddings.shape
    batch_d, num_doc_tokens, _ = document_embeddings.shape

    # Compute token-level similarity: [batch_q, batch_d, num_query_tokens, num_doc_tokens]
    token_similarities = torch.einsum("aik,bjk->abij", query_embeddings, document_embeddings)
    # Reshape to [batch_q, num_query_tokens, batch_d, num_doc_tokens]
    token_similarities = token_similarities.permute(0, 2, 1, 3)

    # Apply document mask: set similarities with padding tokens to -inf
    if document_mask is not None:
        doc_mask_expanded = document_mask.unsqueeze(0).unsqueeze(0)
        token_similarities = token_similarities.masked_fill(doc_mask_expanded == 0, float("-inf"))

    # For each query token, find max similarity to any document token
    max_similarities = token_similarities.max(dim=-1).values

    # Apply query mask: set masked query tokens to 0 before summing
    if query_mask is not None:
        query_mask_expanded = query_mask.unsqueeze(-1)
        max_similarities = max_similarities * query_mask_expanded

    # Sum over query tokens
    scores = max_similarities.sum(dim=1)

    return scores


def maxsim_pairwise(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_mask: Tensor | None = None,
    document_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute pairwise MaxSim similarity between corresponding query-document pairs.

    Unlike :func:`maxsim` which computes all pairs, this function computes similarity
    only for corresponding pairs: score[i] = maxsim(query[i], document[i]).

    Args:
        query_embeddings: Query token embeddings of shape [batch, num_query_tokens, dim].
        document_embeddings: Document token embeddings of shape [batch, num_doc_tokens, dim].
        query_mask: Optional attention mask for queries of shape [batch, num_query_tokens].
        document_mask: Optional attention mask for documents of shape [batch, num_doc_tokens].

    Returns:
        Similarity scores of shape [batch].
    """
    batch_size, num_query_tokens, dim = query_embeddings.shape
    batch_d, num_doc_tokens, _ = document_embeddings.shape

    assert batch_size == batch_d, (
        f"Batch sizes must match for pairwise computation. "
        f"Got query batch size {batch_size} and document batch size {batch_d}. "
        f"Fallback to maxsim() for computing all query-document pairs."
    )

    # Compute token-level similarity for each pair: [batch, num_query_tokens, num_doc_tokens]
    token_similarities = torch.bmm(query_embeddings, document_embeddings.transpose(1, 2))

    # Apply document mask: set similarities with padding tokens to -inf
    if document_mask is not None:
        doc_mask_expanded = document_mask.unsqueeze(1)
        token_similarities = token_similarities.masked_fill(doc_mask_expanded == 0, float("-inf"))

    # For each query token, find max similarity to any document token
    max_similarities = token_similarities.max(dim=-1).values

    # Apply query mask: set masked query tokens to 0 before summing
    if query_mask is not None:
        max_similarities = max_similarities * query_mask

    # Sum over query tokens
    scores = max_similarities.sum(dim=1)

    return scores
