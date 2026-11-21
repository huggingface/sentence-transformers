from __future__ import annotations

import numpy as np
from sklearn.metrics import dcg_score


def _reciprocal_rank(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray, k: int) -> float:
    """Compute reciprocal rank at k for one query, averaged over all orderings of tied scores.

    Args:
        y_true: Binary relevance labels for the documents.
        y_score: Predicted scores for the documents. Higher scores rank first.
        k: Only consider the top k documents.

    Returns:
        The expected reciprocal rank, or 0.0 if there are no relevant documents in the top k.
    """
    scores = np.asarray(y_score)
    positive_scores = scores[np.asarray(y_true, dtype=bool)]
    if positive_scores.size == 0:
        return 0.0

    best_positive_score = positive_scores.max()
    first_rank = 1 + np.count_nonzero(scores > best_positive_score)
    tied_positives = np.count_nonzero(positive_scores == best_positive_score)
    tied_count = np.count_nonzero(scores == best_positive_score)

    rr_score = 0.0
    no_positive_yet = 1.0
    for offset in range(min(tied_count - tied_positives + 1, k - first_rank + 1)):
        positive_probability = tied_positives / (tied_count - offset)
        rr_score += no_positive_yet * positive_probability / (first_rank + offset)
        no_positive_yet *= 1 - positive_probability
    return rr_score


def _average_precision(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    """Compute average precision for one query, averaged over all orderings of tied scores.

    Without ties this matches :func:`sklearn.metrics.average_precision_score`. With ties, sklearn credits
    every relevant document in a tied group with the precision at the end of the group, which can be
    lower or higher than the precision of any actual ordering.

    Args:
        y_true: Binary relevance labels for the documents.
        y_score: Predicted scores for the documents. Higher scores rank first.

    Returns:
        The expected average precision, or 0.0 if there are no relevant documents.

    Raises:
        ValueError: If ``y_score`` contains NaN, like :func:`sklearn.metrics.average_precision_score`.
    """
    scores = np.asarray(y_score, dtype=float)
    # np.unique sorts NaN last, so without this check a NaN score would be ranked first
    if np.isnan(scores).any():
        raise ValueError("Input contains NaN.")
    relevant = np.asarray(y_true, dtype=bool)
    num_relevant = np.count_nonzero(relevant)
    if num_relevant == 0:
        return 0.0

    # Walk the groups of tied scores from best to worst. Within a group of `size` documents holding
    # `positives` relevant ones, each position is relevant with probability positives / size, and given
    # that, (positives - 1) / (size - 1) of the positions before it are relevant on average.
    _, group_ids = np.unique(scores, return_inverse=True)
    group_ids = group_ids.ravel()
    group_sizes = np.bincount(group_ids)[::-1]
    group_positives = np.bincount(group_ids, weights=relevant)[::-1]

    precision_sum = 0.0
    ranked = 0
    hits = 0.0
    for size, positives in zip(group_sizes, group_positives):
        if positives:
            positions = np.arange(1, size + 1)
            earlier_hits = (positions - 1) * (positives - 1) / (size - 1) if size > 1 else 0.0
            precision_sum += positives / size * np.sum((hits + 1 + earlier_hits) / (ranked + positions))
        ranked += size
        hits += positives
    return float(precision_sum / num_relevant)


def _binary_ndcg(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    k: int,
    num_positives: int | None = None,
) -> float:
    """Compute NDCG at k for one query with binary relevance, averaging over tied scores.

    Like :func:`sklearn.metrics.ndcg_score` for binary relevance, but ``num_positives`` can account for
    relevant documents that were not retrieved. These contribute to the ideal DCG without needing
    entries in ``y_true`` or ``y_score``.

    Args:
        y_true: Binary relevance labels for the retrieved documents.
        y_score: Predicted scores for the retrieved documents. Higher scores rank first.
        k: Only consider the top k documents.
        num_positives: Total number of known relevant documents, including those not retrieved.
            Defaults to the number of positives in y_true.

    Returns:
        NDCG at k, or 0.0 if no relevant documents are retrieved or k is 0.
    """
    num_retrieved_positives = sum(y_true)
    if num_retrieved_positives == 0 or k == 0:
        return 0.0
    num_positives = max(num_positives or 0, num_retrieved_positives)
    dcg = dcg_score([y_true], [y_score], k=k) if len(y_true) > 1 else y_true[0]
    ideal_dcg = np.sum(1 / np.log2(np.arange(2, min(num_positives, k) + 2)))
    return dcg / ideal_dcg
