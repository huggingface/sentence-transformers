from __future__ import annotations

import numpy as np


def reciprocal_rank(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray, k: int) -> float:
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
