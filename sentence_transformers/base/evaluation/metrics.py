from __future__ import annotations

import numpy as np


def tie_aware_reciprocal_rank(is_relevant, scores, at_k: int) -> float:
    """
    Computes the reciprocal rank of the first relevant candidate, averaged over the orderings that
    equally scored candidates allow.

    Candidates that share a score may be ordered in any way, so reading the rank off a single argsort
    makes the result depend on the order in which the candidates were passed in. Averaging over the
    orderings of the tied group holding the first relevant candidate removes that dependency, following
    McSherry and Najork (2008), section 2.5. Without tied scores this is ``1 / rank`` of the first
    relevant candidate, i.e. the usual reciprocal rank.

    Args:
        is_relevant: Binary relevance for each candidate.
        scores: Predicted score for each candidate.
        at_k (int): Only the first ``at_k`` positions are scored.

    Returns:
        float: The reciprocal rank, or 0.0 if no relevant candidate can reach the first ``at_k`` positions.
    """
    is_relevant = np.asarray(is_relevant, dtype=bool)
    scores = np.asarray(scores)
    if at_k < 1 or not is_relevant.any():
        return 0.0

    order = np.argsort(-scores, kind="stable")
    scores, is_relevant = scores[order], is_relevant[order]

    group_start = 0
    while group_start < len(scores):
        group_stop = group_start
        while group_stop < len(scores) and scores[group_stop] == scores[group_start]:
            group_stop += 1

        num_relevant = int(is_relevant[group_start:group_stop].sum())
        if num_relevant:
            group_size = group_stop - group_start
            # `all_irrelevant` tracks the fraction of orderings whose first `position - 1` candidates in
            # this group are all irrelevant; the drop between consecutive positions is the fraction that
            # places the first relevant candidate at exactly that position.
            reciprocal_rank = 0.0
            all_irrelevant = 1.0
            for position in range(1, group_size + 1):
                rank = group_start + position
                if rank > at_k:
                    break
                remaining = all_irrelevant * (1 - num_relevant / (group_size - position + 1))
                reciprocal_rank += (all_irrelevant - remaining) / rank
                all_irrelevant = remaining
            return reciprocal_rank

        group_start = group_stop

    return 0.0
