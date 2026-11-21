from __future__ import annotations

from itertools import permutations
from math import log2

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, ndcg_score

from sentence_transformers.util.metrics import _average_precision, _binary_ndcg, _reciprocal_rank


def _brute_force_average_precision(labels: list[int], scores: list[float]) -> float:
    """Average precision averaged over every ordering consistent with the scores, by enumeration."""
    values = []
    for order in permutations(range(len(labels))):
        if any(scores[a] < scores[b] for a, b in zip(order, order[1:])):
            continue
        hits = 0
        precision_sum = 0.0
        for rank, index in enumerate(order, start=1):
            if labels[index]:
                hits += 1
                precision_sum += hits / rank
        values.append(precision_sum / sum(labels))
    return float(np.mean(values))


TIED_CASES = [
    pytest.param([1, 0, 0, 0], [1.0] * 4, id="single-positive-all-tied"),
    pytest.param([1, 1, 0, 0], [1.0] * 4, id="two-positives-all-tied"),
    pytest.param([0, 1, 0, 0], [2.0, 1.0, 1.0, 1.0], id="positive-in-tie-after-negative"),
    pytest.param([1, 0, 1, 0], [3.0, 2.0, 2.0, 1.0], id="positive-in-middle-tie"),
    pytest.param([0, 1, 1, 0, 1, 0], [2.0, 2.0, 2.0, 1.0, 1.0, 1.0], id="two-tied-groups"),
    pytest.param([1, 1], [1.0, 1.0], id="all-positives"),
]


@pytest.mark.parametrize(("labels", "scores"), TIED_CASES)
def test_average_precision_averages_over_tied_orderings(labels, scores):
    expected = _brute_force_average_precision(labels, scores)
    assert _average_precision(labels, scores) == pytest.approx(expected)
    assert _average_precision(np.asarray(labels), np.asarray(scores)) == pytest.approx(expected)


@pytest.mark.parametrize(("labels", "scores"), TIED_CASES)
def test_average_precision_equals_reciprocal_rank_for_one_positive(labels, scores):
    """With a single relevant document, AP is its reciprocal rank, ties included."""
    labels = [int(index == labels.index(1)) for index in range(len(labels))]
    assert _average_precision(labels, scores) == pytest.approx(_reciprocal_rank(labels, scores, k=len(labels)))


@pytest.mark.parametrize(
    ("labels", "scores"),
    [([0, 1, 0], [0.9, 0.8, 0.7]), ([1, 0, 1, 0, 1], [0.5, 0.4, 0.3, 0.2, 0.1]), ([0, 1, 1, 0], [0.1, 0.9, 0.3, 0.7])],
)
def test_average_precision_matches_sklearn_without_ties(labels, scores):
    assert _average_precision(labels, scores) == pytest.approx(average_precision_score(labels, scores))


def test_average_precision_without_positives():
    assert _average_precision([0, 0], [1.0, 0.0]) == 0.0


@pytest.mark.parametrize("labels", [[1, 0, 0], [0, 0, 0]])
def test_average_precision_rejects_nan(labels):
    """NaN scores from a broken model must not be ranked first and reported as a perfect MAP."""
    with pytest.raises(ValueError, match="NaN"):
        _average_precision(labels, [float("nan"), 0.5, 0.1])


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize(
    ("labels", "scores", "k", "expected"),
    [
        pytest.param(
            [1] + [0] * 9, [1.0] * 10, 10, sum(1 / rank for rank in range(1, 11)) / 10, id="tie-ends-at-cutoff"
        ),
        pytest.param([1, 0, 0, 0, 0], [1.0] * 5, 1, 1 / 5, id="top-one"),
        pytest.param([0, 1, 0, 1], [1.0] * 4, 10, 13 / 18, id="interleaved-positives"),
        pytest.param([0, 1, 0, 1], [1.0] * 4, 1, 1 / 2, id="cutoff-inside-tie"),
        pytest.param([1, 1, 0], [1.0, 1.0, 0.0], 10, 1.0, id="only-positives-tied"),
        pytest.param([1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 2, 1, 1, 0], 2, 1 / 4, id="mixed-scores-cutoff"),
        pytest.param([1, 0, 0, 0], [1.0, 2.0, 2.0, 1.0], 2, 0.0, id="tie-below-cutoff"),
        pytest.param([0, 1, 0], [0.9, 0.8, 0.7], 10, 1 / 2, id="distinct-scores"),
        pytest.param([0, 1, 0], [0.9, 0.8, 0.7], 1, 0.0, id="distinct-scores-cutoff"),
        pytest.param([1, 1, 0], [0.0, 2.0, 1.0], 10, 1.0, id="best-positive-not-first"),
        pytest.param([0, 0], [1.0, 1.0], 10, 0.0, id="no-positives"),
        pytest.param([1, 1], [1.0, 1.0], 10, 1.0, id="all-positives"),
    ],
)
def test_reciprocal_rank(as_array, labels, scores, k, expected):
    if as_array:
        labels, scores = np.asarray(labels), np.asarray(scores)
    assert _reciprocal_rank(labels, scores, k=k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [1, 2, 10])
@pytest.mark.parametrize(
    ("labels", "scores"),
    [([0, 1, 0], [0.9, 0.8, 0.7]), ([0, 1, 0, 1], [0.9, 0.8, 0.8, 0.7]), ([0, 1, 0, 1], [1.0] * 4)],
)
def test_binary_ndcg_matches_sklearn(labels, scores, k):
    expected = ndcg_score([labels], [scores], k=k)
    assert _binary_ndcg(np.asarray(labels), np.asarray(scores), k=k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("labels", "scores", "k", "num_positives", "expected"),
    [
        pytest.param([], [], 10, 2, 0.0, id="empty"),
        pytest.param([0, 0], [1.0, 0.0], 10, 2, 0.0, id="no-retrieved-positives"),
        pytest.param([1], [1.0], 10, None, 1.0, id="single-document"),
        pytest.param([1], [1.0], 10, 2, 1 / (1 + 1 / log2(3)), id="missing-positive"),
        pytest.param([1, 0], [0.0, 0.0], 1, 2, 0.5, id="missing-positive-with-ties"),
        pytest.param([1, 1], [1.0, 0.0], 10, 1, 1.0, id="underreported-positives"),
        pytest.param([1], [1.0], 0, 2, 0.0, id="zero-cutoff-single-document"),
        pytest.param([1, 0], [0.0, 0.0], 0, None, 0.0, id="zero-cutoff-ties"),
    ],
)
def test_binary_ndcg(labels, scores, k, num_positives, expected):
    assert _binary_ndcg(labels, scores, k=k, num_positives=num_positives) == pytest.approx(expected)
