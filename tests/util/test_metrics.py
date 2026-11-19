from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.util import reciprocal_rank


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize(
    ("labels", "scores", "k", "expected"),
    [
        pytest.param([0, 1, 0, 1], [1.0] * 4, 10, 13 / 18, id="interleaved-positives"),
        pytest.param([0, 1, 0, 1], [1.0] * 4, 1, 1 / 2, id="cutoff-inside-tie"),
        pytest.param([0, 1, 0], [0.9, 0.8, 0.7], 10, 1 / 2, id="distinct-scores"),
        pytest.param([0, 0], [1.0, 1.0], 10, 0.0, id="no-positives"),
        pytest.param([1, 1], [1.0, 1.0], 10, 1.0, id="all-positives"),
    ],
)
def test_reciprocal_rank(as_array, labels, scores, k, expected):
    if as_array:
        labels, scores = np.asarray(labels), np.asarray(scores)
    assert reciprocal_rank(labels, scores, k=k) == pytest.approx(expected)
