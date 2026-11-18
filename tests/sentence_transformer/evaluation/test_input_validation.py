from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.sentence_transformer.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    TranslationEvaluator,
    TripletEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseTranslationEvaluator,
    SparseTripletEvaluator,
)

# (evaluator class, number of list arguments)
EVALUATORS = [
    (TranslationEvaluator, 2),
    (SparseTranslationEvaluator, 2),
    (EmbeddingSimilarityEvaluator, 3),
    (SparseEmbeddingSimilarityEvaluator, 3),
    (BinaryClassificationEvaluator, 3),
    (SparseBinaryClassificationEvaluator, 3),
    (TripletEvaluator, 3),
    (SparseTripletEvaluator, 3),
]


@pytest.mark.parametrize(("evaluator_cls", "num_inputs"), EVALUATORS)
def test_evaluator_rejects_mismatched_lengths(evaluator_cls, num_inputs) -> None:
    # Two items in all but the last argument, which has a different length (labels/scores are numeric)
    last = (
        [0]
        if evaluator_cls.__name__.endswith(("BinaryClassificationEvaluator", "EmbeddingSimilarityEvaluator"))
        else ["c"]
    )
    inputs = [["a", "b"]] * (num_inputs - 1) + [last]
    with pytest.raises(ValueError, match="must have the same length"):
        evaluator_cls(*inputs)


@pytest.mark.parametrize(("evaluator_cls", "num_inputs"), EVALUATORS)
def test_evaluator_rejects_empty_input(evaluator_cls, num_inputs) -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluator_cls(*[[] for _ in range(num_inputs)])


@pytest.mark.parametrize("evaluator_cls", [BinaryClassificationEvaluator, SparseBinaryClassificationEvaluator])
def test_binary_classification_rejects_non_binary_labels(evaluator_cls) -> None:
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        evaluator_cls(["a", "b"], ["c", "d"], [0, 2])


@pytest.mark.parametrize("labels", [[1, 0], [True, False], [1.0, 0.0], np.array([1, 0])])
def test_binary_classification_accepts_binary_label_types(labels) -> None:
    evaluator = BinaryClassificationEvaluator(["a", "b"], ["c", "d"], labels)
    assert len(evaluator.labels) == 2
