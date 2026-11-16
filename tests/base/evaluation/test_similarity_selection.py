from __future__ import annotations

import pytest

from sentence_transformers.base.evaluation import SequentialEvaluator
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorTripletEvaluator
from sentence_transformers.sentence_transformer.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseTripletEvaluator,
)
from sentence_transformers.util.similarity import SimilarityFunction

MAIN_CASES = [
    (EmbeddingSimilarityEvaluator, "stsb_bert_tiny_model", "main_similarity", ["cosine", "euclidean"]),
    (SparseEmbeddingSimilarityEvaluator, "splade_bert_tiny_model", "main_similarity", ["dot", "cosine"]),
    (TripletEvaluator, "stsb_bert_tiny_model", "main_similarity_function", ["cosine", "euclidean"]),
    (SparseTripletEvaluator, "splade_bert_tiny_model", "main_similarity_function", ["dot", "cosine"]),
    (MultiVectorTripletEvaluator, "mve_bert_tiny_model", "main_similarity_function", ["maxsim", "meanmaxsim"]),
]
ALL_CASES = [
    *MAIN_CASES,
    (BinaryClassificationEvaluator, "stsb_bert_tiny_model", None, ["cosine", "dot"]),
    (SparseBinaryClassificationEvaluator, "splade_bert_tiny_model", None, ["dot", "cosine"]),
]
CASE_PARAMETERS = "evaluator_class, model_fixture, main_parameter, names"


def make_evaluator(evaluator_class, **kwargs):
    sentences1 = ["A man is eating food.", "A cat sits outside.", "The girl plays guitar."]
    sentences2 = ["A man eats something.", "The sky is blue.", "A woman plays a guitar."]
    third_inputs = ["The sky is blue.", "A man eats something.", "A cat sits outside."]
    if not issubclass(evaluator_class, TripletEvaluator):
        third_inputs = [1, 0, 1]
    return evaluator_class(sentences1, sentences2, third_inputs, name="test", **kwargs)


@pytest.mark.parametrize(CASE_PARAMETERS, MAIN_CASES)
@pytest.mark.parametrize("selection", ["model", "empty", "missing", "included"])
@pytest.mark.parametrize("as_enum", [False, True])
def test_main_similarity_selection(evaluator_class, model_fixture, main_parameter, names, selection, as_enum):
    main = names[1]
    requested_names = {"model": None, "empty": [], "missing": names[:1], "included": names.copy()}[selection]
    original_names = requested_names.copy() if requested_names is not None else None
    evaluator = make_evaluator(
        evaluator_class,
        similarity_fn_names=requested_names,
        **{main_parameter: SimilarityFunction(main) if as_enum else main},
    )
    expected_names = names if selection in ("missing", "included") else [main]
    assert evaluator.similarity_fn_names == expected_names
    assert getattr(evaluator, main_parameter) is SimilarityFunction(main)
    assert requested_names == original_names


@pytest.mark.parametrize(CASE_PARAMETERS, MAIN_CASES)
def test_main_similarity_is_evaluated(evaluator_class, model_fixture, main_parameter, names, request):
    evaluator = make_evaluator(evaluator_class, similarity_fn_names=names[:1], **{main_parameter: names[1]})
    metrics = SequentialEvaluator([evaluator])(request.getfixturevalue(model_fixture))
    expected_keys = (
        [f"test_spearman_{name}" for name in names]
        if issubclass(evaluator_class, EmbeddingSimilarityEvaluator)
        else [f"test_{name}_accuracy" for name in names]
    )

    assert all(key in metrics for key in expected_keys)
    assert evaluator.primary_metric == expected_keys[1]
    assert metrics["sequential_score"] == metrics[evaluator.primary_metric]


@pytest.mark.parametrize(CASE_PARAMETERS, ALL_CASES)
def test_unknown_similarity_is_rejected(evaluator_class, model_fixture, main_parameter, names):
    with pytest.raises(ValueError, match="similarity_fn_names.*unknown"):
        make_evaluator(evaluator_class, similarity_fn_names=[names[0], "unknown"])


@pytest.mark.parametrize(CASE_PARAMETERS, MAIN_CASES)
def test_unsupported_main_similarity_is_rejected(evaluator_class, model_fixture, main_parameter, names):
    unsupported = "dot" if evaluator_class is MultiVectorTripletEvaluator else "maxsim"
    with pytest.raises(ValueError, match="similarity_fn_names"):
        make_evaluator(evaluator_class, **{main_parameter: unsupported})
