from __future__ import annotations

import csv

import pytest

from sentence_transformers.base.evaluation import SequentialEvaluator
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
    MultiVectorTripletEvaluator,
)
from sentence_transformers.sentence_transformer.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    TripletEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
    SparseNanoBEIREvaluator,
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
def test_main_similarity_selection(evaluator_class, model_fixture, main_parameter, names, selection):
    main = names[1]
    requested_names = {"model": None, "empty": [], "missing": names[:1], "included": names.copy()}[selection]
    original_names = requested_names.copy() if requested_names is not None else None
    evaluator = make_evaluator(evaluator_class, similarity_fn_names=requested_names, **{main_parameter: main})
    expected_names = names if selection in ("missing", "included") else [main]
    assert evaluator.similarity_fn_names == expected_names
    assert getattr(evaluator, main_parameter) is SimilarityFunction(main)
    assert requested_names == original_names


@pytest.mark.parametrize(CASE_PARAMETERS, MAIN_CASES)
def test_main_similarity_is_evaluated(evaluator_class, model_fixture, main_parameter, names, request):
    evaluator = make_evaluator(
        evaluator_class, similarity_fn_names=names[:1], **{main_parameter: SimilarityFunction(names[1])}
    )
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
@pytest.mark.parametrize("explicit_similarity", [False, True])
def test_evaluator_reuse(
    evaluator_class, model_fixture, main_parameter, names, explicit_similarity, request, tmp_path
):
    model = request.getfixturevalue(model_fixture)
    evaluator = make_evaluator(evaluator_class, similarity_fn_names=names[:1] if explicit_similarity else None)

    for name in names:
        model.similarity_fn_name = name
        output_path = tmp_path / name
        metrics = evaluator(model, output_path=output_path)
        expected_names = names[:1] if explicit_similarity else [name]
        reference = make_evaluator(evaluator_class, similarity_fn_names=expected_names)

        assert metrics == pytest.approx(reference(model))
        assert evaluator.primary_metric == reference.primary_metric
        assert evaluator.similarity_fn_names == expected_names
        with (output_path / evaluator.csv_file).open(newline="", encoding="utf-8") as f:
            header, *rows = csv.reader(f)
        assert header == reference.csv_headers
        assert len(rows) == 1
        assert len(rows[0]) == len(header)


@pytest.mark.parametrize(CASE_PARAMETERS, ALL_CASES)
def test_unknown_similarity_is_rejected(evaluator_class, model_fixture, main_parameter, names):
    with pytest.raises(ValueError, match="similarity_fn_names.*unknown"):
        make_evaluator(evaluator_class, similarity_fn_names=[names[0], "unknown"])


@pytest.mark.parametrize(CASE_PARAMETERS, MAIN_CASES)
def test_unsupported_main_similarity_is_rejected(evaluator_class, model_fixture, main_parameter, names):
    unsupported = "dot" if evaluator_class is MultiVectorTripletEvaluator else "maxsim"
    with pytest.raises(ValueError, match="similarity_fn_names"):
        make_evaluator(evaluator_class, **{main_parameter: unsupported})


IR_CASES = [
    (InformationRetrievalEvaluator, "stsb_bert_tiny_model", ["cosine", "dot"]),
    (NanoBEIREvaluator, "stsb_bert_tiny_model", ["cosine", "dot"]),
    (SparseInformationRetrievalEvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (SparseNanoBEIREvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (MultiVectorInformationRetrievalEvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
    (MultiVectorNanoBEIREvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
]
IR_CASE_PARAMETERS = "evaluator_class, model_fixture, names"


@pytest.fixture
def make_ir_evaluator(monkeypatch):
    subsets = {
        "queries": [{"_id": "q", "text": "What is the capital of France?"}],
        "corpus": [
            {"_id": "d", "text": "Paris is the capital of France."},
            {"_id": "n", "text": "Berlin is the capital of Germany."},
        ],
        "qrels": [{"query-id": "q", "corpus-id": "d"}],
    }
    monkeypatch.setattr(
        NanoBEIREvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: subsets[subset]
    )

    def make(evaluator_class, **kwargs):
        if issubclass(evaluator_class, NanoBEIREvaluator):
            return evaluator_class(dataset_names=["MSMARCO"], **kwargs)
        return evaluator_class(
            queries={"q": subsets["queries"][0]["text"]},
            corpus={row["_id"]: row["text"] for row in subsets["corpus"]},
            relevant_docs={"q": {"d"}},
            name="test",
            **kwargs,
        )

    return make


@pytest.mark.parametrize(IR_CASE_PARAMETERS, IR_CASES)
@pytest.mark.parametrize("configuration", ["empty", "missing_main"])
def test_ir_rejects_missing_main_scorer(evaluator_class, model_fixture, names, configuration, make_ir_evaluator):
    score_functions = {}
    if configuration == "missing_main":
        score_functions = {names[0]: SimilarityFunction.to_similarity_fn(names[0])}
    with pytest.raises(ValueError, match="main_score_function"):
        make_ir_evaluator(evaluator_class, score_functions=score_functions, main_score_function=names[1])


@pytest.mark.parametrize(IR_CASE_PARAMETERS, IR_CASES)
def test_ir_accepts_main_scorer_in_explicit_mapping(evaluator_class, model_fixture, names, make_ir_evaluator, request):
    main = names[1]
    evaluator = make_ir_evaluator(
        evaluator_class,
        score_functions={name: SimilarityFunction.to_similarity_fn(name) for name in names},
        main_score_function=SimilarityFunction(main),
        write_csv=False,
    )
    metrics = evaluator(request.getfixturevalue(model_fixture))

    assert evaluator.score_function_names == sorted(names)
    assert evaluator.primary_metric == f"{evaluator.name}_{main}_ndcg@10"
    for name in evaluator.score_function_names:
        assert f"{evaluator.name}_{name}_ndcg@10" in metrics
