from __future__ import annotations

import codecs
import csv
from unittest.mock import Mock

import pytest

from sentence_transformers.base.evaluation import BaseEvaluator, SequentialEvaluator
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
    MultiVectorNanoRetrievalEvaluator,
    MultiVectorTripletEvaluator,
)
from sentence_transformers.sentence_transformer.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    NanoRetrievalEvaluator,
    TripletEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
    SparseNanoBEIREvaluator,
    SparseNanoRetrievalEvaluator,
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
def test_evaluate_extends_metrics_before_output(
    evaluator_class, model_fixture, main_parameter, names, request, tmp_path, monkeypatch
):
    model = request.getfixturevalue(model_fixture)
    evaluator = make_evaluator(evaluator_class, similarity_fn_names=names)
    store_metrics = Mock()
    monkeypatch.setattr(model.model_card_data, "set_evaluation_metrics", store_metrics)

    raw_metrics = evaluator.evaluate(model, output_path=tmp_path)
    assert not list(tmp_path.iterdir())
    store_metrics.assert_not_called()
    assert not any(key.startswith("test_") for key in raw_metrics)

    write_row = evaluator._write_csv_row

    def write_complete_row(output_path, row):
        assert len(row) == len(evaluator.csv_headers)
        write_row(output_path, row)

    monkeypatch.setattr(evaluator, "_write_csv_row", write_complete_row)
    metrics = evaluator(model, output_path=tmp_path, epoch=2, steps=3)
    assert metrics == pytest.approx({f"test_{key}": value for key, value in raw_metrics.items()})
    store_metrics.assert_called_once_with(evaluator, metrics, 2, 3)


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
    (NanoRetrievalEvaluator, "stsb_bert_tiny_model", ["cosine", "dot"]),
    (SparseInformationRetrievalEvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (SparseNanoBEIREvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (SparseNanoRetrievalEvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (MultiVectorInformationRetrievalEvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
    (MultiVectorNanoBEIREvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
    (MultiVectorNanoRetrievalEvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
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
        NanoRetrievalEvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: subsets[subset]
    )

    def make(evaluator_class, **kwargs):
        if issubclass(evaluator_class, NanoBEIREvaluator):
            kwargs.setdefault("dataset_names", ["MSMARCO"])
            return evaluator_class(**kwargs)
        if issubclass(evaluator_class, NanoRetrievalEvaluator):
            return evaluator_class(dataset_id="test-org/custom-nano", splits=["task"], **kwargs)
        return evaluator_class(
            queries={"q": subsets["queries"][0]["text"]},
            corpus={row["_id"]: row["text"] for row in subsets["corpus"]},
            relevant_docs={"q": {"d"}},
            name="test",
            **kwargs,
        )

    return make


@pytest.mark.parametrize(
    "evaluator_class, model_fixture",
    [
        (NanoBEIREvaluator, "stsb_bert_tiny_model"),
        (SparseNanoBEIREvaluator, "splade_bert_tiny_model"),
        (MultiVectorNanoBEIREvaluator, "mve_bert_tiny_model"),
    ],
)
@pytest.mark.parametrize("dataset_names", [["MSMARCO"], ["MSMARCO", "scifact"]])
def test_nanobeir_stores_complete_aggregates(
    evaluator_class, model_fixture, dataset_names, make_ir_evaluator, request, monkeypatch, tmp_path
):
    model = request.getfixturevalue(model_fixture)
    evaluator = make_ir_evaluator(evaluator_class, dataset_names=dataset_names, aggregate_key="mean_test")
    store_metrics = Mock()
    monkeypatch.setattr(model.model_card_data, "set_evaluation_metrics", store_metrics)
    results = evaluator(model, output_path=tmp_path)

    assert store_metrics.call_count == len(dataset_names) + (len(dataset_names) > 1)
    for call in store_metrics.call_args_list:
        child, metrics, epoch, steps = call.args
        assert all(key.startswith(f"{child.name}_") for key in metrics)
        assert metrics == {key: value for key, value in results.items() if key.startswith(f"{child.name}_")}
        assert epoch == steps == -1

    for child in [*evaluator.evaluators, evaluator]:
        with (tmp_path / child.csv_file).open(newline="", encoding="utf-8") as f:
            header, row = csv.reader(f)
        assert len(header) == len(row)
        if isinstance(child, (SparseNanoBEIREvaluator, SparseInformationRetrievalEvaluator)):
            for key, value in child.sparsity_stats.items():
                assert float(row[header.index(key)]) == pytest.approx(value)
                assert results[f"{child.name}_{key}"] == pytest.approx(value)


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


@pytest.mark.parametrize(
    IR_CASE_PARAMETERS,
    [(cls, model_fixture, names) for cls, model_fixture, _, names in ALL_CASES] + IR_CASES,
)
def test_csv_headers_follow_output_file(
    evaluator_class, model_fixture, names, make_ir_evaluator, request, tmp_path, caplog
):
    model = request.getfixturevalue(model_fixture)
    make = (
        make_evaluator
        if issubclass(evaluator_class, (EmbeddingSimilarityEvaluator, TripletEvaluator, BinaryClassificationEvaluator))
        else make_ir_evaluator
    )
    evaluator = make(evaluator_class)
    model.similarity_fn_name = names[0]
    evaluator(model, output_path=tmp_path)
    originals = {path: path.read_bytes() for path in tmp_path.glob("*.csv")}
    assert originals

    model.similarity_fn_name = names[1]
    expected_metrics = evaluator(model)
    with caplog.at_level("WARNING"):
        assert evaluator(model, output_path=tmp_path) == pytest.approx(expected_metrics)
    assert "Skipping CSV row" in caplog.text
    evaluator(model, output_path=tmp_path / "other")
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert make(evaluator_class)(model, output_path=tmp_path) == pytest.approx(expected_metrics)
    assert "Skipping CSV row" in caplog.text
    assert {path: path.read_bytes() for path in originals} == originals

    model.similarity_fn_name = names[0]
    evaluator(model, output_path=tmp_path)
    for path in originals:
        with path.open(newline="", encoding="utf-8") as f:
            header, *rows = csv.reader(f)
        assert len(rows) == 2
        assert all(len(row) == len(header) for row in rows)


@pytest.mark.parametrize("encoding", [None, "utf-8", "utf-8-sig"])
def test_csv_append(tmp_path, encoding):
    evaluator = BaseEvaluator()
    evaluator.csv_file = "results.csv"
    evaluator.csv_headers = ["epoch", "steps", "score"]
    csv_path = tmp_path / evaluator.csv_file
    if encoding is None:
        csv_path.touch()
    else:
        with csv_path.open("w", newline="", encoding=encoding) as f:
            csv.writer(f, quoting=csv.QUOTE_ALL).writerow(evaluator.csv_headers)

    evaluator._write_csv_row(tmp_path, [0, 10, 0.5])
    evaluator._write_csv_row(tmp_path, [1, 20, 0.75])

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        assert list(csv.reader(f)) == [evaluator.csv_headers, ["0", "10", "0.5"], ["1", "20", "0.75"]]
    assert csv_path.read_bytes().startswith(codecs.BOM_UTF8) == (encoding == "utf-8-sig")
