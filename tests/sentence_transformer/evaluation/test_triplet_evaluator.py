"""
Tests the correct computation of evaluation scores from TripletEvaluator
"""

from __future__ import annotations

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator


def test_TripletEvaluator(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Tests that the TripletEvaluator can be loaded & used"""
    model = stsb_bert_tiny_model
    anchors = [
        "A person on a horse jumps over a broken down airplane.",
        "Children smiling and waving at camera",
        "A boy is jumping on skateboard in the middle of a red bridge.",
    ]
    positives = [
        "A person is outdoors, on a horse.",
        "There are children looking at the camera.",
        "The boy does a skateboarding trick.",
    ]
    negatives = [
        "A person is at a diner, ordering an omelette.",
        "The kids are frowning",
        "The boy skates down the sidewalk.",
    ]
    evaluator = TripletEvaluator(anchors, positives, negatives, name="all_nli_dev")
    metrics = evaluator(model)
    assert evaluator.primary_metric == "all_nli_dev_cosine_accuracy"
    assert metrics[evaluator.primary_metric] == 1.0

    evaluator_with_margin = TripletEvaluator(anchors, positives, negatives, margin=0.7, name="all_nli_dev")
    metrics = evaluator_with_margin(model)
    assert metrics[evaluator.primary_metric] == 0.0


def test_triplet_evaluator_main_similarity_function_is_evaluated(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # The model scores with cosine, so the requested euclidean accuracy must still be computed
    evaluator = TripletEvaluator(["a"], ["b"], ["c"], main_similarity_function="euclidean", name="dev")
    metrics = evaluator(stsb_bert_tiny_model)
    assert evaluator.primary_metric == "dev_euclidean_accuracy"
    assert evaluator.primary_metric in metrics


def test_triplet_evaluator_config_dict_margin() -> None:
    # The default margin is derived from _get_similarity_functions(), so it must stay recognized as
    # the default here rather than being reported as a configured margin.
    triplet = {"anchors": ["a"], "positives": ["p"], "negatives": ["n"]}
    assert "margin" not in TripletEvaluator(**triplet).get_config_dict()

    evaluator = TripletEvaluator(**triplet, margin=0.5)
    assert evaluator.get_config_dict()["margin"] == {"cosine": 0.5, "dot": 0.5, "manhattan": 0.5, "euclidean": 0.5}


def test_triplet_evaluator_rejects_unknown_margin_keys() -> None:
    with pytest.raises(ValueError, match=r"unexpected keys \['cosien'\]"):
        TripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"], margin={"cosien": 0.5})


def test_triplet_evaluator_rejects_unknown_similarity_fn_names() -> None:
    # Without this guard a typo passes construction, weakens the margin validation into accepting
    # the same typo, and only fails deep in __call__ at scoring time.
    with pytest.raises(ValueError, match=r"unexpected names \['cosin'\]"):
        TripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"], similarity_fn_names=["cosin"])


def test_reused_evaluator_uses_each_models_similarity(stsb_bert_tiny_model: SentenceTransformer, tmp_path, caplog):
    """Without similarity_fn_names the similarity is resolved per call, so a reused evaluator
    scores and labels every model with its own similarity rather than with the first one's."""
    model = stsb_bert_tiny_model
    evaluator = TripletEvaluator(
        anchors=["a cat sits", "a man plays guitar"],
        positives=["a kitten rests", "someone plays music"],
        negatives=["the stock market fell", "grass is green"],
        name="test",
    )
    model.similarity_fn_name = "cosine"
    results = evaluator(model, output_path=str(tmp_path))
    assert set(results) == {"test_cosine_accuracy"}
    assert evaluator.primary_metric == "test_cosine_accuracy"

    model.similarity_fn_name = "dot"
    with caplog.at_level("WARNING"):
        results = evaluator(model, output_path=str(tmp_path))

    assert set(results) == {"test_dot_accuracy"}
    assert evaluator.similarity_fn_names == ["dot"]
    assert evaluator.primary_metric == "test_dot_accuracy"
    assert evaluator.csv_headers == ["epoch", "steps", "accuracy_dot"]
    # The CSV keeps the header row of the first call, so the appended row is mislabeled.
    assert "labeled with the previous similarity function" in caplog.text


def test_explicit_similarity_fn_names_survive_reuse(stsb_bert_tiny_model: SentenceTransformer):
    model = stsb_bert_tiny_model
    evaluator = TripletEvaluator(
        anchors=["a cat sits"],
        positives=["a kitten rests"],
        negatives=["the stock market fell"],
        similarity_fn_names=["cosine"],
        write_csv=False,
    )
    evaluator(model)
    model.similarity_fn_name = "dot"
    results = evaluator(model)

    assert evaluator.similarity_fn_names == ["cosine"]
    assert set(results) == {"cosine_accuracy"}
