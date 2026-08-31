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
