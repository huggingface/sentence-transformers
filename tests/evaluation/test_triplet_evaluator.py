"""
Tests the correct computation of evaluation scores from TripletEvaluator
"""

from __future__ import annotations

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
