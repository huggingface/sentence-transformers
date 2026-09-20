"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator


def test_negative_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Basic test with 'negative' format produces valid metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "negative": ["Java is a language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "negative": ["The moon orbits earth", "Mars is red"],
        },
        {"query": "What is water?", "positive": ["Water is H2O"], "negative": ["Salt is NaCl", "Gold is Au"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="neg-test")
    results = evaluator(reranker_bert_tiny_model)

    assert "neg-test_map" in results
    assert f"neg-test_mrr@{evaluator.at_k}" in results
    assert f"neg-test_ndcg@{evaluator.at_k}" in results
    for value in results.values():
        assert 0.0 <= value <= 1.0


def test_documents_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Test with 'documents' format produces both base and reranked metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "documents": ["Java is a language", "Python is a programming language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "documents": ["The moon orbits earth", "The sun is a star", "Mars is red"],
        },
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="doc-test")
    results = evaluator(reranker_bert_tiny_model)

    # Should have both base and reranked metrics
    assert "doc-test_map" in results
    assert "doc-test_base_map" in results
    assert f"doc-test_ndcg@{evaluator.at_k}" in results
    assert f"doc-test_base_ndcg@{evaluator.at_k}" in results


def test_no_relevant_docs_in_documents(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Edge case: positive not in documents list."""
    samples = [
        {"query": "empty", "positive": ["x"], "documents": ["a", "b", "c"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="edge-test")
    results = evaluator(reranker_bert_tiny_model)

    assert f"edge-test_ndcg@{evaluator.at_k}" in results


def test_single_sample(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Single sample still works."""
    samples = [
        {"query": "test", "positive": ["yes"], "negative": ["no"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="single")
    results = evaluator(reranker_bert_tiny_model)

    assert f"single_ndcg@{evaluator.at_k}" in results


def test_string_positive(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Positive as a string (not list) is handled correctly."""
    samples = [
        {"query": "test", "positive": "yes", "negative": ["no", "nope"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="str-pos")
    results = evaluator(reranker_bert_tiny_model)

    assert f"str-pos_ndcg@{evaluator.at_k}" in results


def test_validation_errors() -> None:
    """Invalid samples raise clear errors."""
    with pytest.raises(ValueError, match="query"):
        CrossEncoderRerankingEvaluator([{"positive": ["a"], "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="positive"):
        CrossEncoderRerankingEvaluator([{"query": "q", "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"], "negative": ["b"], "documents": ["c"]}])(
            None
        )


class _PerfectReranker:
    """Scores the documents the evaluator hands it, positives first, nothing else.

    A stub rather than a real cross encoder: the point is what the evaluator does with a
    ranking, and only an exact ranking makes the ceiling visible.
    """

    def __init__(self, positive: list[str]) -> None:
        self.positive = positive
        self.model_card_data = SimpleNamespace(set_evaluation_metrics=lambda *args, **kwargs: None)

    def predict(self, pairs, **kwargs):
        return np.array([1.0 if doc in self.positive else 0.0 for _, doc in pairs])


def test_unretrieved_positives_cap_the_reranked_score_too() -> None:
    """``always_rerank_positives=False`` documents a ceiling below 1.0 when the first stage
    missed a positive. The base ranking counted those positives; the reranked one dropped
    them, so a perfect reranker scored 1.0 against a base that could not, and the reported
    improvement was the retriever's recall gap rather than the model's work."""
    positive = ["p1", "p2"]
    samples = [{"query": "q", "positive": positive, "documents": ["n1", "p1", "n2", "n3"]}]
    evaluator = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False, name="recall")
    results = evaluator(_PerfectReranker(positive))

    ndcg = results[f"recall_ndcg@{evaluator.at_k}"]
    assert results["recall_map"] < 1.0, "one of the two positives was never scored"
    assert ndcg < 1.0
    # And comparable with the base, which charges for the same missed positive.
    assert results["recall_base_map"] <= results["recall_map"]
    assert results[f"recall_base_ndcg@{evaluator.at_k}"] <= ndcg


def test_retrieved_positives_still_reach_one() -> None:
    """Nothing missing, nothing to cap: the ceiling stays 1.0."""
    positive = ["p1"]
    samples = [{"query": "q", "positive": positive, "documents": ["n1", "p1", "n2"]}]
    evaluator = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False, name="full")
    results = evaluator(_PerfectReranker(positive))

    assert results["full_map"] == pytest.approx(1.0)
    assert results[f"full_ndcg@{evaluator.at_k}"] == pytest.approx(1.0)
    assert results[f"full_mrr@{evaluator.at_k}"] == pytest.approx(1.0)
