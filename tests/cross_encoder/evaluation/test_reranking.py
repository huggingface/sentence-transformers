"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

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


# compute_metrics scores one query at a time, so these samples only serve to build the evaluator
METRIC_SAMPLES = [{"query": "q", "positive": ["p"], "negative": ["n"]}]


def test_mrr_is_independent_of_candidate_order_under_ties() -> None:
    """Equally scored candidates may be passed in any order, so MRR should not depend on that order"""
    evaluator = CrossEncoderRerankingEvaluator(METRIC_SAMPLES)
    scores = [0.5, 0.5, 0.5, 0.5, 0.5]

    mrr_first, _, _ = evaluator.compute_metrics([1, 0, 0, 0, 0], scores)
    mrr_last, _, _ = evaluator.compute_metrics([0, 0, 0, 0, 1], scores)

    assert mrr_first == pytest.approx(mrr_last)
    # The positive is equally likely to end up at any of the five positions
    assert mrr_first == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 5)


def test_mrr_with_several_relevant_documents_in_one_tie() -> None:
    """A tie holding more than one relevant document reaches a relevant one sooner"""
    evaluator = CrossEncoderRerankingEvaluator(METRIC_SAMPLES)

    mrr, _, _ = evaluator.compute_metrics([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5])

    # P(first relevant at position 1..3) = 1/2, 1/3, 1/6
    assert mrr == pytest.approx(1 / 2 + (1 / 3) / 2 + (1 / 6) / 3)


def test_mrr_when_every_tied_document_is_relevant() -> None:
    """With no irrelevant document to get in the way the first position is always relevant"""
    evaluator = CrossEncoderRerankingEvaluator(METRIC_SAMPLES)

    mrr, _, _ = evaluator.compute_metrics([1, 1, 1], [0.5, 0.5, 0.5])

    assert mrr == pytest.approx(1.0)


def test_mrr_when_tie_starts_after_at_k() -> None:
    """A tie that begins beyond at_k cannot contribute"""
    evaluator = CrossEncoderRerankingEvaluator(METRIC_SAMPLES, at_k=2)

    mrr, _, _ = evaluator.compute_metrics([0, 0, 1, 0], [0.9, 0.8, 0.5, 0.5])

    assert mrr == pytest.approx(0.0)


def test_mrr_without_ties_is_the_plain_reciprocal_rank() -> None:
    """Distinct scores keep the previous behaviour"""
    evaluator = CrossEncoderRerankingEvaluator(METRIC_SAMPLES)

    assert evaluator.compute_metrics([0, 1, 0], [0.9, 0.8, 0.1])[0] == pytest.approx(1 / 2)
    assert evaluator.compute_metrics([1, 0, 0], [0.9, 0.8, 0.1])[0] == pytest.approx(1.0)
