"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

import math
from unittest.mock import Mock

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


@pytest.mark.parametrize("at_k", [1, 10])
@pytest.mark.parametrize("positive", [[], ["positive"], ["positive1", "positive2"]])
def test_empty_documents_without_positive_injection(positive: list[str], at_k: int) -> None:
    """A zero-hit query contributes zero metrics without invoking the reranker."""
    model = Mock()
    evaluator = CrossEncoderRerankingEvaluator(
        [{"query": "no hits", "positive": positive, "documents": []}],
        always_rerank_positives=False,
        at_k=at_k,
        name="empty",
    )

    results = evaluator(model)

    assert results == {
        "empty_map": 0.0,
        f"empty_mrr@{at_k}": 0.0,
        f"empty_ndcg@{at_k}": 0.0,
        "empty_base_map": 0.0,
        f"empty_base_mrr@{at_k}": 0.0,
        f"empty_base_ndcg@{at_k}": 0.0,
    }
    model.predict.assert_not_called()
    assert evaluator.primary_metric in results


@pytest.mark.parametrize("positive", [[], ["positive1", "positive2"]])
def test_empty_documents_with_positive_injection(positive: list[str]) -> None:
    """The explicit positive-injection option still applies to an empty retrieval."""
    model = Mock()
    model.predict.return_value = np.array([0.8, 0.4])
    evaluator = CrossEncoderRerankingEvaluator(
        [{"query": "no hits", "positive": positive, "documents": []}],
        always_rerank_positives=True,
    )

    results = evaluator(model)

    expected_reranked = 1.0 if positive else 0.0
    assert results["map"] == results["mrr@10"] == results["ndcg@10"] == expected_reranked
    assert results["base_map"] == results["base_mrr@10"] == results["base_ndcg@10"] == 0.0
    if positive:
        model.predict.assert_called_once()
        assert model.predict.call_args.args[0] == [["no hits", doc] for doc in positive]
    else:
        model.predict.assert_not_called()


@pytest.mark.parametrize("at_k", [1, 2, 10])
def test_empty_documents_remain_in_macro_average(at_k: int) -> None:
    """Empty results between nonempty samples must not shift predictions or vanish from the denominator."""
    samples = [
        {"query": "empty1", "positive": ["missing1"], "documents": []},
        {"query": "found1", "positive": ["positive1"], "documents": ["negative1", "positive1"]},
        {"query": "empty2", "positive": ["missing2"], "documents": []},
        {"query": "found2", "positive": ["positive2"], "documents": ["positive2", "negative2"]},
    ]
    model = Mock()
    model.predict.return_value = np.array([0.2, 0.9, 0.8, 0.1])
    evaluator = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False, at_k=at_k)

    results = evaluator(model)

    assert results["map"] == results[f"mrr@{at_k}"] == results[f"ndcg@{at_k}"] == 0.5
    assert results["base_map"] == 0.375
    assert results[f"base_mrr@{at_k}"] == (0.25 if at_k == 1 else 0.375)
    assert results[f"base_ndcg@{at_k}"] == pytest.approx(0.25 if at_k == 1 else (1 / math.log2(3) + 1) / 4)
    model.predict.assert_called_once()
    assert model.predict.call_args.args[0] == [
        ["found1", "negative1"],
        ["found1", "positive1"],
        ["found2", "positive2"],
        ["found2", "negative2"],
    ]


def test_empty_documents_preserve_duplicate_candidate_scoring() -> None:
    """The zero-hit fix does not deduplicate or reweight an existing candidate ranking."""
    samples = [
        {"query": "empty", "positive": ["missing"], "documents": []},
        {"query": "found", "positive": ["positive"], "documents": ["negative", "positive", "positive"]},
    ]
    model = Mock()
    model.predict.return_value = np.array([0.2, 0.9, 0.9])

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False)(model)

    assert results["map"] == results["mrr@10"] == results["ndcg@10"] == 0.5
    assert results["base_map"] == pytest.approx((1 / 2 + 2 / 3) / 4)
    assert results["base_mrr@10"] == 0.25
    assert model.predict.call_args.args[0] == [
        ["found", "negative"],
        ["found", "positive"],
        ["found", "positive"],
    ]
