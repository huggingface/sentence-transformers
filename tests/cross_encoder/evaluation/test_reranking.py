"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

from math import log2
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


def test_empty_documents() -> None:
    samples = [
        {"query": "empty", "positive": ["missing"], "documents": []},
        {"query": "found", "positive": ["positive"], "documents": ["positive", "negative"]},
    ]
    model = Mock()
    model.predict.return_value = [1.0, 0.0]

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False)(model)

    assert results == {
        "map": 0.5,
        "mrr@10": 0.5,
        "ndcg@10": 0.5,
        "base_map": 0.5,
        "base_mrr@10": 0.5,
        "base_ndcg@10": 0.5,
    }
    model.predict.assert_called_once()
    assert model.predict.call_args.args[0] == [["found", "positive"], ["found", "negative"]]


@pytest.mark.parametrize("always_rerank_positives", [False, True])
@pytest.mark.parametrize("at_k", [1, 10])
def test_missing_positives(always_rerank_positives: bool, at_k: int) -> None:
    samples = [{"query": "q", "positive": ["p1", "p2"], "documents": ["n1", "p1", "n2", "n3"]}]
    model = Mock()
    model.predict.return_value = np.array(
        [2.0, 1.0, 0.0, -1.0, -2.0] if always_rerank_positives else [0.0, 1.0, -1.0, -2.0]
    )

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=always_rerank_positives, at_k=at_k)(
        model
    )

    ideal_dcg = 1 + 1 / log2(3) if at_k > 1 else 1
    assert results == pytest.approx(
        {
            "map": 1.0 if always_rerank_positives else 0.5,
            f"mrr@{at_k}": 1.0,
            f"ndcg@{at_k}": 1.0 if always_rerank_positives else 1 / ideal_dcg,
            "base_map": 0.25,
            f"base_mrr@{at_k}": 0.5 if at_k > 1 else 0.0,
            f"base_ndcg@{at_k}": 1 / log2(3) / ideal_dcg if at_k > 1 else 0.0,
        }
    )
    expected_docs = ["p1", "p2", "n1", "n2", "n3"] if always_rerank_positives else samples[0]["documents"]
    model.predict.assert_called_once()
    assert model.predict.call_args.args[0] == [["q", doc] for doc in expected_docs]


@pytest.mark.parametrize("num_positives", [1, 3])
@pytest.mark.parametrize("documents", [["p1"], ["p1", "n1", "n2"]])
def test_unchanged_ranking(num_positives: int, documents: list[str]) -> None:
    positives = [f"p{i}" for i in range(1, num_positives + 1)]
    samples = [{"query": "q", "positive": positives, "documents": documents}]
    model = Mock()
    model.predict.return_value = np.array([3.0, 2.0, 1.0][: len(documents)])

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False)(model)

    expected = {
        "map": 1.0 / num_positives,
        "mrr@10": 1.0,
        "ndcg@10": 1.0 if num_positives == 1 else 1 / (1 + 1 / log2(3) + 0.5),
    }
    assert results == pytest.approx({**expected, **{f"base_{key}": value for key, value in expected.items()}})


def test_missing_positives_with_tied_scores() -> None:
    samples = [{"query": "q", "positive": ["p1", "p2"], "documents": ["p1", "n1"]}]
    model = Mock()
    model.predict.return_value = np.array([0.0, 0.0])

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False, at_k=1)(model)

    assert results["map"] == pytest.approx(0.25)
    assert results["ndcg@1"] == pytest.approx(0.5)
