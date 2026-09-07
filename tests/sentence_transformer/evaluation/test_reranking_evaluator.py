"""
Tests the correct computation of evaluation scores from RerankingEvaluator
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator

SAMPLES = [
    {
        "query": "What is Python?",
        "positive": ["Python is a programming language"],
        "negative": ["Java is a language", "C++ is fast"],
    }
]


def tied_similarity(query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
    """Scores every document identically, e.g. as duplicate documents or quantized embeddings do"""
    return torch.full((1, len(document_embeddings)), 0.5)


def test_RerankingEvaluator_mrr_with_tied_scores(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The positive is equally likely to end up at any position of the tie, so all of them count"""
    evaluator = RerankingEvaluator(SAMPLES, name="tied", similarity_fct=tied_similarity)

    results = evaluator(stsb_bert_tiny_model)

    assert results["tied_mrr@10"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_RerankingEvaluator_mrr_with_tie_reaching_past_at_k(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Positions beyond at_k do not contribute, even when the tie extends past it"""
    evaluator = RerankingEvaluator(SAMPLES, at_k=2, name="cutoff", similarity_fct=tied_similarity)

    results = evaluator(stsb_bert_tiny_model)

    assert results["cutoff_mrr@2"] == pytest.approx((1 + 1 / 2) / 3)
