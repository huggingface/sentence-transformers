"""
Tests the correct computation of evaluation scores from TranslationEvaluator
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator


def test_repeated_sentences_count_as_correct_matches(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A repeated sentence ties with its own copy, and np.argmax takes the lowest index, so every
    later copy was counted wrong. Both indices are correct: the sentences are the same string."""
    model = stsb_bert_tiny_model
    sentences = ["the cat sat on the mat", "hello there", "a dog barked loudly", "hello there"]

    evaluator = TranslationEvaluator(source_sentences=sentences, target_sentences=sentences)
    metrics = evaluator(model)

    assert metrics["src2trg_accuracy"] == 1.0
    assert metrics["trg2src_accuracy"] == 1.0


def test_mismatched_pairs_still_count_as_wrong(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    sources = ["the cat sat on the mat", "a dog barked loudly"]
    targets = ["a dog barked loudly", "the cat sat on the mat"]

    evaluator = TranslationEvaluator(source_sentences=sources, target_sentences=targets)
    metrics = evaluator(model)

    assert metrics["src2trg_accuracy"] == 0.0
    assert metrics["trg2src_accuracy"] == 0.0


@pytest.mark.parametrize(
    ("sources", "targets", "expected_src2trg", "expected_trg2src"),
    [
        (["hi", "hello", "bye"], ["salut", "bonjour", "au revoir"], 1 / 3, 1 / 3),
        (["hello", "hello", "bye"], ["salut", "bonjour", "au revoir"], 1 / 3, 2 / 3),
        (["hi", "hello", "bye"], ["salut", "salut", "au revoir"], 2 / 3, 1 / 3),
    ],
)
def test_tied_scores_only_credit_matching_text(sources, targets, expected_src2trg, expected_trg2src):
    model = Mock()
    model.encode.return_value = [torch.ones(2) for _ in sources]

    metrics = TranslationEvaluator(source_sentences=sources, target_sentences=targets)(model)

    assert metrics["src2trg_accuracy"] == expected_src2trg
    assert metrics["trg2src_accuracy"] == expected_trg2src
