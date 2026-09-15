"""
Tests the correct computation of evaluation scores from TranslationEvaluator
"""

from __future__ import annotations

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
    """Allowing ties must not credit a pair whose own similarity is below the row maximum."""
    model = stsb_bert_tiny_model
    sources = ["the cat sat on the mat", "a dog barked loudly"]
    targets = ["a dog barked loudly", "the cat sat on the mat"]

    evaluator = TranslationEvaluator(source_sentences=sources, target_sentences=targets)
    metrics = evaluator(model)

    assert metrics["src2trg_accuracy"] == 0.0
    assert metrics["trg2src_accuracy"] == 0.0
