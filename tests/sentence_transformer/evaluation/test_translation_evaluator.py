"""
Tests the correct computation of evaluation scores from TranslationEvaluator
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util import cos_sim


@pytest.fixture
def translation_word_model() -> SentenceTransformer:
    vocab = ["pad", "alpha", "beta", "gamma", "alias", "delta"]
    weights = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    model = SentenceTransformer(modules=[WordEmbeddings(WhitespaceTokenizer(vocab=vocab), weights), Pooling(3)])
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.mark.parametrize("num_sentences", [17, 1024, 2051])
def test_translation_blocks_match_full_matrix(translation_word_model, num_sentences):
    words = ["alpha", "beta", "gamma", "alias"]
    sources = [words[i % 4] for i in range(num_sentences)]
    targets = [words[(3 * i + 1) % 4] for i in range(num_sentences)]
    sources[-1] = targets[0] = "delta"
    model = translation_word_model
    scores = cos_sim(model.encode(sources), model.encode(targets))
    expected_forward = (
        sum(targets[i] == targets[j] for i, j in enumerate(scores.argmax(dim=1).tolist())) / num_sentences
    )
    expected_reverse = (
        sum(sources[i] == sources[j] for i, j in enumerate(scores.argmax(dim=0).tolist())) / num_sentences
    )

    metrics = TranslationEvaluator(sources, targets)(model)

    assert metrics["src2trg_accuracy"] == expected_forward
    assert metrics["trg2src_accuracy"] == expected_reverse


def test_translation_prints_wrong_matches_after_block_boundary(translation_word_model, capsys):
    sources = ["alpha"] * 1025 + ["beta"]
    targets = ["alpha"] * 1024 + ["beta", "alpha"]
    metrics = TranslationEvaluator(sources, targets, print_wrong_matches=True)(translation_word_model)

    assert metrics["src2trg_accuracy"] == 1024 / 1026
    output = capsys.readouterr().out
    assert "Source 1024 is most similar to target 0 instead of target 1024" in output
    assert "Source 1025 is most similar to target 1024 instead of target 1025" in output


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
