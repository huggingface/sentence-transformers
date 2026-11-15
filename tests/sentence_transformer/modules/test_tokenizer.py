from __future__ import annotations

import pytest

from sentence_transformers.sentence_transformer.modules.tokenizer import PhraseTokenizer, WhitespaceTokenizer

VOCAB = ["Cairo", "القاهرة", "مصر"]

PUNCTUATED_TEXTS = [
    ("Cairo,", [0]),
    ("$Cairo$", [0]),  # ASCII symbols in string.punctuation are still stripped
    ("القاهرة،", [1]),  # Arabic comma
    ("القاهرة، مصر؟", [1, 2]),  # Arabic comma and question mark
    ("مصر؛", [2]),  # Arabic semicolon
    ("«Cairo»", [0]),  # guillemets
    ("مصر。", [2]),  # CJK full stop
    ("،؟", []),  # only punctuation
]


@pytest.mark.parametrize(("text", "expected"), PUNCTUATED_TEXTS)
def test_whitespace_tokenizer_strips_unicode_punctuation(text: str, expected: list[int]) -> None:
    tokenizer = WhitespaceTokenizer(vocab=VOCAB, stop_words=[])
    assert tokenizer.tokenize(text) == expected


@pytest.mark.parametrize(("text", "expected"), PUNCTUATED_TEXTS)
def test_phrase_tokenizer_strips_unicode_punctuation(text: str, expected: list[int]) -> None:
    pytest.importorskip("nltk")
    tokenizer = PhraseTokenizer(vocab=VOCAB, stop_words=[])
    assert tokenizer.tokenize(text) == expected
