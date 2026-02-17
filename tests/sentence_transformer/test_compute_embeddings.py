"""
Computes embeddings
"""

from __future__ import annotations

import numpy as np

from sentence_transformers import SentenceTransformer


def test_encode_token_embeddings(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = paraphrase_distilroberta_base_v1_model
    sent = [
        "Hello Word, a test sentence",
        "Here comes another sentence",
        "My final sentence",
        "Sentences",
        "Sentence five five five five five five five",
    ]
    emb = model.encode(sent, output_value="token_embeddings", batch_size=2)
    assert len(emb) == len(sent)

    for s, e in zip(sent, emb):
        assert len(model.preprocess([s])["input_ids"][0]) == e.shape[0]


def test_encode_single_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence")
    assert emb.shape == (768,)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 22.968266) < 0.007


def test_encode_normalize(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        normalize_embeddings=True,
    )
    assert emb.shape == (3, 768)
    for norm in np.linalg.norm(emb, axis=1):
        assert abs(norm - 1) < 0.001


def test_encode_tuple_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Input a sentence tuple
    emb = model.encode([("Hello Word, a test sentence", "Second input for model")])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 9.503508) < 0.002

    # List of sentence tuples
    emb = model.encode(
        [
            ("Hello Word, a test sentence", "Second input for model"),
            ("My second tuple", "With two inputs"),
            ("Final tuple", "final test"),
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 32.14627) < 0.002
