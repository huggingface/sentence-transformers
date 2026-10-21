"""
Computes embeddings
"""

from __future__ import annotations

import numpy as np

from sentence_transformers import SentenceTransformer


def test_encode_token_embeddings(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = stsb_bert_tiny_model
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


def test_encode_single_sentences(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    dimension = model.get_embedding_dimension()
    # Single sentence
    emb = model.encode("Hello Word, a test sentence")
    assert emb.shape == (dimension,)
    single_embedding = emb

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"])
    assert emb.shape == (1, dimension)
    np.testing.assert_allclose(emb[0], single_embedding, atol=1e-5)

    # Sentence list
    sentences = ["Hello Word, a test sentence", "Here comes another sentence", "My final sentence"]
    emb = model.encode(sentences)
    assert emb.shape == (3, dimension)
    np.testing.assert_allclose(emb.sum(axis=1), [-5.098289, -4.975013, -6.540779], rtol=0, atol=0.002)
    np.testing.assert_allclose(emb, np.stack([model.encode(sentence) for sentence in sentences]), atol=1e-5)


def test_encode_normalize(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        normalize_embeddings=True,
    )
    assert emb.shape == (3, model.get_embedding_dimension())
    for norm in np.linalg.norm(emb, axis=1):
        assert abs(norm - 1) < 0.001


def test_encode_tuple_sentences(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    dimension = model.get_embedding_dimension()
    # Input a sentence tuple
    emb = model.encode([("Hello Word, a test sentence", "Second input for model")])
    assert emb.shape == (1, dimension)
    single_embedding = emb[0]
    assert not np.allclose(single_embedding, model.encode("Hello Word, a test sentence"))

    # List of sentence tuples
    pairs = [
        ("Hello Word, a test sentence", "Second input for model"),
        ("My second tuple", "With two inputs"),
        ("Final tuple", "final test"),
    ]
    emb = model.encode(pairs)
    assert emb.shape == (3, dimension)
    np.testing.assert_allclose(emb.sum(axis=1), [-5.307772, -6.655279, -6.159927], rtol=0, atol=0.002)
    np.testing.assert_allclose(emb[0], single_embedding, atol=1e-5)
    np.testing.assert_allclose(emb, np.concatenate([model.encode([pair]) for pair in pairs]), atol=1e-5)


def test_encode_routes_through_module_call(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """encode() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = stsb_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.encode("Hello world")
    finally:
        handle.remove()
    assert calls, "encode() should invoke the model via __call__, not call forward() directly"
