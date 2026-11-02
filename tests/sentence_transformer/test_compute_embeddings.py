"""
Computes embeddings
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_encode_token_embeddings(stsb_bert_tiny_model: SentenceTransformer, padding_side: str) -> None:
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = stsb_bert_tiny_model
    model.tokenizer.padding_side = padding_side
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


def test_encode_empty_query_batch(word_embeddings_model: SentenceTransformer) -> None:
    model = word_embeddings_model.to("cpu")
    corpus = model.encode(["hello", "world"])
    queries = model.encode([])

    assert semantic_search(queries, corpus) == []
    assert queries.shape == (0, 300)


@pytest.mark.parametrize("encode_method", ["encode", "encode_query", "encode_document"])
@pytest.mark.parametrize("convert_to_tensor", [False, True])
@pytest.mark.parametrize("device", [None, ["cpu"]])
@pytest.mark.parametrize(
    ("precision", "expected_dim", "expected_dtype"),
    [
        ("float32", 300, np.float32),
        ("int8", 300, np.int8),
        ("uint8", 300, np.uint8),
        ("binary", 38, np.int8),
        ("ubinary", 38, np.uint8),
    ],
)
def test_empty_encode_output_format(
    word_embeddings_model: SentenceTransformer,
    encode_method: str,
    convert_to_tensor: bool,
    device: list[str] | None,
    precision: str,
    expected_dim: int,
    expected_dtype: type[np.generic],
) -> None:
    model = word_embeddings_model.to("cpu")
    embeddings = getattr(model, encode_method)(
        [], convert_to_tensor=convert_to_tensor, device=device, precision=precision
    )

    assert embeddings.shape == (0, expected_dim)
    if convert_to_tensor:
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.device.type == "cpu"
        embeddings = embeddings.numpy()
    else:
        assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype == expected_dtype


@pytest.mark.parametrize("convert_to_tensor", [False, True])
@pytest.mark.parametrize("precision", ["float32", "binary"])
@pytest.mark.parametrize(
    ("model_truncate_dim", "truncate_dim", "expected_dim", "expected_binary_dim"),
    [(None, 10, 10, 2), (10, None, 10, 2), (5, 12, 12, 2), (12, 5, 5, 1), (None, 400, 300, 38), (None, 0, 0, 0)],
)
def test_empty_encode_truncate_dim(
    word_embeddings_model: SentenceTransformer,
    convert_to_tensor: bool,
    precision: str,
    model_truncate_dim: int | None,
    truncate_dim: int | None,
    expected_dim: int,
    expected_binary_dim: int,
) -> None:
    model = word_embeddings_model.to("cpu")
    with model.truncate_embeddings(model_truncate_dim):
        embeddings = model.encode(
            [], convert_to_tensor=convert_to_tensor, truncate_dim=truncate_dim, precision=precision
        )

    assert embeddings.shape == (0, expected_binary_dim if precision == "binary" else expected_dim)


@pytest.mark.parametrize("device", [None, ["cpu"]])
@pytest.mark.parametrize(
    ("output_value", "convert_to_numpy"), [("sentence_embedding", False), ("token_embeddings", True), (None, True)]
)
def test_empty_encode_list_output(
    word_embeddings_model: SentenceTransformer,
    device: list[str] | None,
    output_value: str | None,
    convert_to_numpy: bool,
) -> None:
    model = word_embeddings_model.to("cpu")
    assert model.encode([], output_value=output_value, convert_to_numpy=convert_to_numpy, device=device) == []
