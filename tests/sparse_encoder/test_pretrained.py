from __future__ import annotations

import pytest
from torch import Tensor

from sentence_transformers import SparseEncoder


@pytest.mark.parametrize(
    "model_name",
    [
        ("sentence-transformers/all-MiniLM-L6-v2"),
    ],
)
def test_load_and_encode(model_name: str) -> None:
    # Ensure that SparseEncoder can be initialized with a base model and can encode
    try:
        model = SparseEncoder(model_name)
    except Exception as e:
        pytest.fail(f"Failed to load SparseEncoder with {model_name}: {e}")

    sentences = [
        "This is a test sentence.",
        "Another example sentence here.",
        "Sparse encoders are interesting.",
    ]

    try:
        embeddings = model.encode(sentences)
    except Exception as e:
        pytest.fail(f"SparseEncoder failed to encode sentences: {e}")

    assert embeddings is not None

    assert isinstance(embeddings, Tensor), "Embeddings should be a tensor for sparse encoders"
    assert len(embeddings) == len(sentences), "Number of embeddings should match number of sentences"

    decoded_embeddings = model.decode(embeddings)
    assert len(decoded_embeddings) == len(sentences), "Decoded embeddings should match number of sentences"
    assert all(isinstance(emb, list) for emb in decoded_embeddings), "Decoded embeddings should be a list of lists"

    # Check a known property: encoding a single sentence
    single_sentence_emb = model.encode(["A single sentence."], convert_to_tensor=False)
    assert isinstance(single_sentence_emb, list), (
        "Encoding a single sentence with convert_to_tensor=False should return a list of len 1"
    )
    assert len(single_sentence_emb) == 1, "Single sentence embedding dict should not be empty"

    # If we're using a string instead of a list, we should get a single tensor embedding
    single_sentence_emb_tensor = model.encode("A single sentence.", convert_to_tensor=False)
    assert isinstance(single_sentence_emb_tensor, Tensor), (
        "Encoding a single sentence with convert_to_tensor=False should return a tensor"
    )
    assert single_sentence_emb_tensor.dim() == 1, "Single sentence embedding tensor should be 1D"

    # Check encoding with show_progress_bar
    try:
        embeddings_with_progress = model.encode(sentences, show_progress_bar=True)
        assert len(embeddings_with_progress) == len(sentences)
    except Exception as e:
        pytest.fail(f"SparseEncoder failed to encode with progress bar: {e}")
