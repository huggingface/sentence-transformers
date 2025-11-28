from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer

# These tests fail if optimum.intel.openvino is imported, because openvinotoolkit/nncf
# patches torch._C._nn.gelu in a way that breaks pickling. As a result, we may have issues
# when running both backend tests and multi-process tests in the same session.


@pytest.mark.slow
@pytest.mark.parametrize("normalize_embeddings", (False, True))
@pytest.mark.parametrize("prompt_name", (None, "retrieval"))
def test_encode_multi_process(
    stsb_bert_tiny_model: SentenceTransformer, normalize_embeddings: bool, prompt_name: str | None
) -> None:
    model = stsb_bert_tiny_model
    model.prompts = {"retrieval": "Represent this sentence for searching relevant passages: "}
    sentences = [f"This is sentence {i}" for i in range(40)]

    # Start the multi-process pool on e.g. two CPU devices & compute the embeddings using the pool
    pool = model.start_multi_process_pool(["cpu", "cpu"])
    emb = model.encode(
        sentences, normalize_embeddings=normalize_embeddings, prompt_name=prompt_name, pool=pool, chunk_size=10
    )
    model.stop_multi_process_pool(pool)
    assert emb.shape == (len(sentences), 128)

    # Make sure the embeddings aren't just all 0
    assert emb.sum() != 0.0

    # Compare against normal embeddings
    emb_normal = model.encode(sentences, normalize_embeddings=normalize_embeddings, prompt_name=prompt_name)
    diff = np.max(np.abs(emb - emb_normal))
    assert diff < 0.001

    # Ensure that after normalizing, the means are all almost 0, and otherwise not
    assert np.all(np.abs(emb.mean(1)) < 0.01) == normalize_embeddings


@pytest.mark.slow
def test_multi_process_encode_same_as_standard_encode(stsb_bert_tiny_model: SentenceTransformer):
    model = stsb_bert_tiny_model
    # Test that multi-process encoding gives the same result as standard encoding
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 5

    # Standard encode
    embeddings_standard = model.encode(texts)

    # Multi-process encode with device=["cpu"] * 2
    embeddings_multi = model.encode(texts, device=["cpu"] * 2)

    # Should produce the same embeddings
    assert np.allclose(embeddings_standard, embeddings_multi, atol=1e-6)


@pytest.mark.slow
def test_multi_process_pool(stsb_bert_tiny_model: SentenceTransformer):
    # Test the start_multi_process_pool and stop_multi_process_pool functions
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 5

    # Standard encode
    embeddings_standard = model.encode(texts)

    pool = model.start_multi_process_pool(["cpu"] * 2)
    try:
        # Encode using the pool
        embeddings_multi = model.encode(texts, pool=pool)

    finally:
        model.stop_multi_process_pool(pool)

    # Should be numpy array with correct shape and the same embeddings
    assert isinstance(embeddings_multi, np.ndarray)
    assert embeddings_multi.shape == (len(texts), model.get_sentence_embedding_dimension())
    assert np.allclose(embeddings_standard, embeddings_multi, atol=1e-6)


@pytest.mark.slow
def test_multi_process_with_args(stsb_bert_tiny_model: SentenceTransformer):
    # Test multi-process encoding with various arguments
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence."]

    # Create a pool
    pool = model.start_multi_process_pool(["cpu"] * 2)

    try:
        # Test with normalize_embeddings and convert_to_tensor
        embeddings = model.encode(texts, pool=pool, normalize_embeddings=True, convert_to_tensor=True)

        # Should be a tensor with normalized vectors
        assert isinstance(embeddings, torch.Tensor)
        # Verify that embeddings are normalized (unit vectors) when normalize_embeddings=True
        norm = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

        # Test with precision options
        embeddings_int8 = model.encode(texts, pool=pool, precision="int8")

        # Should be quantized
        assert embeddings_int8.dtype == np.int8
    finally:
        model.stop_multi_process_pool(pool)


@pytest.mark.slow
def test_multi_process_output_values(stsb_bert_tiny_model: SentenceTransformer):
    # Test that different output_value options work with multi-process
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence."]

    # Regular encoding with output_value=None
    embeddings_standard = model.encode(texts, output_value=None)

    # Multi-process encoding with output_value=None
    embeddings_multi = model.encode(texts, device=["cpu"] * 2, output_value=None)

    # Both should return a list of dictionaries
    assert isinstance(embeddings_standard, list)
    assert isinstance(embeddings_multi, list)
    assert isinstance(embeddings_standard[0], dict)
    assert isinstance(embeddings_multi[0], dict)
    assert "sentence_embedding" in embeddings_standard[0]
    assert "sentence_embedding" in embeddings_multi[0]

    # Make sure the sentence embeddings match
    for i in range(len(texts)):
        assert torch.allclose(
            embeddings_standard[i]["sentence_embedding"].cpu(),
            embeddings_multi[i]["sentence_embedding"],
            atol=1e-6,
        )


@pytest.mark.slow
def test_multi_process_chunk_size(stsb_bert_tiny_model: SentenceTransformer):
    # Test explicit chunk_size parameter
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 10

    # Test with explicit chunk size
    embeddings = model.encode(texts, device=["cpu"] * 2, chunk_size=5)

    # Should produce correct embeddings
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())


@pytest.mark.slow
def test_multi_process_with_prompt(stsb_bert_tiny_model: SentenceTransformer):
    # Test multi-process encoding with prompts
    model = stsb_bert_tiny_model
    model.prompts = {"retrieval": "Represent this sentence for searching relevant passages: "}
    texts = ["First sentence.", "Second sentence."] * 5

    standard_embeddings = model.encode(texts, prompt_name="retrieval", normalize_embeddings=True)

    # Create a pool
    pool = model.start_multi_process_pool(["cpu"] * 2)

    try:
        # Encode with prompt
        multi_embeddings = model.encode(texts, pool=pool, prompt_name="retrieval", normalize_embeddings=True)
    finally:
        model.stop_multi_process_pool(pool)

    # Should be a numpy array with correct shape
    assert isinstance(multi_embeddings, np.ndarray)
    assert multi_embeddings.shape == (len(texts), 128)

    # Verify normalization
    norm = np.linalg.norm(multi_embeddings, axis=1)
    assert np.allclose(norm, 1.0, atol=1e-6)

    # Compare with standard encoding
    assert np.allclose(standard_embeddings, multi_embeddings, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("output_value", [None, "sentence_embedding", "token_embeddings"])
def test_multi_process_with_empty_texts(
    stsb_bert_tiny_model: SentenceTransformer,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    output_value: str | None,
):
    # Test encoding with empty texts
    model = stsb_bert_tiny_model
    texts = []

    # Encode with empty texts
    standard_embeddings = model.encode(
        texts, convert_to_tensor=convert_to_tensor, convert_to_numpy=convert_to_numpy, output_value=output_value
    )
    multi_embeddings = model.encode(
        texts,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        output_value=output_value,
    )

    # Should return empty arrays, identical types as without multi-processing
    assert type(standard_embeddings) is type(multi_embeddings)
    assert len(standard_embeddings) == 0
    assert len(multi_embeddings) == 0


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("output_value", [None, "sentence_embedding", "token_embeddings"])
def test_multi_process_with_one_single_string(
    stsb_bert_tiny_model: SentenceTransformer,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    output_value: str | None,
):
    # Test encoding with a single text
    model = stsb_bert_tiny_model
    texts = "This is a single sentence."

    # Encode with single text
    standard_embeddings = model.encode(
        texts, convert_to_tensor=convert_to_tensor, convert_to_numpy=convert_to_numpy, output_value=output_value
    )
    multi_embeddings = model.encode(
        texts,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        output_value=output_value,
    )

    # Assert that the embeddings are the same type and shape
    assert type(standard_embeddings) is type(multi_embeddings)
    if isinstance(standard_embeddings, (np.ndarray, torch.Tensor)):
        assert standard_embeddings.shape == multi_embeddings.shape
    else:
        assert len(standard_embeddings) == len(multi_embeddings)
        # Check that dictionary items are the same
        if isinstance(standard_embeddings, dict):
            assert standard_embeddings.keys() == multi_embeddings.keys()
            for key in standard_embeddings:
                if isinstance(standard_embeddings[key], torch.Tensor):
                    assert torch.allclose(standard_embeddings[key].cpu(), multi_embeddings[key], atol=1e-5)
                elif isinstance(standard_embeddings[key], np.ndarray):
                    assert np.allclose(standard_embeddings[key], multi_embeddings[key], atol=1e-5)
                else:
                    assert standard_embeddings[key] == multi_embeddings[key]
        elif isinstance(standard_embeddings, list) and len(standard_embeddings) > 0:
            for std_item, multi_item in zip(standard_embeddings, multi_embeddings):
                assert set(std_item.keys()) == set(multi_item.keys())
                for key in std_item:
                    if isinstance(std_item[key], torch.Tensor):
                        assert torch.allclose(std_item[key].cpu(), multi_item[key], atol=1e-5)
                    elif isinstance(std_item[key], np.ndarray):
                        assert np.allclose(std_item[key], multi_item[key], atol=1e-5)
                    else:
                        assert std_item[key] == multi_item[key]


@pytest.mark.slow
def test_multi_process_more_workers_than_texts(stsb_bert_tiny_model: SentenceTransformer):
    # Test with more workers than texts
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence."]

    embeddings = model.encode(texts, device=["cpu"] * 3)

    # Should be numpy array with correct shape
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())


@pytest.mark.slow
def test_multi_process_with_large_chunk_size(stsb_bert_tiny_model: SentenceTransformer):
    # Test with a large chunk size
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence."] * 10  # 20 sentences

    # Use a large chunk size
    embeddings = model.encode(texts, device=["cpu"] * 2, chunk_size=30)

    # Should produce correct embeddings
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available to experiment with 2 separate devices"
)
def test_multi_process_output_tensors_two_devices(stsb_bert_tiny_model: SentenceTransformer):
    # Test with two separate devices
    model = stsb_bert_tiny_model
    texts = ["First sentence.", "Second sentence."]

    # Ensure that embeddings are moved to CPU so they can be concatenated
    embeddings = model.encode(texts, device=["cpu", "cuda"], convert_to_tensor=True)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.device.type == "cpu"
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())

    # But the default is still just numpy
    embeddings = model.encode(texts, device=["cpu", "cuda"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())
