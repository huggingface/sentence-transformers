from __future__ import annotations

import copy
import re
from unittest.mock import patch

import pytest
import torch

from sentence_transformers.models import Pooling, Router, Transformer
from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseAutoEncoder, SpladePooling
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from tests.sparse_encoder.utils import sparse_allclose


@pytest.mark.parametrize(
    ("texts", "top_k", "expected_shape"),
    [
        # Single text, default top_k (None)
        (["The weather is nice!"], None, 1),
        # Single text, specific top_k
        (["The weather is nice!"], 3, 1),
        # String text, specific top_k, expect a non-nested list
        ("The weather is nice!", 8, 8),
        # Multiple texts, default top_k (None)
        (["The weather is nice!", "It's sunny outside"], None, 2),
        # Multiple texts, specific top_k
        (["The weather is nice!", "It's sunny outside"], 3, 2),
    ],
)
def test_decode_shapes(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, top_k: int, expected_shape: int
) -> None:
    model = splade_bert_tiny_model
    embeddings = model.encode(texts)
    decoded = model.decode(embeddings, top_k=top_k)

    assert len(decoded) == expected_shape

    if isinstance(texts, list):
        if len(texts) == 1:
            assert isinstance(decoded[0], tuple) or isinstance(decoded, list)
            if top_k is not None:
                assert len(decoded) <= top_k
        else:
            assert isinstance(decoded, list)
            for item in decoded:
                assert isinstance(item, list)
                if top_k is not None:
                    assert len(item) <= top_k


@pytest.mark.parametrize(
    ("text", "expected_token_types"),
    [
        ("The weather is nice!", str),
        ("It's sunny outside", str),
    ],
)
def test_decode_token_types(splade_bert_tiny_model: SparseEncoder, text: str, expected_token_types: type) -> None:
    model = splade_bert_tiny_model
    embeddings = model.encode(text)
    decoded = model.decode(embeddings)

    # Check the first item in the batch
    for token, weight in decoded:
        assert isinstance(token, expected_token_types)
        assert isinstance(weight, float)


@pytest.mark.parametrize(
    ("text", "top_k"),
    [
        ("The weather is nice!", 1),
        ("It's sunny outside", 3),
        ("Hello world", 5),
    ],
)
def test_decode_top_k_respects_limit(splade_bert_tiny_model: SparseEncoder, text: str, top_k: int) -> None:
    model = splade_bert_tiny_model
    embeddings = model.encode([text])
    decoded = model.decode(embeddings, top_k=top_k)

    assert len(decoded) <= top_k


@pytest.mark.parametrize(
    ("texts", "format_type"),
    [
        ("The weather is nice!", "1d"),
        (["The weather is nice!"], "1d"),
        (["The weather is nice!", "It's sunny outside"], "2d"),
    ],
)
def test_decode_handles_sparse_dense_inputs(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, format_type: str
):
    model = splade_bert_tiny_model
    # Get embeddings and test both sparse and dense format handling
    embeddings = model.encode(texts)

    # Test with sparse tensor
    if not embeddings.is_sparse:
        embeddings_sparse = embeddings.to_sparse()
    else:
        embeddings_sparse = embeddings

    decoded_sparse = model.decode(embeddings_sparse)

    # Test with dense tensor
    if embeddings.is_sparse:
        embeddings_dense = embeddings.to_dense()
    else:
        embeddings_dense = embeddings

    decoded_dense = model.decode(embeddings_dense)

    # Verify both produce the same result structure
    if format_type == "1d":
        assert len(decoded_sparse) == len(decoded_dense)
    else:
        assert len(decoded_sparse) == len(decoded_dense)
        for i in range(len(decoded_sparse)):
            # Sort both results to ensure consistent comparison
            sorted_sparse = sorted(decoded_sparse[i], key=lambda x: (x[1], x[0]), reverse=True)
            sorted_dense = sorted(decoded_dense[i], key=lambda x: (x[1], x[0]), reverse=True)
            assert len(sorted_sparse) == len(sorted_dense)


def test_decode_empty_tensor(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model
    # Create an empty sparse tensor
    empty_sparse = torch.sparse_coo_tensor(
        indices=torch.zeros((2, 0), dtype=torch.long),
        values=torch.zeros((0,), dtype=torch.float),
        size=(1, model.get_sentence_embedding_dimension()),
    )

    decoded = model.decode(empty_sparse)
    assert len(decoded) == 0 or (isinstance(decoded, list) and all(not item for item in decoded))


@pytest.mark.parametrize(
    "top_k",
    [None, 5, 1000],
)
@pytest.mark.parametrize(
    "texts",
    [
        ("The weather is nice!"),
        (["The weather is nice!"]),
        (["The weather is nice!", "It's sunny outside", "Hello world"]),
        (["Short text", "This is a longer text with more words to encode"]),
    ],
)
def test_decode_returns_sorted_weights(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, top_k: int | None
) -> None:
    model = splade_bert_tiny_model
    embeddings = model.encode(texts)
    decoded = model.decode(embeddings, top_k=top_k)

    if isinstance(texts, list):
        for item in decoded:
            weights = [weight for _, weight in item]
            assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))
    else:
        weights = [weight for _, weight in decoded]
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_inference_free_splade(inference_free_splade_bert_tiny_model: SparseEncoder):
    model = inference_free_splade_bert_tiny_model
    dimensionality = model.get_sentence_embedding_dimension()

    query = "What is the capital of France?"
    document = "The capital of France is Paris."
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(document)

    assert query_embeddings.shape == (dimensionality,)
    assert document_embeddings.shape == (dimensionality,)

    decoded_query = model.decode(query_embeddings)
    decoded_document = model.decode(document_embeddings)
    assert len(decoded_query) == len(model.tokenize(query, task="query")["input_ids"][0])
    assert len(decoded_document) >= 50

    assert model.max_seq_length == 512
    assert model[0].sub_modules["query"][0].max_seq_length == 512
    assert model[0].sub_modules["document"][0].max_seq_length == 512

    model.max_seq_length = 256
    assert model.max_seq_length == 256
    assert model[0].sub_modules["query"][0].max_seq_length == 256
    assert model[0].sub_modules["document"][0].max_seq_length == 256


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("prompt_name", [None, "query", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
def test_encode_query(
    splade_bert_tiny_model: SparseEncoder,
    sentences: str | list[str],
    prompt_name: str | None,
    prompt: str | None,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
):
    model = splade_bert_tiny_model
    # Create a mock model with required prompts
    model.prompts = {"query": "query: ", "custom": "custom: "}

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_query
        model.encode_query(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=32,
            convert_to_tensor=convert_to_tensor,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
        )

        # Verify that encode was called with the correct parameters
        if prompt_name:
            expected_prompt_name = prompt_name
        elif prompt is not None:
            expected_prompt_name = None
        else:
            expected_prompt_name = "query"

        mock_encode.assert_called_once()
        args, kwargs = mock_encode.call_args

        # Check that sentences were passed correctly
        assert kwargs["sentences"] == sentences

        # Check prompt handling
        assert kwargs["prompt"] == prompt
        assert kwargs["prompt_name"] == expected_prompt_name

        # Check other parameters
        assert kwargs["convert_to_tensor"] == convert_to_tensor
        assert kwargs["convert_to_sparse_tensor"] == convert_to_sparse_tensor
        assert kwargs["task"] == "query"


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("prompt_name", [None, "document", "passage", "corpus", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
def test_encode_document(
    splade_bert_tiny_model: SparseEncoder,
    sentences: str | list[str],
    prompt_name: str | None,
    prompt: str | None,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
):
    # Create a mock model with required prompts
    model = splade_bert_tiny_model
    model.prompts = {"document": "document: ", "passage": "passage: ", "corpus": "corpus: ", "custom": "custom: "}

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_document
        model.encode_document(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=32,
            convert_to_tensor=convert_to_tensor,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
        )

        # Verify that encode was called with the correct parameters
        mock_encode.assert_called_once()
        args, kwargs = mock_encode.call_args

        if prompt_name:
            expected_prompt_name = prompt_name
        elif prompt is not None:
            expected_prompt_name = None
        else:
            expected_prompt_name = "document"

        # Check that sentences were passed correctly
        assert kwargs["sentences"] == sentences

        # Check prompt handling
        assert kwargs["prompt"] == prompt
        assert kwargs["prompt_name"] == expected_prompt_name

        # Check other parameters
        assert kwargs["convert_to_tensor"] == convert_to_tensor
        assert kwargs["convert_to_sparse_tensor"] == convert_to_sparse_tensor
        assert kwargs["task"] == "document"


def test_encode_document_prompt_priority(splade_bert_tiny_model: SparseEncoder):
    """Test that proper prompt priority is respected when multiple options are available"""
    model = splade_bert_tiny_model
    model.prompts = {
        "document": "document: ",
        "passage": "passage: ",
        "corpus": "corpus: ",
    }

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_document with no explicit prompt
        model.encode_document("test")

        # It should select "document" by default since that's first in the priority list
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "document"

        # Remove document, should fall back to passage
        mock_encode.reset_mock()
        model.prompts = {
            "passage": "passage: ",
            "corpus": "corpus: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "passage"

        # Remove passage, should fall back to corpus
        mock_encode.reset_mock()
        model.prompts = {
            "corpus": "corpus: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "corpus"

        # No relevant prompts defined
        mock_encode.reset_mock()
        model.prompts = {
            "query": "query: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] is None


def test_encode_advanced_parameters(splade_bert_tiny_model: SparseEncoder):
    """Test that additional parameters are correctly passed to encode"""
    model = splade_bert_tiny_model

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call with advanced parameters
        model.encode_query(
            "test",
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
            max_active_dims=128,
            chunk_size=10,
            custom_param="value",
        )

        # Verify all parameters were passed correctly
        args, kwargs = mock_encode.call_args
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["batch_size"] == 64
        assert kwargs["show_progress_bar"] is True
        assert kwargs["max_active_dims"] == 128
        assert kwargs["chunk_size"] == 10
        assert kwargs["custom_param"] == "value"


@pytest.mark.parametrize("inputs", ["test sentence", ["test sentence"]])
def test_encode_query_document_vs_encode(splade_bert_tiny_model: SparseEncoder, inputs: str | list[str]):
    """Test the actual integration with encode vs encode_query/encode_document"""
    # This test requires a real model, but we'll use a small one
    model = splade_bert_tiny_model
    model.prompts = {"query": "query: ", "document": "document: "}

    # Get embeddings with encode_query and encode_document
    query_embeddings = model.encode_query(inputs)
    document_embeddings = model.encode_document(inputs)

    # And the same but with encode via prompts (task doesn't help here)
    encode_query_embeddings = model.encode(inputs, prompt_name="query")
    encode_document_embeddings = model.encode(inputs, prompt_name="document")

    # With prompts they should be the same
    assert sparse_allclose(query_embeddings, encode_query_embeddings)
    assert sparse_allclose(document_embeddings, encode_document_embeddings)

    # Without prompts they should be different
    query_embeddings_without_prompt = model.encode(inputs)
    document_embeddings_without_prompt = model.encode(inputs)

    # Embeddings should differ when different prompts are used
    assert not sparse_allclose(query_embeddings_without_prompt, query_embeddings)
    assert not sparse_allclose(document_embeddings_without_prompt, document_embeddings)


def test_default_prompt(splade_bert_tiny_model: SparseEncoder):
    """Test that the default prompt is used when no prompt is specified"""
    model = splade_bert_tiny_model
    model.prompts = {"query": "query: ", "document": "document: "}
    model.default_prompt_name = "query"

    # Call encode_query without specifying a prompt
    query_embeddings = model.encode_query("test")
    assert query_embeddings.shape == (model.get_sentence_embedding_dimension(),)

    # Call encode_document without specifying a prompt
    document_embeddings = model.encode_document("test")
    assert document_embeddings.shape == (model.get_sentence_embedding_dimension(),)

    default_embeddings = model.encode("test")
    assert default_embeddings.shape == (model.get_sentence_embedding_dimension(),)

    # Make sure that the default prompt is used
    assert sparse_allclose(query_embeddings, default_embeddings)
    assert not sparse_allclose(document_embeddings, default_embeddings)

    # Also check that if the default prompt is not set, the default embeddings aren't the same as query
    model.default_prompt_name = None
    default_embeddings_no_default = model.encode("test")
    assert not sparse_allclose(default_embeddings_no_default, default_embeddings)


def test_wrong_prompt(splade_bert_tiny_model: SparseEncoder):
    """Test that using a wrong prompt raises an error"""
    model = splade_bert_tiny_model
    model.prompts = {"query": "query: ", "document": "document: "}

    for encode_method in [model.encode_query, model.encode_document, model.encode]:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Prompt name 'invalid_prompt' not found in the configured prompts dictionary with keys ['query', 'document']."
            ),
        ):
            encode_method("test", prompt_name="invalid_prompt")


def test_max_active_dims_set_init(splade_bert_tiny_model: SparseEncoder, csr_bert_tiny_model: SparseEncoder, tmp_path):
    splade_bert_tiny_model.save_pretrained(str(tmp_path / "splade_bert_tiny"))
    csr_bert_tiny_model.save_pretrained(str(tmp_path / "csr_bert_tiny"))

    # Load the models with max_active_dims set
    loaded_model = SparseEncoder(str(tmp_path / "splade_bert_tiny"))
    assert loaded_model.max_active_dims is None
    loaded_model = SparseEncoder(str(tmp_path / "splade_bert_tiny"), max_active_dims=13)
    assert loaded_model.max_active_dims == 13

    loaded_model = SparseEncoder(str(tmp_path / "csr_bert_tiny"))
    assert loaded_model.max_active_dims == 16  # Based on the SparseAutoEncoder's k value
    loaded_model = SparseEncoder(str(tmp_path / "csr_bert_tiny"), max_active_dims=13)
    assert loaded_model.max_active_dims == 13


def test_detect_mlm():
    model = SparseEncoder("distilbert/distilbert-base-uncased")

    assert isinstance(model[0], MLMTransformer)
    assert isinstance(model[1], SpladePooling)


def test_default_to_csr():
    # NOTE: bert-tiny is actually MLM-based, but the config isn't modern enough to allow us to detect it,
    # so we should default to CSR here.
    model = SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert isinstance(model[0], Transformer)
    assert isinstance(model[1], Pooling)
    assert isinstance(model[2], SparseAutoEncoder)


def test_sparsity(splade_bert_tiny_model: SparseEncoder):
    model = splade_bert_tiny_model

    # Check that the sparsity is applied correctly
    embeddings = model.encode_query(["What is the capital of France?", "Who has won the World Cup in 2016?"])
    sparsity = model.sparsity(embeddings)
    assert isinstance(sparsity, dict)
    assert "active_dims" in sparsity
    assert "sparsity_ratio" in sparsity
    assert sparsity["active_dims"] < 100 and sparsity["active_dims"] > 0
    assert sparsity["sparsity_ratio"] < 1.0 and sparsity["sparsity_ratio"] >= 0.99

    # Also check with dense tensors
    dense_sparsity = model.sparsity(embeddings.to_dense())
    assert dense_sparsity == sparsity, "Sparsity should be the same for dense and sparse tensors"

    # Check that 1-dimensional embeddings work correctly
    sparsity_one = model.sparsity(embeddings[0])
    sparsity_two = model.sparsity(embeddings[1])
    assert (sparsity_one["active_dims"] + sparsity_two["active_dims"]) / 2 == sparsity["active_dims"]


def test_splade_pooling_chunk_size(splade_bert_tiny_model: SparseEncoder):
    model = splade_bert_tiny_model

    # The chunk size defaults to None, i.e. no chunking
    assert model.splade_pooling_chunk_size is None
    # But we can chunk the pooling to save memory at the cost of some speed
    model.splade_pooling_chunk_size = 13
    assert model.splade_pooling_chunk_size == 13
    assert isinstance(model[1], SpladePooling)
    assert model[1].chunk_size == 13


def test_intersection(splade_bert_tiny_model: SparseEncoder):
    model = splade_bert_tiny_model

    # Test intersection with a single text
    query = "Where can I deposit my money?"
    document = "I'm sitting by the river."
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(document)
    query_sparsity = model.sparsity(query_embeddings)
    document_sparsity = model.sparsity(document_embeddings)

    # Let's check that the intersection is a tensor and has the correct shape
    intersection = model.intersection(query_embeddings, document_embeddings)
    assert isinstance(intersection, torch.Tensor)
    assert intersection.shape == (model.get_sentence_embedding_dimension(),)

    # Check that the intersection sparsity is less than both query and document sparsities
    intersection_sparsity = model.sparsity(intersection)
    assert (
        intersection_sparsity["active_dims"] < query_sparsity["active_dims"]
        and intersection_sparsity["active_dims"] < document_sparsity["active_dims"]
    )

    # Test with multiple texts
    query = "Who has won the World Cup in 2016?"
    documents = ["The capital of France is Paris.", "Germany won the World Cup in 2014."]
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(documents)

    intersection_batch = model.intersection(query_embeddings, document_embeddings)
    assert isinstance(intersection_batch, torch.Tensor)
    assert intersection_batch.shape == (len(documents), model.get_sentence_embedding_dimension())

    decoded_intersection_batch = model.decode(intersection_batch)
    assert len(decoded_intersection_batch) == len(documents)


def test_encode_with_dataset_column(splade_bert_tiny_model: SparseEncoder) -> None:
    """Test that encode can handle a dataset column as input."""
    model = splade_bert_tiny_model
    from datasets import Dataset

    # Create a simple dataset with a text column
    dataset = Dataset.from_dict({"text": ["This is a test.", "Another sentence."]})

    # Encode the dataset column
    embeddings = model.encode(dataset["text"], convert_to_tensor=True)

    # Check the shape of the embeddings
    assert embeddings.shape == (2, model.get_sentence_embedding_dimension())


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
@pytest.mark.parametrize("save_to_cpu", [True, False])
@pytest.mark.parametrize("max_active_dims", [None, 64, 128])
def test_empty_encode(
    splade_bert_tiny_model: SparseEncoder,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
    save_to_cpu: bool,
    max_active_dims: int | None,
):
    model = splade_bert_tiny_model
    embeddings = model.encode(
        [],
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
        save_to_cpu=save_to_cpu,
        max_active_dims=max_active_dims,
    )

    if convert_to_tensor:
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.numel() == 0
        if save_to_cpu:
            assert embeddings.device == torch.device("cpu")
        else:
            assert embeddings.device == model.device

        if convert_to_sparse_tensor:
            assert embeddings.is_sparse
        else:
            assert not embeddings.is_sparse
    else:
        assert embeddings == []


def test_get_model_kwargs(splade_bert_tiny_model: SparseEncoder) -> None:
    """Test that get_model_kwargs returns the correct keyword arguments."""
    model = splade_bert_tiny_model

    # Check that the forward kwargs are as expected, i.e. no extra forward kwargs
    # for this basic model
    forward_kwargs = model.get_model_kwargs()
    assert forward_kwargs == []
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), this model does not accept "
            "any additional keyword arguments."
        ),
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")

    # If one of the modules has additional forward kwargs, they should be included
    model[0].forward_kwargs = {"foo"}
    model[1].forward_kwargs = {"bar", "baz"}
    assert set(model.get_model_kwargs()) == {"foo", "bar", "baz"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'bar'|'baz'|, ){5}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r"(MLMTransformer\.)?forward\(\) got an unexpected keyword argument '(foo|bar)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", foo=True, bar=False)

    # And also if we have a Router in place
    query_pooling_copy = copy.deepcopy(model[1])
    query_pooling_copy.forward_kwargs = {"query_arg"}
    document_pooling_copy = copy.deepcopy(model[1])
    document_pooling_copy.forward_kwargs = {"document_arg_1", "document_arg_2"}
    model[1] = Router.for_query_document(
        query_modules=[query_pooling_copy],
        document_modules=[document_pooling_copy],
    )
    assert set(model.get_model_kwargs()) == {"foo", "task", "query_arg", "document_arg_1", "document_arg_2"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'task'|'query_arg'|'document_arg_1'|'document_arg_2'|, ){9}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", task="query", normalize=True)
    # This should run fine
    model.encode("Test sentence", task="document")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r"(MLMTransformer\.)?forward\(\) got an unexpected keyword argument '(foo|document_arg_1)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", task="document", foo=True, document_arg_1=12)
