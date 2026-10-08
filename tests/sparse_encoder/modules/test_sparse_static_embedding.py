from __future__ import annotations

import json
from pathlib import Path

import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import Router, SparseStaticEmbedding
from tests.sparse_encoder.utils import sparse_allclose


def test_sparse_static_embedding_padding_ignored(inference_free_splade_bert_tiny_model: SparseEncoder) -> None:
    model = inference_free_splade_bert_tiny_model

    input_texts = ["This is a test input", "This is a considerably longer test input to check padding behavior."]

    # Encode the input texts
    batch_embeddings = model.encode_query(input_texts, save_to_cpu=True)

    single_embeddings = [model.encode_query(text, save_to_cpu=True) for text in input_texts]
    single_embeddings = torch.stack(single_embeddings)

    # Check that the batch embeddings match the single embeddings
    assert sparse_allclose(batch_embeddings, single_embeddings, atol=1e-6), (
        "Batch encoding does not match single encoding."
    )


def test_sparse_static_embedding_save_load(
    inference_free_splade_bert_tiny_model: SparseEncoder, tmp_path: Path
) -> None:
    model = inference_free_splade_bert_tiny_model

    assert isinstance(model[0].sub_modules.query[0], SparseStaticEmbedding), "SparseStaticEmbedding component missing"

    # Let's randomize the weights to ensure that we can check if they are maintained after saving and loading
    model[0].sub_modules.query[0].weight == torch.rand_like(model[0].sub_modules.query[0].weight)

    # Define test inputs
    test_inputs = ["This is a simple test.", "Another example text for testing."]

    # Get embeddings before saving
    original_embeddings = model.encode_query(test_inputs, save_to_cpu=True)

    # Save the model
    save_path = tmp_path / "test_sparse_static_embedding_model"
    model.save_pretrained(save_path)

    # Load the model
    loaded_model = SparseEncoder(str(save_path))

    # Get embeddings after loading
    loaded_embeddings = loaded_model.encode_query(test_inputs, save_to_cpu=True)

    # Check if embeddings are the same before and after save/load
    assert sparse_allclose(original_embeddings, loaded_embeddings, atol=1e-6), "Embeddings changed after save and load"

    # Check if SparseStaticEmbedding weights are maintained after loading
    assert isinstance(loaded_model[0].sub_modules.query[0], SparseStaticEmbedding), (
        "SparseStaticEmbedding component missing after loading"
    )
    assert torch.allclose(model[0].sub_modules.query[0].weight, loaded_model[0].sub_modules.query[0].weight), (
        "SparseStaticEmbedding weights changed after save and load"
    )


def test_sparse_static_embedding_from_json_partial_vocabulary(
    splade_bert_tiny_model: SparseEncoder, tmp_path: Path
) -> None:
    # An IDF file computed on a custom corpus only contains the tokens that occur in that corpus
    tokenizer = splade_bert_tiny_model.tokenizer
    idf_path = tmp_path / "idf.json"
    idf_path.write_text(json.dumps({"paris": 3.1, "capital": 2.4, "france": 2.9}), encoding="utf-8")

    query_module = SparseStaticEmbedding.from_json(str(idf_path), tokenizer=tokenizer, frozen=True)
    assert query_module.get_embedding_dimension() == len(tokenizer.get_vocab())

    model = SparseEncoder(
        modules=[
            Router.for_query_document(
                query_modules=[query_module],
                document_modules=[splade_bert_tiny_model[0], splade_bert_tiny_model[1]],
            )
        ]
    )
    query_embedding = model.encode_query("capital of france", convert_to_sparse_tensor=False)
    document_embedding = model.encode_document("Paris is the capital of France.", convert_to_sparse_tensor=False)
    assert query_embedding.shape == document_embedding.shape
    assert model.similarity(query_embedding, document_embedding).shape == (1, 1)
    assert query_embedding[tokenizer.convert_tokens_to_ids("capital")] == 2.4

    # Tokens that are not in the IDF file get a weight of 0
    unknown_embedding = model.encode_query("zebra", convert_to_sparse_tensor=False)
    assert unknown_embedding.abs().sum() == 0
