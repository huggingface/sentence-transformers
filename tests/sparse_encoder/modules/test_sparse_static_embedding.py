from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding
from tests.sparse_encoder.utils import sparse_allclose


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("pad_token", ["[UNK]", "[PAD]"])
def test_sparse_static_embedding_padding_preserves_valid_token(padding_side: str, pad_token: str) -> None:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "known": 2, "longer": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token=pad_token,
        padding_side=padding_side,
        model_max_length=32,
    )
    module = SparseStaticEmbedding(tokenizer=tokenizer, weight=torch.tensor([2.0, 5.0, 3.0, 4.0]))
    model = SparseEncoder(modules=[module], device="cpu")
    texts = ["unseen", "known longer"]

    single_embeddings = model.encode(texts, batch_size=1, convert_to_sparse_tensor=False)
    batch_embeddings = model.encode(texts, batch_size=2, convert_to_sparse_tensor=False)

    torch.testing.assert_close(batch_embeddings, single_embeddings)
    torch.testing.assert_close(batch_embeddings[0], torch.tensor([2.0, 0.0, 0.0, 0.0]))


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
    tokenizer = splade_bert_tiny_model.tokenizer
    idf_path = tmp_path / "idf.json"
    idf_path.write_text(json.dumps({"paris": 3.1, "capital": 2.4, "france": 2.9}), encoding="utf-8")

    module = SparseStaticEmbedding.from_json(str(idf_path), tokenizer=tokenizer, frozen=True)
    assert module.get_embedding_dimension() == len(tokenizer.get_vocab())

    embeddings = module(module.preprocess(["capital", "zebra"]))["sentence_embedding"]
    assert embeddings[0, tokenizer.convert_tokens_to_ids("capital")] == 2.4
    assert embeddings[1].count_nonzero() == 0
