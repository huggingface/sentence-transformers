from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding
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
    tokenizer = splade_bert_tiny_model.tokenizer
    idf_path = tmp_path / "idf.json"
    idf_path.write_text(json.dumps({"paris": 3.1, "capital": 2.4, "france": 2.9}), encoding="utf-8")

    module = SparseStaticEmbedding.from_json(str(idf_path), tokenizer=tokenizer, frozen=True)
    assert module.get_embedding_dimension() == len(tokenizer.get_vocab())

    embeddings = module(module.preprocess(["capital", "zebra"]))["sentence_embedding"]
    assert embeddings[0, tokenizer.convert_tokens_to_ids("capital")] == 2.4
    assert embeddings[1].count_nonzero() == 0


@pytest.mark.parametrize("explicit_cache_folder", [False, True])
def test_sparse_static_embedding_loads_json_checkpoint(tmp_path: Path, explicit_cache_folder: bool) -> None:
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    )
    module_path = tmp_path / "query"
    module_path.mkdir()
    tokenizer.save_pretrained(module_path)
    idf_path = module_path / "idf.json"
    idf_path.write_text(json.dumps({"alpha": 2.5, "beta": 1.0}), encoding="utf-8")
    (module_path / "config.json").write_text(json.dumps({"path": str(idf_path), "frozen": True}), encoding="utf-8")

    loaded = SparseStaticEmbedding.load(
        str(tmp_path),
        subfolder="query",
        cache_folder=str(tmp_path / "cache") if explicit_cache_folder else None,
        local_files_only=True,
    )

    assert not loaded.weight.requires_grad
    model = SparseEncoder(modules=[loaded], device="cpu")
    expected = torch.tensor([[0.0, 0.0, 2.5, 1.0], [0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(model.encode_query(["alpha beta", "beta"], convert_to_sparse_tensor=False), expected)
