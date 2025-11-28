from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.models import Dense, StaticEmbedding


def test_dense_load_and_save_in_other_precisions(static_embedding_model: StaticEmbedding, tmp_path: Path) -> None:
    base_model = SentenceTransformer(modules=[static_embedding_model, Dense(768, 256, activation_function=nn.Tanh())])
    test_text = ["This is a test"]
    base_embedding = base_model.encode(test_text, convert_to_tensor=True)

    base_path = str(tmp_path / "model")
    base_model.save_pretrained(base_path)

    loaded_model = SentenceTransformer(base_path)
    assert loaded_model[1].linear.weight.dtype == torch.float32
    assert loaded_model[1].linear.weight.shape == (256, 768)
    loaded_embedding = loaded_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_embedding, atol=1e-6)

    fp64_model = deepcopy(base_model).to(torch.float64)
    fp64_path = str(tmp_path / "fp64")
    fp64_model.save_pretrained(fp64_path)
    loaded_fp64_model = SentenceTransformer(fp64_path)
    assert loaded_fp64_model[1].linear.weight.dtype == torch.float64
    assert loaded_fp64_model[1].linear.weight.shape == (256, 768)
    loaded_fp64_embedding = loaded_fp64_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_fp64_embedding.to(torch.float32), atol=1e-6)

    fp16_model = deepcopy(base_model).to(torch.float16)
    fp16_path = str(tmp_path / "fp16")
    fp16_model.save_pretrained(fp16_path)
    loaded_fp16_model = SentenceTransformer(fp16_path)
    assert loaded_fp16_model[1].linear.weight.dtype == torch.float16
    assert loaded_fp16_model[1].linear.weight.shape == (256, 768)
    loaded_fp16_embedding = loaded_fp16_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_fp16_embedding.to(torch.float32), atol=1e-3)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        bf16_model = deepcopy(base_model).to(torch.bfloat16)
        bf16_path = str(tmp_path / "bf16")
        bf16_model.save_pretrained(bf16_path)
        loaded_bf16_model = SentenceTransformer(bf16_path)
        assert loaded_bf16_model[1].linear.weight.dtype == torch.bfloat16
        assert loaded_bf16_model[1].linear.weight.shape == (256, 768)
        loaded_bf16_embedding = loaded_bf16_model.encode(test_text, convert_to_tensor=True)
        assert torch.allclose(base_embedding, loaded_bf16_embedding.to(torch.float32), atol=1e-2)


def test_dense_custom_features_key() -> None:
    """Test that Dense can operate on custom feature keys."""
    # Test with sentence_embedding (default)
    dense_default = Dense(768, 256, activation_function=nn.Tanh())
    features_default = {"sentence_embedding": torch.randn(2, 768)}
    output_default = dense_default.forward(features_default)
    assert "sentence_embedding" in output_default
    assert output_default["sentence_embedding"].shape == (2, 256)

    # Test with custom input key
    dense_custom = Dense(768, 256, activation_function=nn.Tanh(), module_input_name="token_embeddings")
    features_custom = {"token_embeddings": torch.randn(2, 10, 768)}
    output_custom = dense_custom.forward(features_custom)
    assert "token_embeddings" in output_custom
    assert output_custom["token_embeddings"].shape == (2, 10, 256)

    # Test with different input and output keys
    dense_different = Dense(
        768,
        256,
        activation_function=nn.Tanh(),
        module_input_name="input_embeddings",
        module_output_name="output_embeddings",
    )
    features_different = {"input_embeddings": torch.randn(2, 768)}
    output_different = dense_different.forward(features_different)
    assert "output_embeddings" in output_different
    assert output_different["output_embeddings"].shape == (2, 256)
    assert "input_embeddings" in output_different  # Original key should still be present


def test_dense_save_load_custom_keys(static_embedding_model: StaticEmbedding, tmp_path: Path) -> None:
    """Test that Dense with custom keys can be saved and loaded."""
    import os

    # Create a Dense layer with custom keys
    dense = Dense(
        768,
        256,
        activation_function=nn.Tanh(),
        module_input_name="custom_input",
        module_output_name="custom_output",
    )

    # Save config
    model_path = str(tmp_path / "dense_custom_keys")
    os.makedirs(model_path, exist_ok=True)
    dense.save(model_path)

    # Load and verify
    loaded_dense = Dense.load(model_path)
    assert loaded_dense.module_input_name == "custom_input"
    assert loaded_dense.module_output_name == "custom_output"
    assert loaded_dense.in_features == 768
    assert loaded_dense.out_features == 256

    # Verify forward pass works correctly
    features = {"custom_input": torch.randn(2, 768)}
    output = loaded_dense.forward(features)
    assert "custom_output" in output
    assert output["custom_output"].shape == (2, 256)
