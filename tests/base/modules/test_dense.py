from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def test_dense_load_and_save_in_other_precisions(static_embedding: StaticEmbedding, tmp_path: Path) -> None:
    base_model = SentenceTransformer(modules=[static_embedding, Dense(768, 256, activation_function=nn.Tanh())])
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


def test_dense_save_load_custom_keys(static_embedding: StaticEmbedding, tmp_path: Path) -> None:
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


def test_dense_load_ignores_unknown_config_keys(tmp_path: Path, caplog) -> None:
    """A newer or foreign save (e.g. a future PyLate ``Dense``) may record a config key this class's
    constructor does not accept. ``Dense.load`` should drop such keys with a warning rather than crash in
    ``cls(**config)``."""
    import json
    import logging

    dense = Dense(768, 256, bias=False, activation_function=nn.Identity())
    model_dir = tmp_path / "dense_extra_key"
    model_dir.mkdir()
    dense.save(str(model_dir))

    # Inject a config key that ``Dense.__init__`` does not accept.
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["future_pylate_param"] = 42
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="sentence_transformers.base.modules.dense"):
        loaded = Dense.load(str(model_dir))

    # Loaded successfully; the unknown key was dropped and the module is otherwise intact.
    assert isinstance(loaded, Dense)
    assert not hasattr(loaded, "future_pylate_param")
    assert loaded.in_features == 768
    assert loaded.out_features == 256
    # A warning names the dropped key.
    assert any("future_pylate_param" in record.message for record in caplog.records)

    # The reloaded module still runs a forward pass.
    output = loaded.forward({"sentence_embedding": torch.randn(2, 768)})
    assert output["sentence_embedding"].shape == (2, 256)


def test_dense_config_omits_default_use_residual(tmp_path: Path) -> None:
    """``use_residual`` at its default (False) must not be written to config.json: released
    5.4-5.6 ``Dense.load`` does ``cls(**config)`` without unknown-key dropping, so any new key at
    default would make fresh saves unloadable there. A non-default value must still round-trip."""
    import json

    default_dir = tmp_path / "dense_default"
    default_dir.mkdir()
    Dense(16, 8, bias=False, activation_function=nn.Identity()).save(str(default_dir))
    config = json.loads((default_dir / "config.json").read_text(encoding="utf-8"))
    assert "use_residual" not in config

    residual_dir = tmp_path / "dense_residual"
    residual_dir.mkdir()
    Dense(16, 8, bias=False, activation_function=nn.Identity(), use_residual=True).save(str(residual_dir))
    config = json.loads((residual_dir / "config.json").read_text(encoding="utf-8"))
    assert config["use_residual"] is True
    loaded = Dense.load(str(residual_dir))
    assert loaded.use_residual is True
