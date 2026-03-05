from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from packaging.version import parse as parse_version
from transformers import __version__ as transformers_version

from sentence_transformers.modules import Transformer


class TestTransformerMaxSeqLength:
    """Test the max_seq_length property for both tokenizer-based and config-based models."""

    def test_max_seq_length_from_tokenizer(self):
        """max_seq_length should return the tokenizer's model_max_length when a tokenizer is available."""
        transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert transformer.max_seq_length is not None
        assert transformer.max_seq_length == transformer.tokenizer.model_max_length

    def test_max_seq_length_setter(self):
        """Setting max_seq_length should update the tokenizer's model_max_length."""
        transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        original = transformer.max_seq_length
        transformer.max_seq_length = 64
        assert transformer.max_seq_length == 64
        assert transformer.tokenizer.model_max_length == 64
        # Restore
        transformer.max_seq_length = original

    def test_max_seq_length_init_kwarg(self):
        """Passing max_seq_length to __init__ should set the tokenizer's model_max_length."""
        transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", max_seq_length=42)
        assert transformer.max_seq_length == 42

    def test_max_seq_length_capped_by_max_position_embeddings(self):
        """max_seq_length should not exceed the model's max_position_embeddings."""
        transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        if hasattr(transformer.config, "max_position_embeddings"):
            assert transformer.max_seq_length <= transformer.config.max_position_embeddings


class TestTransformerDeprecatedKwargs:
    """Test that old keyword argument names still work with deprecation warnings."""

    def test_model_args_deprecated(self, caplog):
        """Using model_args= should work but emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                model_args={},
            )
        assert any("model_args" in record.message and "deprecated" in record.message for record in caplog.records)
        assert transformer is not None

    def test_tokenizer_args_deprecated(self, caplog):
        """Using tokenizer_args= should work but emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                tokenizer_args={},
            )
        assert any("tokenizer_args" in record.message and "deprecated" in record.message for record in caplog.records)
        assert transformer is not None

    def test_config_args_deprecated(self, caplog):
        """Using config_args= should work but emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                config_args={},
            )
        assert any("config_args" in record.message and "deprecated" in record.message for record in caplog.records)
        assert transformer is not None

    def test_new_kwargs_no_warning(self, caplog):
        """Using new keyword names should not emit deprecation warnings."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                model_kwargs={},
                processor_kwargs={},
                config_kwargs={},
            )
        assert not any("deprecated" in record.message for record in caplog.records)
        assert transformer is not None


class TestTransformerModalityConfigValidation:
    """Test modality_config validation in __init__."""

    def test_valid_modality_config(self):
        """A valid modality_config should be accepted."""
        transformer = Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
            module_output_name="token_embeddings",
        )
        assert transformer.modality_config == {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"}
        }

    def test_invalid_modality_config_missing_method(self):
        """A modality_config entry missing 'method' should raise ValueError."""
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                modality_config={"text": {"method_output_name": "last_hidden_state"}},
                module_output_name="token_embeddings",
            )

    def test_invalid_modality_config_missing_output_name(self):
        """A modality_config entry missing 'method_output_name' should raise ValueError."""
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                modality_config={"text": {"method": "forward"}},
                module_output_name="token_embeddings",
            )

    def test_modality_config_requires_module_output_name(self):
        """Providing modality_config without module_output_name should raise ValueError."""
        with pytest.raises(ValueError, match="module_output_name"):
            Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, expected_class_name",
    [
        (
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            "BertModel",
        ),
        ("hf-internal-testing/tiny-random-t5", "T5EncoderModel"),
        ("hf-internal-testing/tiny-random-mt5", "MT5EncoderModel"),
        ("google/t5gemma-s-s-prefixlm", "T5GemmaEncoderModel"),
        ("google/t5gemma-2-270m-270m", "T5Gemma2Encoder"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available() and torch.cuda.device_count() == 0, reason="Requires torch backend")
def test_transformer_load_save_roundtrip(tmp_path: Path, model_name: str, expected_class_name: str):
    if parse_version(transformers_version) < parse_version("5.0.0") and expected_class_name == "T5Gemma2Encoder":
        pytest.skip("T5Gemma2Encoder requires transformers>=5.0.0")
    if parse_version(transformers_version) < parse_version("4.54.1") and expected_class_name == "T5GemmaEncoderModel":
        pytest.skip("T5GemmaEncoderModel requires transformers>=4.54.1")

    # Load module via SentenceTransformer's Transformer building block
    transformer = Transformer(model_name_or_path=model_name)

    # Check that underlying model class matches expectation
    actual_class_name = type(transformer.auto_model).__name__
    assert actual_class_name == expected_class_name

    # Hack to mirror the fix from https://github.com/huggingface/transformers/pull/43633, required for T5Gemma2Encoder inference
    if expected_class_name == "T5Gemma2Encoder":
        transformer.auto_model.config._attn_implementation = "eager"

    # Prepare a tiny batch
    texts = ["hello world", "goodbye world"]
    features = transformer.tokenize(texts)

    with torch.no_grad():
        out1 = transformer(features)

    # Save and reload
    save_dir = tmp_path / "model"
    transformer.save(str(save_dir))

    reloaded = Transformer.load(str(save_dir))

    # Check that underlying model class matches expectation
    actual_class_name = type(reloaded.auto_model).__name__
    assert actual_class_name == expected_class_name

    # Hack to mirror the fix from https://github.com/huggingface/transformers/pull/43633, required for T5Gemma2Encoder inference
    if expected_class_name == "T5Gemma2Encoder":
        reloaded.auto_model.config._attn_implementation = "eager"

    # Retokenize just in case
    features = reloaded.tokenize(texts)

    with torch.no_grad():
        out2 = reloaded(features)

    for key in out1.keys():
        value1 = out1[key]
        value2 = out2[key]
        if isinstance(value1, torch.Tensor):
            assert torch.allclose(value1, value2), f"Outputs for key {key} differ after save/load"
        else:
            assert value1 == value2, f"Outputs for key {key} differ after save/load"
