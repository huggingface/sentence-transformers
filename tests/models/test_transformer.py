from __future__ import annotations

from pathlib import Path

import pytest
import torch
from packaging.version import parse as parse_version
from transformers import __version__ as transformers_version

from sentence_transformers.models import Transformer


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
        assert torch.allclose(out1[key], out2[key]), f"Outputs for key {key} differ after save/load"
