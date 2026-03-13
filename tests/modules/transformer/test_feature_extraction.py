"""
Test suite for Transformer module with feature-extraction task (default).
Tests various tiny model architectures with pre-built tiny models from
hf-internal-testing and tiny-random organizations on the Hugging Face Hub,
and tests various input modalities (text, image, audio, video, message).
"""

from __future__ import annotations

from contextlib import nullcontext
from itertools import combinations
from typing import get_args

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as transformers_version
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from sentence_transformers.base.modality_types import Modality
from sentence_transformers.base.model import BaseModel
from sentence_transformers.modules import Transformer
from sentence_transformers.util.tensor import batch_to_device

from .conftest import (
    EXPECT_FORWARD_FAIL,
    EXPECT_IMAGE_ONLY_FAILURE,
    EXPECT_IMAGE_VIDEO_FAILURE,
    EXPECT_MULTIMODAL_FAILURE,
    EXPECT_MULTIMODAL_SUCCESS,
    FAULTY_CHECKPOINTS,
    REQUIRES_CUDA,
    TINY_MODEL_MAPPING,
    TRANSFORMERS_V4_XFAIL_ARCHITECTURES,
    XFAIL_ARCHITECTURES,
    create_modality_samples,
    load_transformer,
)

EXPECT_FORWARD_FAIL = EXPECT_FORWARD_FAIL.copy() | {
    "qwen2_5_vl": (  # Doesn't nicely work with image+video, gives StopIteration on processing
        "image+video (url, url)",
        "text+image+video (text, url, url)",
        "image+video as message (structured, url, url)",
    )
}

XFAIL_FEATURE_EXTRACTION = [
    "reformer",  # Outputs last_hidden_state with 2 * hidden_size dimensionality as it concats the same tensor for reversible ResNet, low prio
]


@pytest.fixture(
    params=[
        key
        for key, value in TINY_MODEL_MAPPING.items()
        if value is not None
        and key not in XFAIL_ARCHITECTURES
        and key not in XFAIL_FEATURE_EXTRACTION
        and (key not in TRANSFORMERS_V4_XFAIL_ARCHITECTURES or Version(transformers_version) >= Version("5.0.0"))
        and key not in FAULTY_CHECKPOINTS
        and key in MODEL_MAPPING_NAMES
    ],
    scope="class",
)
def arch(request):
    """Get the model architecture name."""
    return request.param


@pytest.fixture(scope="class")
def arch_model(arch):
    model = load_transformer(arch, transformer_task="feature-extraction")
    return arch, model


@pytest.fixture
def arch_model_modalities(arch_model):
    """Create a Transformer instance and return it with its supported modalities.

    Returns:
        tuple: (arch, model, supported_modalities_list)
    """
    try:
        arch, model = arch_model
        modalities = model.modalities
        return arch, model, modalities
    except Exception as e:
        pytest.fail(f"Failed to get modalities: {e}")


class TestTransformerArchitectures:
    """Test suite for Transformer module with various tiny model architectures (feature-extraction task)."""

    def test_get_embedding_dimension(self, arch_model):
        """Test that word embedding dimension can be retrieved."""
        arch, model = arch_model
        dim = model.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_save_load(self, arch_model, tmp_path):
        """Test saving and loading the model."""
        arch, model = arch_model

        save_path = tmp_path / "model"
        save_path.mkdir(exist_ok=True)

        model.save(str(save_path))
        loaded_model = Transformer(str(save_path))

        assert loaded_model.get_embedding_dimension() == model.get_embedding_dimension()

    def test_modalities_property(self, arch_model_modalities):
        """Test that the modalities property returns a list."""
        arch, model, modalities = arch_model_modalities
        assert isinstance(modalities, list)
        assert len(modalities) > 0

    def test_supports_consistency(self, arch_model_modalities, subtests):
        """Test that supports() is consistent with the modalities property.

        For each architecture, verifies that:
        1. Each listed modality is supported.
        2. Unlisted single modalities are not supported.
        3. Tuple modalities composed of listed parts are supported iff "message" is also listed.
        4. Tuple modalities with unlisted parts are never supported.
        """
        arch, model, modalities = arch_model_modalities
        has_message = "message" in modalities
        non_message = [m for m in modalities if isinstance(m, str) and m != "message"]
        all_single: list[Modality] = list(get_args(get_args(Modality)[0]))

        for modality in modalities:
            with subtests.test(msg=f"{modality} listed => supported"):
                assert BaseModel.supports(model, modality)

        for modality in all_single:
            if modality not in modalities:
                with subtests.test(msg=f"{modality} unlisted => not supported"):
                    assert not BaseModel.supports(model, modality)

        # Test tuple modalities composed of listed non-message parts
        if len(non_message) >= 2:
            for combo in combinations(non_message, 2):
                with subtests.test(msg=f"{combo} tuple => {'supported' if has_message else 'not supported'}"):
                    if has_message:
                        assert BaseModel.supports(model, combo)
                    else:
                        # Without message, tuple is only supported if explicitly listed
                        assert BaseModel.supports(model, combo) == (combo in modalities)

        # Test tuple with a truly unlisted part is never supported.
        # A modality is "truly unlisted" if it doesn't appear as a single modality,
        # nor as a part of any explicit tuple modality in the modalities list.
        all_parts = set()
        for m in modalities:
            if isinstance(m, tuple):
                all_parts.update(m)
            elif m != "message":
                all_parts.add(m)
        truly_unlisted = [m for m in all_single if m not in all_parts]
        if non_message and truly_unlisted:
            bad_tuple = (non_message[0], truly_unlisted[0])
            with subtests.test(msg=f"{bad_tuple} with unlisted part => not supported"):
                assert not BaseModel.supports(model, bad_tuple)

    def test_inference_with_supported_modalities(self, arch_model_modalities, subtests):
        """Test inference with each supported modality (single and multi-modal)."""
        arch, model, modalities = arch_model_modalities
        if arch in REQUIRES_CUDA:
            if not torch.cuda.is_available():
                pytest.skip(f"{arch} requires CUDA for inference, but CUDA is not available.")
            else:
                model = model.to("cuda")

        # Create all valid test samples for the model's supported modalities
        test_samples = create_modality_samples(
            model, modalities, n=2, message_format=model.input_formatter.message_format
        )

        for modality_desc, inputs in test_samples.items():
            with subtests.test(msg=f"Testing {modality_desc}"):
                context = nullcontext()
                expected_fail = EXPECT_FORWARD_FAIL.get(arch)
                if expected_fail is None and arch in EXPECT_FORWARD_FAIL:
                    context = pytest.raises(Exception)
                elif expected_fail is not None and modality_desc in expected_fail:
                    context = pytest.raises(Exception)
                elif arch in EXPECT_IMAGE_VIDEO_FAILURE and "image" in modality_desc and "video" in modality_desc:
                    context = pytest.raises((ValueError, TypeError, AttributeError))
                elif arch in EXPECT_IMAGE_ONLY_FAILURE and "image" in modality_desc and "+" not in modality_desc:
                    context = pytest.raises((ValueError, TypeError, AttributeError))
                elif (
                    model.module_output_name == "sentence_embedding"
                    and "+" in modality_desc
                    and arch not in EXPECT_MULTIMODAL_SUCCESS
                ) or ("+" in modality_desc and arch in EXPECT_MULTIMODAL_FAILURE):
                    # If a model outputs sentence embeddings directly, that's likely via the get_..._features methods,
                    # which typically don't support multimodal inputs (except blip), so we can expect a failure in that case.
                    # BaseModel.preprocess already raises a clear ValueError via `self.supports(modality)` before
                    # the Transformer module is invoked, so end users get a meaningful error message.
                    context = pytest.raises(ValueError)

                with context:
                    # Preprocess the data & forward
                    try:
                        features = model.preprocess(inputs)
                    except ValueError as exc:
                        if (
                            "Could not make a flat list of images from" in str(exc)
                            and "image" in modality_desc
                            and ("url" in modality_desc or "path" in modality_desc)
                        ):
                            pytest.skip(
                                f"The {arch!r} architecture with an older transformers version doesn't support image URLs, skipping this modality format."
                            )
                        raise
                    except KeyError as exc:
                        if "'height'" in str(exc) and "image" in modality_desc:
                            pytest.skip(
                                f"The {arch!r} architecture with an older transformers version doesn't yet nicely extract the size of image inputs, skipping this modality format."
                            )
                        raise

                    if arch in REQUIRES_CUDA:
                        features = batch_to_device(features, torch.device("cuda"))
                    with torch.no_grad():
                        output = model.forward(features)

                    # Verify output has expected keys
                    assert model.module_output_name in output, (
                        f"Expected '{model.module_output_name}' in output for {modality_desc}"
                    )

                    # Verify batch size is correct
                    output_tensor = output[model.module_output_name]
                    assert output_tensor.shape[0] == len(inputs), f"Batch size mismatch for {modality_desc}"
                    assert output_tensor.shape[-1] == model.get_embedding_dimension(), (
                        f"Embedding dimension mismatch for {modality_desc}"
                    )

                    # token_embeddings should be 3D (batch, seq_len, hidden_dim)
                    # sentence_embedding should be 2D (batch, hidden_dim)
                    if model.module_output_name == "token_embeddings":
                        assert output_tensor.ndim == 3, (
                            f"Expected 3D tensor for token_embeddings, got {output_tensor.ndim}D for {modality_desc}"
                        )
                    elif model.module_output_name == "sentence_embedding":
                        assert output_tensor.ndim == 2, (
                            f"Expected 2D tensor for sentence_embedding, got {output_tensor.ndim}D for {modality_desc}"
                        )
