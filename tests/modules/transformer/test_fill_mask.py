"""
Test suite for Transformer module with fill-mask task.
Tests various tiny model architectures that support masked language modeling,
which is used by SparseEncoder (SPLADE-style models).
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as transformers_version
from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES

from sentence_transformers.modules import Transformer
from sentence_transformers.util.tensor import batch_to_device

from .conftest import (
    EXPECT_FORWARD_FAIL,
    FAULTY_CHECKPOINTS,
    REQUIRES_CUDA,
    TINY_MODEL_MAPPING,
    TRANSFORMERS_V4_XFAIL_ARCHITECTURES,
    XFAIL_ARCHITECTURES,
    create_modality_samples,
    load_transformer,
)

# Architectures that fail specifically for fill-mask
XFAIL_FILL_MASK = []


def _get_fill_mask_archs() -> list[str]:
    """Get architectures that support fill-mask (masked LM)."""
    return [
        key
        for key, value in TINY_MODEL_MAPPING.items()
        if value is not None
        and key not in XFAIL_ARCHITECTURES
        and key not in XFAIL_FILL_MASK
        and (key not in TRANSFORMERS_V4_XFAIL_ARCHITECTURES or Version(transformers_version) >= Version("5.0.0"))
        and key not in FAULTY_CHECKPOINTS
        and key in MODEL_FOR_MASKED_LM_MAPPING_NAMES
        and key in MODEL_MAPPING_NAMES
    ]


@pytest.fixture(params=_get_fill_mask_archs(), scope="class")
def arch(request):
    """Get the model architecture name for fill-mask task."""
    return request.param


@pytest.fixture(scope="class")
def arch_model(arch):
    model = load_transformer(arch, transformer_task="fill-mask")
    return arch, model


@pytest.fixture
def arch_model_modalities(arch_model):
    """Create a Transformer instance and return it with its supported modalities."""
    try:
        arch, model = arch_model
        modalities = model.modalities
        return arch, model, modalities
    except Exception as e:
        pytest.fail(f"Failed to get modalities: {e}")


class TestFillMaskArchitectures:
    """Test suite for Transformer module with fill-mask task."""

    def test_get_embedding_dimension(self, arch_model):
        """Test that embedding dimension can be retrieved."""
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

    def test_module_output_name(self, arch_model):
        """Test that the module output name is 'token_embeddings' for fill-mask."""
        arch, model = arch_model
        assert model.module_output_name == "token_embeddings"

    def test_inference_with_supported_modalities(self, arch_model_modalities, subtests):
        """Test inference with each supported modality."""
        arch, model, modalities = arch_model_modalities
        if arch in REQUIRES_CUDA:
            if not torch.cuda.is_available():
                pytest.skip(f"{arch} requires CUDA for inference, but CUDA is not available.")
            else:
                model = model.to("cuda")

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

                with context:
                    features = model.preprocess(inputs)
                    if arch in REQUIRES_CUDA:
                        features = batch_to_device(features, torch.device("cuda"))
                    with torch.no_grad():
                        output = model.forward(features)
                    assert model.module_output_name in output, (
                        f"Expected '{model.module_output_name}' in output for {modality_desc}"
                    )

                    output_tensor = output[model.module_output_name]
                    assert output_tensor.shape[0] == len(inputs), f"Batch size mismatch for {modality_desc}"
