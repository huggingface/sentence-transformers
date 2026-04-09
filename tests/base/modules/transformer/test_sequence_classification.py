"""
Test suite for Transformer module with sequence-classification task.
Tests various tiny model architectures that support sequence classification,
which is the default task used by CrossEncoder.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as transformers_version
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from sentence_transformers.base.modules import Transformer
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
    create_modality_pair_samples,
    create_modality_samples,
    load_transformer,
    modify_processor_for_pairs,
)

# Architectures that fail specifically for sequence-classification
# (beyond the general XFAIL_ARCHITECTURES)
XFAIL_SEQUENCE_CLASSIFICATION = [
    "luke",  # The LUKE tokenize doesn't work conveniently with text pairs
]


def _get_seq_cls_archs() -> list[str]:
    """Get architectures that support sequence-classification."""
    return [
        key
        for key, value in TINY_MODEL_MAPPING.items()
        if value is not None
        and key not in XFAIL_ARCHITECTURES
        and key not in XFAIL_SEQUENCE_CLASSIFICATION
        and (key not in TRANSFORMERS_V4_XFAIL_ARCHITECTURES or Version(transformers_version) >= Version("5.0.0"))
        and key not in FAULTY_CHECKPOINTS
        and key in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        and key in MODEL_MAPPING_NAMES  # Ensure we also have a base model
    ]


@pytest.fixture(params=_get_seq_cls_archs(), scope="class")
def arch(request):
    """Get the model architecture name for sequence-classification task."""
    return request.param


@pytest.fixture(scope="class")
def arch_model(arch):
    model = load_transformer(arch, transformer_task="sequence-classification")
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


class TestSequenceClassificationArchitectures:
    """Test suite for Transformer module with sequence-classification task."""

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
        """Test that the module output name is 'scores' for sequence-classification."""
        arch, model = arch_model
        assert model.module_output_name == "scores"

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
                    context = pytest.raises(ValueError)

                with context:
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

                    assert model.module_output_name in output, (
                        f"Expected '{model.module_output_name}' in output for {modality_desc}"
                    )

                    output_tensor = output[model.module_output_name]
                    assert output_tensor.shape[0] == len(inputs), f"Batch size mismatch for {modality_desc}"
                    assert output_tensor.shape[-1] == 1, f"Output dimension mismatch for {modality_desc}"

    def test_inference_with_supported_modality_pairs(self, arch_model_modalities, subtests):
        """Test inference with pair inputs for each supported modality combination."""
        arch, model, modalities = arch_model_modalities
        if arch in REQUIRES_CUDA:
            if not torch.cuda.is_available():
                pytest.skip(f"{arch} requires CUDA for inference, but CUDA is not available.")
            else:
                model = model.to("cuda")

        if "message" in modalities:
            modify_processor_for_pairs(model)

        test_pairs = create_modality_pair_samples(model, modalities, n=2)

        for pair_desc, pairs in test_pairs.items():
            with subtests.test(msg=f"Testing {pair_desc}"):
                context = nullcontext()
                if arch in EXPECT_FORWARD_FAIL and EXPECT_FORWARD_FAIL[arch] is None:
                    context = pytest.raises(Exception)

                with context:
                    features = model.preprocess(pairs)
                    if arch in REQUIRES_CUDA:
                        features = batch_to_device(features, torch.device("cuda"))
                    with torch.no_grad():
                        output = model.forward(features)

                    assert model.module_output_name in output, (
                        f"Expected '{model.module_output_name}' in output for {pair_desc}"
                    )
                    output_tensor = output[model.module_output_name]
                    assert output_tensor.shape[0] == len(pairs), f"Batch size mismatch for {pair_desc}"
                    assert output_tensor.shape[-1] == 1, f"Output dimension mismatch for {pair_desc}"
