"""
Test different backends (PyTorch, ONNX, OpenVINO) for MultiVectorEncoder.

This module tests exporting, loading, and using models with different inference backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils import is_ci

try:
    from optimum.intel import OVModelForFeatureExtraction
    from optimum.onnxruntime import ORTModelForFeatureExtraction
except ImportError:
    pytest.skip("OpenVINO and ONNX backends are not available", allow_module_level=True)

from sentence_transformers import (
    MultiVectorEncoder,
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
)

if is_ci():
    pytest.skip("Skip test in CI to try and avoid 429 Client Error", allow_module_level=True)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    # The tiny checkpoint gets a randomly initialised projection on conversion, so save it once
    # and reload from disk per backend. Otherwise every load would produce different embeddings.
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model[0].query_expansion = {"strategy": "pad_skip", "length": 16}
    save_dir = str(tmp_path_factory.mktemp("mve_backends") / "mve-tiny")
    model.save_pretrained(save_dir)
    return save_dir


@pytest.fixture(scope="module")
def torch_embeddings(model_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model = MultiVectorEncoder(model_dir, backend="torch")
    queries = model.encode_query(["What is the capital of France?"])
    documents = model.encode_document(["Paris is the capital of France.", "Berlin is the capital of Germany."])
    return queries, documents


def encode_and_compare(model: MultiVectorEncoder, torch_embeddings, atol: float = 1e-5) -> None:
    reference_queries, reference_documents = torch_embeddings
    queries = model.encode_query(["What is the capital of France?"])
    documents = model.encode_document(["Paris is the capital of France.", "Berlin is the capital of Germany."])
    assert queries[0].shape == (16, model.get_embedding_dimension())
    for reference, embedding in zip(reference_queries + reference_documents, queries + documents):
        assert reference.shape == embedding.shape
        assert np.allclose(reference, embedding, atol=atol)


@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForFeatureExtraction),
        ("openvino", OVModelForFeatureExtraction),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs, model_dir, torch_embeddings) -> None:
    model = MultiVectorEncoder(model_dir, backend=backend, model_kwargs=model_kwargs)
    assert model.get_backend() == backend
    assert isinstance(model[0].auto_model, expected_auto_model_class)
    assert isinstance(model.transformers_model, expected_auto_model_class)
    encode_and_compare(model, torch_embeddings)


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_backend_save_and_load(backend, model_dir, torch_embeddings, tmp_path) -> None:
    export_dir = str(tmp_path / "exported")
    model = MultiVectorEncoder(model_dir, backend=backend)
    model.save_pretrained(export_dir)

    exported_model = MultiVectorEncoder(export_dir, backend=backend)
    encode_and_compare(exported_model, torch_embeddings)


def test_onnx_optimize_and_quantize(model_dir, torch_embeddings, tmp_path) -> None:
    export_dir = str(tmp_path / "exported")
    model = MultiVectorEncoder(model_dir, backend="onnx")
    model.save_pretrained(export_dir)

    export_optimized_onnx_model(model, "O1", export_dir)
    optimized_model = MultiVectorEncoder(export_dir, backend="onnx", model_kwargs={"file_name": "onnx/model_O1.onnx"})
    encode_and_compare(optimized_model, torch_embeddings, atol=1e-3)

    export_dynamic_quantized_onnx_model(model, "avx512_vnni", export_dir)
    quantized_model = MultiVectorEncoder(
        export_dir, backend="onnx", model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"}
    )
    reference_queries, reference_documents = torch_embeddings
    queries = quantized_model.encode_query(["What is the capital of France?"])
    documents = quantized_model.encode_document(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    )
    # Quantization shifts values, so only the shapes are compared.
    for reference, embedding in zip(reference_queries + reference_documents, queries + documents):
        assert reference.shape == embedding.shape


def test_incorrect_backend(model_dir) -> None:
    with pytest.raises(ValueError):
        MultiVectorEncoder(model_dir, backend="incorrect_backend")
