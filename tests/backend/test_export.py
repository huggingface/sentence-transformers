from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.backend.export import (
    MIN_TRANSFORMERS_VERSION,
    has_export_config,
    is_transformers_export_available,
)

SENTENCES = ["Hello, World!", "A much longer sentence about the weather today."]
PAIRS = [("Which planet is red?", "Mars is the red planet."), ("Which planet is red?", "Not Venus.")]

MODELS = {
    "sentence_transformer": (SentenceTransformer, "sentence-transformers-testing/stsb-bert-tiny-safetensors"),
    "cross_encoder": (CrossEncoder, "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"),
    "sparse_encoder": (SparseEncoder, "sparse-encoder-testing/splade-bert-tiny-nq"),
    "multi_vector_encoder": (MultiVectorEncoder, "multi-vector-encoder-testing/bert-tiny-multi-vector"),
}
EXPORT_FORMATS = ["onnx", "openvino"]

# Exported by a previous version of sentence-transformers: only its Transformer module is exported
LEGACY_MODEL = "sentence-transformers-testing/stsb-bert-tiny-openvino"


def has_quantizer(name: str, packages: list[str]) -> bool:
    if not is_transformers_export_available() or any(importlib.util.find_spec(p) is None for p in packages):
        return False
    import transformers.exporters

    return hasattr(transformers.exporters, name)


requires_exporters = pytest.mark.skipif(
    not is_transformers_export_available() or any(importlib.util.find_spec(p) is None for p in EXPORT_FORMATS),
    reason=f"exporting a model requires transformers>={MIN_TRANSFORMERS_VERSION} and {EXPORT_FORMATS}",
)

requires_onnxruntime_quantizer = pytest.mark.skipif(
    not has_quantizer("OnnxRuntimeQuantizer", [*EXPORT_FORMATS, "onnxruntime"]),
    reason="quantizing with ONNX Runtime requires a transformers version with OnnxRuntimeQuantizer, and onnxruntime",
)
requires_nncf_onnx_quantizer = pytest.mark.skipif(
    not has_quantizer("NNCFOnnxQuantizer", [*EXPORT_FORMATS, "onnxruntime", "nncf"]),
    reason="quantizing with NNCF requires a transformers version with NNCFOnnxQuantizer, onnxruntime and nncf",
)
requires_nncf_openvino_quantizer = pytest.mark.skipif(
    not has_quantizer("NNCFOpenVINOQuantizer", [*EXPORT_FORMATS, "nncf"]),
    reason="quantizing with NNCF requires a transformers version with NNCFOpenVINOQuantizer and nncf",
)

QUANTIZED_ONNX_OPS = {"QuantizeLinear", "DynamicQuantizeLinear", "MatMulInteger", "QLinearMatMul", "MatMulNBits"}


def compute(model) -> list[np.ndarray]:
    if isinstance(model, CrossEncoder):
        return [np.asarray(model.predict(PAIRS))]
    if isinstance(model, SparseEncoder):
        return [model.encode(SENTENCES).to_dense().numpy()]
    if isinstance(model, MultiVectorEncoder):
        return [np.asarray(embedding) for embedding in model.encode(SENTENCES)]
    return [model.encode(SENTENCES)]


def export_model(architecture, tmp_path, export_format="torch", export_config=None):
    """Export a model, with an export format or an export config (e.g. one with a quantizer), and save it. Returns the
    model and the directory it is saved to"""
    model_class, model_id = MODELS[architecture]
    model_kwargs = {"export_config": export_config} if export_config is not None else {}
    model = model_class(model_id, backend=export_format, model_kwargs=model_kwargs)
    model.save(str(tmp_path / "export"))
    return model, tmp_path / "export"


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
@pytest.mark.parametrize("architecture", MODELS)
def test_exported_model_matches(architecture, export_format, tmp_path):
    model_class, model_id = MODELS[architecture]
    expected = compute(model_class(model_id))
    model, _ = export_model(architecture, tmp_path, export_format=export_format)
    assert model.get_backend() == export_format
    for expected_output, output in zip(expected, compute(model)):
        assert np.allclose(expected_output, output, atol=1e-3)


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_saved_export_is_loaded_not_exported_again(export_format, tmp_path, monkeypatch):
    expected = compute(SentenceTransformer(MODELS["sentence_transformer"][1]))
    _, directory = export_model("sentence_transformer", tmp_path, export_format=export_format)
    assert has_export_config(directory, export_format)

    reloaded = SentenceTransformer(str(directory), backend=export_format)
    for expected_output, output in zip(expected, compute(reloaded)):
        assert np.allclose(expected_output, output, atol=1e-3)


@requires_exporters
def test_export_without_pytorch_weights_is_loaded():

    model = SentenceTransformer("optimum-intel-internal-testing/stsb-bert-tiny-openvino", backend="openvino")
    assert model._exported_model is not None
    assert model[0].model is None
    assert model.get_backend() == "openvino"
    for expected_output, output in zip(
        compute(SentenceTransformer(MODELS["sentence_transformer"][1])), compute(model)
    ):
        assert np.allclose(expected_output, output, atol=5e-2)


def test_legacy_export_is_loaded():
    pytest.importorskip("optimum.intel")
    model = SentenceTransformer(LEGACY_MODEL, backend="openvino")
    assert model._exported_model is None
    assert type(model[0].model).__name__.startswith("OV")
    for expected_output, output in zip(compute(SentenceTransformer(LEGACY_MODEL)), compute(model)):
        assert np.allclose(expected_output, output, atol=5e-2)


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_legacy_export_is_exported_again(export_format, tmp_path):
    expected = compute(SentenceTransformer(LEGACY_MODEL))
    model = SentenceTransformer(LEGACY_MODEL, backend=export_format, model_kwargs={"export": True})
    assert model._exported_model is not None
    for expected_output, output in zip(expected, compute(model)):
        assert np.allclose(expected_output, output, atol=5e-2)

    model.save(str(tmp_path))
    assert has_export_config(tmp_path, export_format)


def onnx_op_types(directory) -> set[str]:
    import onnx

    (path,) = directory.glob("*.onnx")
    return {node.op_type for node in onnx.load(path, load_external_data=False).graph.node}


def openvino_op_types(directory) -> set[str]:
    import openvino

    (path,) = directory.glob("*.xml")
    operations = openvino.Core().read_model(path).get_ops()
    types = {operation.get_type_name() for operation in operations}
    for operation in operations:
        if operation.get_type_name() == "Constant":
            types.add(operation.get_output_element_type(0).get_type_name())
    return types


@requires_onnxruntime_quantizer
@pytest.mark.parametrize("architecture", MODELS)
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_onnxruntime_quantized_export(architecture, mode, tmp_path):
    from transformers.exporters import (
        OnnxConfig,
        OnnxRuntimeDynamicConfig,
        OnnxRuntimeQuantizer,
        OnnxRuntimeStaticConfig,
    )

    config = OnnxRuntimeDynamicConfig() if mode == "dynamic" else OnnxRuntimeStaticConfig()
    export_config = OnnxConfig(dynamic=True, quantizer=OnnxRuntimeQuantizer(config))
    model, directory = export_model(architecture, tmp_path, export_config=export_config)

    assert model.get_backend() == "onnx"
    assert onnx_op_types(directory) & QUANTIZED_ONNX_OPS


@requires_nncf_onnx_quantizer
@pytest.mark.parametrize("architecture", MODELS)
def test_nncf_quantized_onnx_export(architecture, tmp_path):
    from transformers.exporters import NNCFOnnxQuantizer, OnnxConfig

    export_config = OnnxConfig(dynamic=True, quantizer=NNCFOnnxQuantizer())
    model, directory = export_model(architecture, tmp_path, export_config=export_config)

    assert model.get_backend() == "onnx"
    assert onnx_op_types(directory) & QUANTIZED_ONNX_OPS


@requires_nncf_openvino_quantizer
@pytest.mark.parametrize("architecture", MODELS)
@pytest.mark.parametrize("mode", ["quantization", "weight_compression"])
def test_nncf_quantized_openvino_export(architecture, mode, tmp_path):
    from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig, OpenVINOConfig

    config = NNCFWeightCompressionConfig() if mode == "weight_compression" else None
    export_config = OpenVINOConfig(dynamic=True, quantizer=NNCFOpenVINOQuantizer(config))
    model, directory = export_model(architecture, tmp_path, export_config=export_config)

    assert model.get_backend() == "openvino"
    op_types = openvino_op_types(directory)
    assert "FakeQuantize" in op_types if mode == "quantization" else {"i8", "u8"} & op_types


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_backend_is_the_export_format_of_the_export_config(export_format):
    from transformers.exporters import export_backend

    export_config = export_backend(export_format, "config")(dynamic=True)
    model = SentenceTransformer(MODELS["sentence_transformer"][1], model_kwargs={"export_config": export_config})
    assert model.get_backend() == export_format
    assert model._exported_model is not None


@requires_exporters
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_backend_is_the_export_format_of_the_export_config_dictionary(export_format):
    model_kwargs = {"export_config": {"export_format": export_format, "dynamic": True}}
    model = SentenceTransformer(MODELS["sentence_transformer"][1], model_kwargs=model_kwargs)
    assert model.get_backend() == export_format
    assert model._exported_model is not None


@requires_exporters
def test_export_config_exports_again_a_saved_export(tmp_path):
    from transformers.exporters import OnnxConfig

    directory = str(tmp_path / "model")
    SentenceTransformer(MODELS["sentence_transformer"][1]).save(directory)
    SentenceTransformer(directory, backend="onnx").save(directory)

    loaded = SentenceTransformer(directory, backend="onnx")
    assert loaded._exported_model.source is not None and loaded._exported_model.artifacts is None

    exported = SentenceTransformer(directory, model_kwargs={"export_config": OnnxConfig(dynamic=True)})
    assert exported._exported_model.artifacts is not None
