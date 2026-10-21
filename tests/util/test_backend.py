from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentence_transformers.backend import export_static_quantized_openvino_model
from sentence_transformers.backend.utils import backend_should_export


def test_backend_listing_uses_resolved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    class ResolvedRevision(str):
        resolved = "0123456789abcdef"

    list_repo_files_mock = MagicMock(return_value=[])
    monkeypatch.setattr("sentence_transformers.backend.utils.list_repo_files", list_repo_files_mock)

    backend_should_export(
        Path("some-org/some-model"),
        is_local=False,
        model_kwargs={"revision": ResolvedRevision("branch")},
        target_file_name="model.onnx",
        target_file_glob="*.onnx",
        backend_name="ONNX",
    )

    assert list_repo_files_mock.call_args.kwargs["revision"] == "0123456789abcdef"


@pytest.mark.parametrize("config_type", ["dict", "object", "default"])
def test_static_openvino_quantization_config(monkeypatch: pytest.MonkeyPatch, config_type: str) -> None:
    openvino = pytest.importorskip("optimum.intel.openvino")
    pytest.importorskip("datasets")
    from optimum.intel.openvino.modeling import OVModel

    config = {"num_samples": 2, "bits": 8, "weight_only": False}
    if config_type == "object":
        config = openvino.OVQuantizationConfig(num_samples=2)
    elif config_type == "default":
        config = None

    model = MagicMock(transformers_model=MagicMock(spec=OVModel))
    quantizer = MagicMock()
    monkeypatch.setattr(openvino.OVQuantizer, "from_pretrained", MagicMock(return_value=quantizer))
    save = MagicMock()
    monkeypatch.setattr("sentence_transformers.backend.quantize.save_or_push_to_hub_model", save)

    export_static_quantized_openvino_model(model, config, "unused")

    expected_samples = openvino.OVQuantizationConfig().num_samples if config is None else 2
    assert quantizer.get_calibration_dataset.call_args.kwargs["num_samples"] == expected_samples
    normalized_config = save.call_args.kwargs["config"]
    assert isinstance(normalized_config, openvino.OVQuantizationConfig)
    assert normalized_config.num_samples == expected_samples
    save.call_args.kwargs["export_function"]("output")
    assert quantizer.quantize.call_args.args == (quantizer.get_calibration_dataset.return_value,)
    assert quantizer.quantize.call_args.kwargs["ov_config"].quantization_config is normalized_config
