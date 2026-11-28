from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub import constants

from sentence_transformers.backend import export_static_quantized_openvino_model
from sentence_transformers.backend.utils import backend_should_export


def test_backend_listing_uses_resolved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    class ResolvedRevision(str):
        resolved = "0123456789abcdef"

    list_repo_files_mock = MagicMock(return_value=[])
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
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


@pytest.mark.parametrize(
    "backend_name, file_name, file_glob",
    [("ONNX", "model.onnx", "*.onnx"), ("OpenVINO", "openvino_model.xml", "openvino*.xml")],
)
@pytest.mark.parametrize("offline", [False, True], ids=["local_files_only", "HF_HUB_OFFLINE"])
def test_backend_listing_uses_cached_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    file_name: str,
    file_glob: str,
    offline: bool,
) -> None:
    repo_cache = tmp_path / "models--some-org--some-model"
    commit = "a" * 40
    refs = repo_cache / "refs"
    refs.mkdir(parents=True)
    (refs / "test-branch").write_text(commit, encoding="utf-8")
    backend = backend_name.lower()
    model_dir = repo_cache / "snapshots" / commit / backend
    model_dir.mkdir(parents=True)
    (model_dir / file_name).touch()

    list_repo_files_mock = MagicMock(side_effect=AssertionError("Cache-only loading must not query the Hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setattr("sentence_transformers.backend.utils.list_repo_files", list_repo_files_mock)
    export, model_kwargs = backend_should_export(
        Path("some-org/some-model"),
        is_local=False,
        model_kwargs={"cache_dir": str(tmp_path), "revision": "test-branch", "local_files_only": not offline},
        target_file_name=file_name,
        target_file_glob=file_glob,
        backend_name=backend_name,
    )

    list_repo_files_mock.assert_not_called()
    assert not export
    assert model_kwargs["subfolder"] == backend
    assert model_kwargs["file_name"] == file_name


def test_backend_listing_keeps_export_fallback_on_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.backend.utils.list_repo_files",
        MagicMock(side_effect=AssertionError("Cache-only loading must not query the Hub")),
    )
    export, _ = backend_should_export(
        Path("some-org/some-model"),
        is_local=False,
        model_kwargs={"cache_dir": str(tmp_path), "local_files_only": True},
        target_file_name="model.onnx",
        target_file_glob="*.onnx",
        backend_name="ONNX",
    )
    assert export


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
