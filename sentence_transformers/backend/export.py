from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
import torch
import transformers
from packaging import version
from torch import nn
from transformers.utils import logging as transformers_logging

if TYPE_CHECKING:
    from transformers.exporters import ExportArtifacts, ExportConfigMixin

    from sentence_transformers.base.model import BaseModel

logger = transformers_logging.get_logger(__name__)

# `ExportArtifacts` introduced in v5.21
MIN_TRANSFORMERS_VERSION = "5.21"
LEGACY_EXPORT_FORMATS = ["onnx", "openvino"]
EXPORT_MANIFEST_FILE = "export.json"
LEGACY_BACKEND_FILES = {
    "onnx": ("model.onnx", "*.onnx"),
    "openvino": ("openvino_model.xml", "openvino*.xml"),
}

# Sample texts used to trace the model
_SAMPLE_TEXTS = ["Hello world!", "A somewhat longer sentence, so that the sequence dimension is not constant."]


def is_transformers_export_available() -> bool:
    """Whether the installed ``transformers`` version is compatible with exporting a model with ``transformers.exporters``"""
    installed = version.parse(transformers.__version__)
    return installed.release >= version.parse(MIN_TRANSFORMERS_VERSION).release


def require_transformers_exporters():
    """Return the ``transformers.exporters`` module, or raise an ``ImportError`` if the installed ``transformers`` is not compatible"""
    if not is_transformers_export_available():
        raise ImportError(
            f"Exporting or loading the whole model requires `transformers>={MIN_TRANSFORMERS_VERSION}` (found {transformers.__version__}). "
            "Models exported by previous versions of sentence-transformers can still be loaded."
        )
    import transformers.exporters as exporters

    return exporters


def get_export_formats() -> list[str]:
    """Return the names of the export formats registered in ``transformers.exporters``, e.g. ``["onnx", ...]``.

    Without the exporters (incompatible ``transformers`` version), only the formats of the legacy exports are returned.
    """
    if not is_transformers_export_available():
        return LEGACY_EXPORT_FORMATS
    return sorted(require_transformers_exporters().EXPORT_BACKENDS)


class _ExportableModel(nn.Module):
    """
    Wraps a model so that it can be traced: tensors in, a flat dictionary of the newly computed tensors out.
    """

    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, **features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(dict(features))
        return {
            key: value
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value is not features.get(key)
        }


def _sample_features(model: BaseModel) -> dict[str, torch.Tensor]:
    """Preprocess a small batch of texts (or text pairs for cross-encoders) to trace the model with."""
    inputs = [(text, text) for text in _SAMPLE_TEXTS] if model.model_type == "CrossEncoder" else _SAMPLE_TEXTS
    features = model.preprocess(inputs)
    return {key: value.to(model.device) for key, value in features.items() if isinstance(value, torch.Tensor)}


def resolve_backend(backend: str, export_config: ExportConfigMixin | dict[str, Any] | None) -> str:
    """
    Extract backend from ``export_config``.
    """
    if export_config is None:
        return backend
    if isinstance(export_config, dict):
        export_format = export_config.get("export_format")
        export_format = getattr(export_format, "value", export_format)
    else:
        export_format = export_config.export_format.value
    return export_format or backend


def _resolve_export_config(export_format: str, export_config: ExportConfigMixin | dict[str, Any] | None):
    from transformers.exporters import get_export_config_class

    config_class = get_export_config_class(export_format)
    if export_config is None:
        # Dynamic shapes, as texts of any batch size and length must be supported
        return config_class(dynamic=True)
    if isinstance(export_config, dict):
        arguments = {key: value for key, value in export_config.items() if key != "export_format"}
        return config_class(**{"dynamic": True, **arguments})
    return export_config


def export(
    model: BaseModel,
    export_format: str,
    export_config: ExportConfigMixin | dict[str, Any] | None = None,
    sample_features: dict[str, torch.Tensor] | None = None,
) -> ExportArtifacts:
    """
    Export all modules of a model as a single graph with ``transformers.exporters``.

    Args:
        model: The model to export.
        export_format: The export format, i.e. one of the formats registered in ``transformers.exporters``
            such as ``"onnx"`` or ``"openvino"``.
        export_config: The export configuration, either an export config object (e.g.
            :class:`~transformers.exporters.OnnxConfig`) or a dictionary of its arguments. Defaults to an export
            config with dynamic shapes.
        sample_features: The tensors to trace the model with, as returned by ``model.preprocess(...)``. Defaults to
            the preprocessed output of two short texts (or text pairs for a cross-encoder).

    Returns:
        ExportArtifacts: The exported graph. Save it with ``save_pretrained``.
    """
    exporters = require_transformers_exporters()
    config = _resolve_export_config(export_format, export_config)
    exporter = exporters.AutoHfExporter.from_config(config)
    if sample_features is None:
        sample_features = _sample_features(model)

    return exporter.export(_ExportableModel(model), sample_features, config)


class ExportedModule:
    """Runs an exported graph on the features of a model, in place of its modules, and saves the export."""

    def __init__(self, artifacts: ExportArtifacts, device: str | None = None):
        self.artifacts = artifacts
        self.model = artifacts.runtime(device=device)
        # TODO: get device from self.model and remove
        self._device = torch.device(device) if device is not None else None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def input_names(self) -> tuple[str, ...]:
        """The names of the forward kwargs the graph was traced with."""
        return tuple(self.model.runner.metadata.kwargs)

    def __call__(self, features: dict[str, Any], **kwargs) -> dict[str, Any]:
        if kwargs:
            logger.warning_once(
                f"The forward keyword arguments {kwargs} are ignored, as exported graph was traced without them. "
            )
        missing = [name for name in self.input_names if name not in features]
        if missing:
            raise ValueError(
                f"The exported model takes the features {list(self.input_names)}, but {missing} are missing from "
                f"the preprocessed inputs {sorted(features)}."
            )

        features.update(self.model(**features))
        return features

    def save(self, output_path: str | os.PathLike) -> None:
        """Save the exported graph and its manifest to ``output_path``."""
        self.artifacts.save_pretrained(output_path)


def has_export_config(model_name_or_path: str | Path, export_format: str, **kwargs) -> bool:
    """
    Whether ``model_name_or_path``, a local directory or a Hub model, holds a model exported with
    ``transformers.exporters``. It is told by its manifest (``export.json``), which the exports of previous versions of
    sentence-transformers, that only hold the Transformer model, don't have.

    Args:
        model_name_or_path: The local directory or Hub model ID.
        export_format: whether the model was exported to this format, e.g. ``"onnx"``.
        **kwargs: ``revision``, ``token``, ``cache_dir`` and ``local_files_only`` to locate a Hub model.
    """
    if os.path.isdir(model_name_or_path):
        manifest = Path(model_name_or_path, EXPORT_MANIFEST_FILE)
    else:
        try:
            manifest = Path(huggingface_hub.hf_hub_download(str(model_name_or_path), EXPORT_MANIFEST_FILE, **kwargs))
        except (huggingface_hub.errors.HfHubHTTPError, huggingface_hub.errors.LocalEntryNotFoundError):
            return False  # There is no manifest, or no such model, or it is not in the cache
    if not manifest.is_file():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("export_format") == export_format


def load_exported_model(
    model_name_or_path: str, export_format: str, device: str | torch.device | None = None, **kwargs
) -> ExportedModule | None:
    """
    Load the model exported with ``transformers.exporters`` in a local directory or Hugging Face repository.

    Args:
        model_name_or_path: Local directory or Hub model ID.
        export_format: The export format the model must have been exported to, e.g. ``"onnx"``.
        device: The device the exported graph should run on.
        **kwargs: ``revision``, ``token``, ``cache_dir`` and ``local_files_only`` to locate the model.

    Returns:
        ExportedModule | None: The loaded model, or None if the model was not exported to ``export_format``.
    """
    if not has_export_config(model_name_or_path, export_format, **kwargs):
        return None
    exporters = require_transformers_exporters()

    # ``ExportArtifacts.from_pretrained`` do not take ``local_files_only``, remove once the case
    kwargs = {key: value for key, value in kwargs.items() if key != "local_files_only"}
    artifacts = exporters.ExportArtifacts.from_pretrained(model_name_or_path, **kwargs)
    return ExportedModule(artifacts, device=device)
