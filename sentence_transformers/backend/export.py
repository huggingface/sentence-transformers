from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import transformers
from huggingface_hub import hf_hub_download, list_repo_files
from packaging import version
from torch import Tensor, nn

from sentence_transformers.backend.utils import backend_should_export

if TYPE_CHECKING:
    from transformers.exporters import ExportArtifacts

    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)

EXPORT_CONFIG_FILE = "export.json"
MODEL_FILE_NAMES = {"onnx": "model.onnx", "openvino": "openvino_model.xml"}
MODEL_FILE_GLOBS = {"onnx": "*.onnx", "openvino": "openvino*.xml"}
BACKEND_NAMES = {"onnx": "ONNX", "openvino": "OpenVINO"}
# first version to include `XxxExporter`, `XxxModelRunner`, XxxQuantizer` is v5.20
MIN_TRANSFORMERS_VERSION = "5.19.0.dev0"


def is_transformers_export_available() -> tuple[bool, str]:
    """Whether the installed ``transformers`` ships the exporters and runners, with the reason if it doesn't."""
    installed = version.parse(transformers.__version__)
    if installed.release < version.parse(MIN_TRANSFORMERS_VERSION).release:
        return False, f"`transformers>={MIN_TRANSFORMERS_VERSION}` is required (found {transformers.__version__})"
    return True, ""


class _ExportWrapper(nn.Module):
    """Runs the modules of a model on keyword features and returns the requested output features."""

    def __init__(self, model: BaseModel, output_names: list[str]):
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, **features: Tensor) -> dict[str, Tensor]:
        outputs = self.model(dict(features))
        return {name: outputs[name] for name in self.output_names}


def export_model(
    model: BaseModel,
    backend: Literal["onnx", "openvino"],
    quantizer: Any = None,
) -> ExportArtifacts:
    """Export the whole model with ``backend``. The result runs as is with ``.runtime()``, or is saved to disk with
    :func:`save_export_artifacts`."""
    available, reason = is_transformers_export_available()
    if not available:
        raise ImportError(f"Cannot export the whole model to {BACKEND_NAMES[backend]}: {reason}.")

    features = model.preprocess(model._export_sample_inputs())
    sample_inputs = {key: value for key, value in features.items() if isinstance(value, Tensor)}

    with torch.no_grad():
        outputs = model(dict(sample_inputs))
    output_names = [
        name
        for name, value in outputs.items()
        if isinstance(value, Tensor) and not any(value is sample for sample in sample_inputs.values())
    ]  # Outputs that are an input as it came are returned by the features of the model already
    wrapper = _ExportWrapper(model, output_names).eval()

    if backend == "onnx":
        from transformers.exporters import OnnxConfig, OnnxExporter

        return OnnxExporter().export(wrapper, sample_inputs, config=OnnxConfig(dynamic=True, quantizer=quantizer))
    else:
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        return OpenVINOExporter().export(
            wrapper, sample_inputs, config=OpenVINOConfig(dynamic=True, stateful=False, quantizer=quantizer)
        )


def save_export_artifacts(artifacts: ExportArtifacts, directory: str | Path, backend: str) -> None:
    """Save an export of the whole model, with the file names that its legacy counterpart has."""
    directory = Path(directory)
    artifacts.save_pretrained(directory)
    if backend == "openvino":
        _rename_openvino_files(directory)


def _rename_openvino_files(directory: Path) -> None:
    """Name the OpenVINO files `openvino_model.xml` and `openvino_model.bin` and update the config to match."""
    config_path = directory / EXPORT_CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    component = config["components"]["model"]
    source = directory / component["file"]
    target = source.with_name(MODEL_FILE_NAMES["openvino"])
    if source != target:
        source.rename(target)
        source.with_suffix(".bin").rename(target.with_suffix(".bin"))
        component["file"] = target.name
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


BACKEND_MODEL_KWARGS = (
    "provider",
    "provider_options",
    "session_options",
    "ov_config",
    "file_name",
    "export",
    "subfolder",
)


def is_legacy_export(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any],
    token: bool | str | None = None,
    revision: Any = None,
) -> bool:
    """Whether the exported files were written by a previous version of sentence-transformers.

    It only exported the Transformer model, without the ``export.json`` that an export of the whole model has
    next to its graph, in ``<subfolder>/<backend>/``.
    """
    config_path = Path(model_kwargs.get("subfolder") or "", backend, EXPORT_CONFIG_FILE)
    load_path = Path(model_name_or_path)
    if load_path.exists():
        return not (load_path / config_path).exists()
    repo_files = list_repo_files(
        load_path.as_posix(),
        repo_type="model",
        revision=getattr(revision, "resolved", revision),
        token=token,
    )
    return config_path.as_posix() not in repo_files


def has_exported_files(
    model_name_or_path: str,
    backend: str,
    model_kwargs: dict[str, Any],
    token: bool | str | None = None,
    revision: Any = None,
) -> bool:
    """Whether the backend files (e.g. ``model.onnx``) to load are in ``model_name_or_path``, local or on the Hub."""
    load_path = Path(model_name_or_path)
    legacy_kwargs = {key: model_kwargs[key] for key in ("file_name", "subfolder") if key in model_kwargs}
    legacy_kwargs.update(token=token, revision=revision)
    should_export, _ = backend_should_export(
        load_path,
        load_path.exists(),
        legacy_kwargs,
        MODEL_FILE_NAMES[backend],
        MODEL_FILE_GLOBS[backend],
        BACKEND_NAMES[backend],
    )
    return not should_export


def check_can_export(model_name_or_path: str, backend: str, export: bool) -> None:
    """Raise an ``ImportError`` if the installed ``transformers`` cannot run the whole model with ``backend``.

    Both exporting the whole model and loading an export of it need the exporters and runners of ``transformers``.

    Args:
        model_name_or_path: The model name or path, for the error message.
        backend: The backend, ``"onnx"`` or ``"openvino"``.
        export: Whether the whole model is exported, as opposed to loading an export that exists.
    """
    available, reason = is_transformers_export_available()
    if available:
        return
    backend_name = BACKEND_NAMES[backend]
    if export:
        raise ImportError(
            f"No {backend_name} model was found in {model_name_or_path!r}, and the whole model cannot be exported: "
            f"{reason}."
        )
    raise ImportError(
        f"The {backend_name} model in {model_name_or_path!r} holds the whole model, which cannot be loaded: {reason}."
    )


def split_backend_kwargs(
    model_kwargs: dict[str, Any] | None, backend: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``model_kwargs`` into the ones for the backend runtime and the ones for ``from_pretrained``."""
    model_kwargs = dict(model_kwargs or {})
    backend_kwargs = {key: model_kwargs.pop(key) for key in BACKEND_MODEL_KWARGS if key in model_kwargs}
    if backend == "openvino" and "device" in model_kwargs:
        backend_kwargs["device"] = model_kwargs.pop("device")
    return backend_kwargs, model_kwargs


def read_export_config(directory: str | Path) -> dict[str, Any]:
    """The ``export.json`` of a saved export."""
    return json.loads((Path(directory) / EXPORT_CONFIG_FILE).read_text(encoding="utf-8"))


def export_files(directory: str | Path) -> list[Path]:
    """The files making up a saved export: the graph, its weights, ``export.json`` and ``config.json``."""
    directory = Path(directory)
    graph = Path(read_export_config(directory)["components"]["model"]["file"])
    # The weights are stored next to the graph: `model.bin`, or `model.onnx_data` / `model.onnx.data`
    weights = (
        [graph.with_suffix(".bin").name] if graph.suffix == ".xml" else [f"{graph.stem}.onnx_data", f"{graph}.data"]
    )
    names = [graph.name, *weights, EXPORT_CONFIG_FILE, "config.json"]
    return [directory / name for name in names if (directory / name).exists()]


def save_export(source_directory: str | Path, directory: str | Path) -> None:
    """Copy the files of a saved export into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for source in export_files(source_directory):
        target = directory / source.name
        if not (target.exists() and target.samefile(source)):
            shutil.copy(source, target)


def resolve_export_directory(
    model_name_or_path: str,
    subfolder: str | None = None,
    token: bool | str | None = None,
    revision: Any = None,
    cache_dir: str | None = None,
) -> Path:
    """The local directory of an export, in a local directory or a cached snapshot of a Hub repository."""
    if Path(model_name_or_path).is_dir():
        return Path(model_name_or_path, subfolder or "")
    config = hf_hub_download(
        model_name_or_path,
        (Path(subfolder or "") / EXPORT_CONFIG_FILE).as_posix(),
        revision=getattr(revision, "resolved", revision),
        token=token,
        cache_dir=cache_dir,
    )
    return Path(config).parent
