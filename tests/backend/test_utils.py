from __future__ import annotations

from pathlib import Path

import pytest

from sentence_transformers.backend import utils
from sentence_transformers.backend.utils import backend_should_export


def should_export(
    layout: list[str],
    model_kwargs: dict[str, str],
    is_local: bool,
    target_file_name: str,
    target_file_glob: str,
    backend_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> bool:
    """Run the export decision over ``layout``, either as a local directory or as a Hub repository."""
    if is_local:
        for file_name in layout:
            path = tmp_path / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        load_path = tmp_path
    else:
        monkeypatch.setattr(utils, "list_repo_files", lambda *args, **kwargs: list(layout))
        load_path = Path("sentence-transformers-testing/some-model")

    export, _ = backend_should_export(
        load_path, is_local, dict(model_kwargs), target_file_name, target_file_glob, backend_name
    )
    return export


@pytest.mark.parametrize(
    ["target_file_name", "target_file_glob", "backend_name"],
    [("model.onnx", "*.onnx", "ONNX"), ("openvino_model.xml", "openvino*.xml", "OpenVINO")],
    ids=["onnx", "openvino"],
)
@pytest.mark.parametrize(
    ["layout", "model_kwargs", "expected_export"],
    [
        (["config.json", "{file}"], {}, False),
        (["config.json", "{backend}/{file}"], {}, False),
        (["config.json", "{backend}/{file}"], {"subfolder": "{backend}"}, False),
        (["config.json", "{backend}/nested/{file}"], {"subfolder": "{backend}"}, True),
        (["config.json", "model.safetensors"], {}, True),
    ],
    ids=["in the root", "in the backend subfolder", "in a given subfolder", "below a given subfolder", "absent"],
)
def test_backend_should_export_agrees_between_a_directory_and_a_repository(
    layout: list[str],
    model_kwargs: dict[str, str],
    expected_export: bool,
    target_file_name: str,
    target_file_glob: str,
    backend_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One layout must give one decision, whether it is read from disk or from the Hub.

    The two branches used to disagree: a local directory is globbed with pathlib, where ``**/`` means
    zero or more directories, while a repository was matched with ``fnmatch`` on that same pattern.
    ``fnmatch`` knows no ``**``, and its ``*`` already crosses directories, so ``**/*.onnx`` compiled
    to a regex that demanded a directory. A model file in the root of a repository never matched, and
    the model was re-exported even though it was right there -- which fails outright for a repository
    that ships no torch weights to export from, such as ``Qdrant/all-MiniLM-L6-v2-onnx``.
    """
    backend = backend_name.lower()
    layout = [entry.format(file=target_file_name, backend=backend) for entry in layout]
    model_kwargs = {key: value.format(backend=backend) for key, value in model_kwargs.items()}

    arguments = (target_file_name, target_file_glob, backend_name, tmp_path, monkeypatch)
    assert should_export(layout, model_kwargs, True, *arguments) is expected_export
    assert should_export(layout, model_kwargs, False, *arguments) is expected_export
