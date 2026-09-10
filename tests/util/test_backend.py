from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
