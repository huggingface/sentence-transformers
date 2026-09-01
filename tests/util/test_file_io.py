from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from sentence_transformers.util.file_io import (
    IncompleteSnapshotError,
    RevisionResolutionError,
    _resolve_model_revision,
    load_dir_path,
    load_file_path,
)


def test_resolve_model_revision_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve_mock = MagicMock(return_value="resolved-revision")
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", resolve_mock)

    revision = _resolve_model_revision(
        "some-org/some-model",
        "branch",
        token="token",
        cache_folder="/cache",
        local_files_only=True,
    )

    assert revision == "resolved-revision"
    resolve_mock.assert_called_once_with(
        "some-org/some-model",
        revision="branch",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
    )


def test_resolve_model_revision_is_noop_with_legacy_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", None)

    assert _resolve_model_revision("some-org/some-model", "branch") == "branch"


def test_resolve_model_revision_falls_back_when_resolution_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    resolve_mock = MagicMock(side_effect=RuntimeError("rate limited"))
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", resolve_mock)

    with caplog.at_level(logging.WARNING, logger="sentence_transformers.util.file_io"):
        assert _resolve_model_revision("some-org/some-model", "branch") == "branch"

    assert "Could not resolve the revision of 'some-org/some-model'" in caplog.text


def test_resolve_model_revision_is_quiet_when_nothing_is_cached(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    resolve_mock = MagicMock(side_effect=RevisionResolutionError("nothing cached"))
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", resolve_mock)

    with caplog.at_level(logging.WARNING, logger="sentence_transformers.util.file_io"):
        assert _resolve_model_revision("some-org/some-model", "branch", local_files_only=True) == "branch"

    assert caplog.text == ""


@pytest.mark.parametrize("error_cls", [RepositoryNotFoundError, RevisionNotFoundError])
def test_resolve_model_revision_raises_for_missing_repository_or_revision(
    error_cls: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
    resolve_mock = MagicMock(side_effect=error_cls("missing", response=response))
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", resolve_mock)

    with pytest.raises(error_cls):
        _resolve_model_revision("some-org/some-model", "branch")


@pytest.mark.parametrize("error_cls", [RepositoryNotFoundError, GatedRepoError])
def test_resolve_model_revision_uses_cache_for_inaccessible_repository(
    error_cls: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
    resolve_mock = MagicMock(side_effect=[error_cls("no token", response=response), "cached-revision"])
    monkeypatch.setattr("sentence_transformers.util.file_io._hub_resolve_revision", resolve_mock)

    assert _resolve_model_revision("some-org/some-model", "branch", cache_folder="/cache") == "cached-revision"
    assert resolve_mock.call_args.kwargs == {"revision": "branch", "cache_dir": "/cache", "local_files_only": True}


def test_load_file_path_local_missing_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the parent is a real local directory but the requested file is missing,
    `load_file_path` should return ``None`` without calling `hf_hub_download`.

    Without the #3370 guard the call would still return ``None`` (the
    `HFValidationError` raised by the Hub for a Windows-style path is in the catch
    list), so a plain return-value assertion would not detect a regression: this
    test explicitly checks that the Hub call is never issued.
    """
    hub_mock = MagicMock()
    monkeypatch.setattr("sentence_transformers.util.file_io.hf_hub_download", hub_mock)

    result = load_file_path(str(tmp_path), "modules.json", local_files_only=True)

    assert result is None
    hub_mock.assert_not_called()


def test_load_file_path_local_exists(tmp_path: Path) -> None:
    """`load_file_path` returns the local path when the file is present."""
    target = tmp_path / "config.json"
    target.write_text("{}")

    result = load_file_path(str(tmp_path), "config.json", local_files_only=True)
    assert result is not None
    assert Path(result) == target


def test_load_file_path_nonlocal_path_calls_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the path is not an existing local directory, the local-dir guard must
    not fire and `hf_hub_download` should still be invoked.
    """
    hub_mock = MagicMock(return_value="/fake/cache/modules.json")
    monkeypatch.setattr("sentence_transformers.util.file_io.hf_hub_download", hub_mock)

    result = load_file_path("some-org/some-model", "modules.json", local_files_only=True)

    assert result == "/fake/cache/modules.json"
    hub_mock.assert_called_once()


def test_load_dir_path_local_missing_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for #3370. The local-dir guard short-circuits to ``None`` and
    `snapshot_download` is never called, so callers don't see a confusing
    `HFValidationError` for what's actually just a missing local file.
    """
    snapshot_mock = MagicMock()
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path(str(tmp_path), "1_Pooling", local_files_only=True)

    assert result is None
    snapshot_mock.assert_not_called()


def test_load_dir_path_local_subfolder_exists(tmp_path: Path) -> None:
    """`load_dir_path` returns the resolved local path when the subfolder exists."""
    subfolder = tmp_path / "1_Pooling"
    subfolder.mkdir()

    result = load_dir_path(str(tmp_path), "1_Pooling", local_files_only=True)
    assert result is not None
    assert Path(result) == subfolder


def test_load_dir_path_nonlocal_path_calls_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the path is not an existing local directory, the local-dir guard must
    not fire and `snapshot_download` should still be invoked.
    """
    (tmp_path / "1_Pooling").mkdir()
    snapshot_mock = MagicMock(return_value=str(tmp_path))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling", local_files_only=True)

    assert result == str(tmp_path / "1_Pooling")
    snapshot_mock.assert_called_once()


def test_load_file_path_entry_not_found_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EntryNotFoundError` (file genuinely missing from the repo) should be swallowed
    and return ``None`` so callers can fall back to defaults (e.g. wrapping
    `bert-base-uncased` as a SentenceTransformer when `modules.json` doesn't exist).
    """
    monkeypatch.setattr(
        "sentence_transformers.util.file_io.hf_hub_download",
        MagicMock(side_effect=EntryNotFoundError("file not in repo")),
    )

    result = load_file_path("some-org/some-model", "modules.json")
    assert result is None


def test_load_file_path_transient_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient/auth errors (rate limit, 401, network) must propagate. Returning
    ``None`` silently would let `_load_modules` fall back to a vanilla transformer
    when the user actually has a SentenceTransformer model behind the failed call.
    """
    monkeypatch.setattr(
        "sentence_transformers.util.file_io.hf_hub_download",
        MagicMock(side_effect=RuntimeError("simulated rate limit")),
    )

    with pytest.raises(RuntimeError, match="simulated rate limit"):
        load_file_path("some-org/some-model", "modules.json")


def test_load_dir_path_local_entry_not_found_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LocalEntryNotFoundError` from the first `snapshot_download` (e.g. offline +
    nothing cached) should short-circuit to ``None`` without retrying the cache,
    since the retry would do the same thing.
    """
    snapshot_mock = MagicMock(side_effect=LocalEntryNotFoundError("not cached"))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling", local_files_only=True)
    assert result is None
    snapshot_mock.assert_called_once()


def test_load_dir_path_hf_validation_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HFValidationError` (malformed repo id) is unambiguously "not on the Hub".
    Short-circuit to ``None`` rather than retrying.
    """
    snapshot_mock = MagicMock(side_effect=HFValidationError("bad repo id"))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("not a repo id", "1_Pooling")
    assert result is None
    snapshot_mock.assert_called_once()


def test_load_dir_path_transient_with_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the first `snapshot_download` fails with a transient error but the cache
    has the model, `load_dir_path` should return the cached path.
    """
    (tmp_path / "1_Pooling").mkdir()
    snapshot_mock = MagicMock(side_effect=[RuntimeError("simulated network error"), str(tmp_path)])
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling")
    assert result == str(tmp_path / "1_Pooling")
    assert snapshot_mock.call_count == 2
    assert snapshot_mock.call_args_list[1].kwargs["local_files_only"] is True


def test_load_dir_path_transient_with_cache_miss_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the first call fails with a transient error and the cache also lacks the
    model, the original transient error is re-raised (not the cache miss). The cache
    miss would mask the real cause. Users see the rate-limit/auth/network error instead.
    """
    snapshot_mock = MagicMock(
        side_effect=[RuntimeError("simulated network error"), LocalEntryNotFoundError("not cached")],
    )
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    with pytest.raises(RuntimeError, match="simulated network error"):
        load_dir_path("some-org/some-model", "1_Pooling")
    assert snapshot_mock.call_count == 2


def test_load_dir_path_transient_with_partial_cache_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached snapshot that never received the subfolder is a cache miss as well."""
    snapshot_mock = MagicMock(side_effect=[RuntimeError("simulated network error"), str(tmp_path)])
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    with pytest.raises(RuntimeError, match="simulated network error"):
        load_dir_path("some-org/some-model", "1_Pooling")


def test_load_dir_path_raises_when_the_snapshot_lacks_the_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`snapshot_download` can hand back a cached snapshot without the requested subfolder when its Hub
    call fails and no tree listing is cached. That must not surface as a dangling path.
    """
    snapshot_mock = MagicMock(return_value=str(tmp_path))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    with pytest.raises(OSError, match="Could not download '1_Pooling'"):
        load_dir_path("some-org/some-model", "1_Pooling")
    snapshot_mock.assert_called_once()


def test_load_dir_path_incomplete_snapshot_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`IncompleteSnapshotError` subclasses `LocalEntryNotFoundError` but is transient, so it must
    propagate rather than turn into a `None` module path."""
    error = IncompleteSnapshotError("incomplete", "/fake/snapshot")
    snapshot_mock = MagicMock(side_effect=error)
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    with pytest.raises(IncompleteSnapshotError):
        load_dir_path("some-org/some-model", "1_Pooling")
    snapshot_mock.assert_called_once()
