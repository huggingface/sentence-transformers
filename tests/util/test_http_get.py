from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest

from sentence_transformers.util import file_io
from sentence_transformers.util.file_io import httpx


@pytest.mark.parametrize("relative_path", ["embeddings.bin", "downloads/embeddings.bin"])
@pytest.mark.parametrize("existing", [False, True])
def test_http_get_publishes_complete_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, existing: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path(relative_path)
    if existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous download")
    content = b"complete download" * 1024
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=content))

    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        file_io.http_get("https://example.test/embeddings", relative_path)

    assert path.read_bytes() == content
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("second_fails", [False, True], ids=["both-succeed", "one-fails"])
def test_http_get_isolates_concurrent_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_fails: bool
) -> None:
    path = tmp_path / "downloads" / "embeddings.bin"
    path.parent.mkdir()
    path.write_bytes(b"previous download")
    first_started = Event()
    release_first = Event()
    first_prefix = b"a" * 1024
    first_suffix = b"first download tail"
    second_content = b"b" * 2048

    class DownloadStream(httpx.SyncByteStream):
        def __init__(self, first: bool) -> None:
            self.first = first

        def __iter__(self) -> Iterator[bytes]:
            if self.first:
                # Fill one iter_bytes chunk before allowing the other download to start.
                yield first_prefix
                first_started.set()
                if not release_first.wait(timeout=10):
                    raise RuntimeError("Timed out waiting for the second download")
                yield first_suffix
            else:
                yield second_content
                if second_fails:
                    raise httpx.ReadError("Interrupted second download")

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=DownloadStream(first=request.url.path == "/first"))

    progress = MagicMock()
    monkeypatch.setattr(file_io, "tqdm", MagicMock(return_value=progress))
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(file_io.http_get, "https://example.test/first", str(path))
            try:
                assert first_started.wait(timeout=10)
                second = executor.submit(file_io.http_get, "https://example.test/second", str(path))
                if second_fails:
                    with pytest.raises(httpx.ReadError, match="Interrupted second download"):
                        second.result(timeout=10)
                    assert path.read_bytes() == b"previous download"
                else:
                    second.result(timeout=10)
                    assert path.read_bytes() == second_content
            finally:
                release_first.set()
            first.result(timeout=10)

    assert path.read_bytes() == first_prefix + first_suffix
    assert list(path.parent.iterdir()) == [path]
    assert progress.close.call_count == 2


def test_http_get_preserves_destination_on_stream_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "embeddings.bin"
    path.write_bytes(b"previous download")

    class InterruptedStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b"partial content" * 1024
            raise httpx.ReadError("Interrupted download")

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=InterruptedStream())

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        with pytest.raises(httpx.ReadError, match="Interrupted download"):
            file_io.http_get("https://example.test/embeddings", str(path))

    assert path.read_bytes() == b"previous download"
    assert list(tmp_path.iterdir()) == [path]


def test_http_get_cleans_temporary_directory_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "embeddings.bin"
    path.write_bytes(b"previous download")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"complete download"))
    monkeypatch.setattr(file_io.os, "replace", MagicMock(side_effect=PermissionError("Cannot replace destination")))

    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        with pytest.raises(PermissionError, match="Cannot replace destination"):
            file_io.http_get("https://example.test/embeddings", str(path))

    assert path.read_bytes() == b"previous download"
    assert list(tmp_path.iterdir()) == [path]
