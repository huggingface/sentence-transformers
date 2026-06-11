from __future__ import annotations

from types import SimpleNamespace

import pytest
from transformers import AutoProcessor

from sentence_transformers.util.environment import (
    get_device_name,
    is_dist_initialized,
    suggest_extra_on_exception,
)


def test_get_device_name_cuda_without_distributed_is_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    fake_distributed = SimpleNamespace(is_available=lambda: False)

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr("sentence_transformers.util.environment.torch.cuda", fake_cuda)
    monkeypatch.setattr("sentence_transformers.util.environment.torch.distributed", fake_distributed)

    assert get_device_name() == "cuda:0"


def test_is_dist_initialized_returns_false_when_distributed_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # ROCm / CPU-only builds expose is_available() but do not define is_initialized; the helper must short-circuit.
    monkeypatch.setattr(
        "sentence_transformers.util.environment.torch.distributed",
        SimpleNamespace(is_available=lambda: False),
    )

    assert is_dist_initialized() is False


@pytest.mark.parametrize(
    ("exception_cls", "message", "expected_snippet"),
    [
        (ImportError, "No module named 'PIL'", r"sentence-transformers\[image\]"),
        (ImportError, "No module named 'pillow'", r"sentence-transformers\[image\]"),
        (ImportError, "No module named 'torchvision'", r"sentence-transformers\[image\]"),
        (ImportError, "No module named 'torchcodec'", r"pip install -U torchcodec"),
        (AttributeError, "module has no attribute 'read_video'", r"sentence-transformers\[video\]"),
        (ImportError, "No module named 'soundfile'", r"sentence-transformers\[audio\]"),
        (ImportError, "No module named 'librosa'", r"sentence-transformers\[audio\]"),
    ],
)
def test_suggest_extra_on_exception_adds_hint(
    exception_cls: type[Exception], message: str, expected_snippet: str
) -> None:
    with pytest.raises(exception_cls, match=expected_snippet):
        with suggest_extra_on_exception():
            raise exception_cls(message)


def test_suggest_extra_on_exception_preserves_exception_type() -> None:
    with pytest.raises(AttributeError):
        with suggest_extra_on_exception():
            raise AttributeError("module has no attribute 'read_video'")


def test_suggest_extra_on_exception_reraises_unmatched() -> None:
    with pytest.raises(ImportError, match="some unknown module"):
        with suggest_extra_on_exception():
            raise ImportError("some unknown module")


def test_suggest_extra_on_exception_ignores_other_exceptions() -> None:
    with pytest.raises(ValueError, match="not an import error"):
        with suggest_extra_on_exception():
            raise ValueError("not an import error")


def test_suggest_extra_on_exception_no_exception() -> None:
    with suggest_extra_on_exception():
        pass


def test_suggest_extra_monkeypatch_missing_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate AutoProcessor.from_pretrained raising ImportError for missing Pillow."""

    def mock_from_pretrained(*args, **kwargs):
        raise ImportError("No module named 'PIL'")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", mock_from_pretrained)

    with pytest.raises(ImportError, match=r"sentence-transformers\[image\]"):
        with suggest_extra_on_exception():
            AutoProcessor.from_pretrained("some-model")


def test_suggest_extra_monkeypatch_missing_torchcodec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate AutoProcessor.from_pretrained raising AttributeError for missing torchcodec."""

    def mock_from_pretrained(*args, **kwargs):
        raise AttributeError("module 'torchcodec' has no attribute 'decoders'")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", mock_from_pretrained)

    with pytest.raises(AttributeError, match=r"pip install -U torchcodec"):
        with suggest_extra_on_exception():
            AutoProcessor.from_pretrained("some-model")
