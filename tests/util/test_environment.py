from __future__ import annotations

import logging
import platform
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import AutoProcessor

from sentence_transformers.util.environment import (
    check_version_requirements,
    get_device_name,
    get_installed_version,
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
    # ROCm / CPU-only builds expose is_available() but do not define is_initialized. The helper must short-circuit.
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


def test_get_installed_version_python() -> None:
    assert get_installed_version("python") == platform.python_version()


def test_get_installed_version_missing_package() -> None:
    assert get_installed_version("this-package-does-not-exist") is None


def test_check_version_requirements_resolves_pytorch_alias() -> None:
    """Model authors copying the "pytorch" spelling from the `__version__` block should still work."""
    check_version_requirements({"pytorch": ">=2.0"})

    # `pip install pytorch` installs a stub package, so the error must name torch instead
    with pytest.raises(ImportError, match=r'pip install -U "torch>=99\.0"'):
        check_version_requirements({"pytorch": ">=99.0"})


def test_get_installed_version_prefers_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Editable installs report stale distribution metadata, so the imported module wins."""
    monkeypatch.setattr("sentence_transformers.__version__", "99.0.0.dev0")

    assert get_installed_version("sentence-transformers") == "99.0.0.dev0"


def test_get_installed_version_falls_back_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modules without a usable __version__ resolve via the distribution metadata."""
    monkeypatch.setattr("pytest.__version__", None)

    assert get_installed_version("pytest") == version("pytest")


@pytest.mark.parametrize(
    "requirements",
    [None, {}, {"transformers": ">=1.0"}, {"python": f">={platform.python_version()}"}, {"transformers": ""}],
    ids=["none", "empty", "satisfied", "python", "installed_only"],
)
def test_check_version_requirements_satisfied(requirements: dict[str, Any] | None) -> None:
    check_version_requirements(requirements, source="test/model")


def test_check_version_requirements_unmet() -> None:
    with pytest.raises(ImportError) as exc_info:
        check_version_requirements({"transformers": ">=999"}, source="test/model")

    message = str(exc_info.value)
    assert "test/model" in message
    assert "transformers>=999" in message
    assert 'pip install -U "transformers>=999"' in message


def test_check_version_requirements_reports_all_unmet() -> None:
    with pytest.raises(ImportError) as exc_info:
        check_version_requirements({"transformers": ">=999", "this-package-does-not-exist": ">=1.0"})

    message = str(exc_info.value)
    assert "transformers>=999" in message
    assert "this-package-does-not-exist>=1.0, but it is not installed" in message


def test_check_version_requirements_unmet_python_omits_pip_install() -> None:
    """`pip install "python>=99.0"` isn't a command anyone can run."""
    with pytest.raises(ImportError) as exc_info:
        check_version_requirements({"python": ">=99.0"}, source="test/model")

    message = str(exc_info.value)
    assert f"python>=99.0, but python=={platform.python_version()} is installed" in message
    assert "pip install" not in message


def test_check_version_requirements_unmet_python_excluded_but_others_listed() -> None:
    with pytest.raises(ImportError) as exc_info:
        check_version_requirements({"python": ">=99.0", "transformers": ">=999"})

    message = str(exc_info.value)
    assert 'pip install -U "transformers>=999"' in message
    assert '"python' not in message


def test_check_version_requirements_includes_reason() -> None:
    requirements = {"transformers": {"specifier": ">=999", "reason": "Otherwise the adapter is randomly initialized."}}

    with pytest.raises(ImportError, match="Otherwise the adapter is randomly initialized."):
        check_version_requirements(requirements)


def test_check_version_requirements_allows_prereleases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Development installs and torch nightlies must not trip the check."""
    monkeypatch.setattr("sentence_transformers.__version__", "6.0.0.dev0")

    check_version_requirements({"sentence_transformers": ">=5.0,<7"})


def test_check_version_requirements_ignores_local_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA builds report versions like 2.13.0+cu126."""
    monkeypatch.setattr("torch.__version__", "2.13.0+cu126")

    check_version_requirements({"torch": ">=2.13"})


@pytest.mark.parametrize(
    "package_name", ["transformers", "this-package-does-not-exist"], ids=["installed", "not_installed"]
)
@pytest.mark.parametrize(
    "specifier",
    [
        "5.15",
        {"specifier": "@ git+https://github.com/huggingface/transformers"},
        5.15,
        [">=5.15"],
    ],
    ids=["missing_operator", "not_a_specifier", "not_a_string", "list_of_specifiers"],
)
def test_check_version_requirements_invalid_specifier_warns(
    package_name: str, specifier: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Specifiers are validated before the installed version, so a typo can't raise for anyone."""
    with caplog.at_level(logging.WARNING):
        check_version_requirements({package_name: specifier}, source="test/model")

    assert "Could not verify" in caplog.text


def test_check_version_requirements_invalid_installed_version_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("torch.__version__", "2.13.0.unknown")

    with caplog.at_level(logging.WARNING):
        check_version_requirements({"torch": ">=2.13"})

    assert "Could not verify" in caplog.text


def test_check_version_requirements_non_dict_warns(caplog: pytest.LogCaptureFixture) -> None:
    requirements: Any = ["transformers>=999"]

    with caplog.at_level(logging.WARNING):
        check_version_requirements(requirements, source="test/model")

    assert "expected a dictionary" in caplog.text
