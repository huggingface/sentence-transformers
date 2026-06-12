from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from sentence_transformers.util.misc import import_module_class


def test_import_module_class_forwards_token_to_dynamic_module():
    """Custom remote modules in private repos require the auth token to be forwarded to
    `get_class_from_dynamic_module`. Without it, the modeling file fetch 401s, the OSError
    is silently swallowed, and the user gets a confusing ImportError instead of an auth
    error. See #3367.

    Also covers `code_revision`: when a user pins a separate revision for the modeling code
    (via `model_kwargs={"code_revision": ...}`), it must reach `get_class_from_dynamic_module`
    so the right version of the modeling file is fetched.
    """
    with patch("transformers.dynamic_module_utils.get_class_from_dynamic_module") as mock_get:
        mock_get.return_value = type("FakeClass", (), {})
        import_module_class(
            "modeling_dragon.DragonEmbedder",
            model_name_or_path="ahmad-abd/dragon-embedding-model",
            trust_remote_code=True,
            revision="main",
            code_revision="abc123",
            token="hf_my_secret_token",
            cache_folder="/tmp/my_cache",
            local_files_only=False,
        )

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["token"] == "hf_my_secret_token"
    assert call_kwargs["cache_dir"] == "/tmp/my_cache"
    assert call_kwargs["local_files_only"] is False
    assert call_kwargs["revision"] == "main"
    assert call_kwargs["code_revision"] == "abc123"


def test_import_module_class_sentence_transformers_namespace_skips_dynamic_loading():
    """Built-in `sentence_transformers.*` classes should never trigger the dynamic-module
    code path, even when token/trust_remote_code are passed.
    """
    with patch("transformers.dynamic_module_utils.get_class_from_dynamic_module") as mock_get:
        cls = import_module_class(
            "sentence_transformers.models.Pooling",
            model_name_or_path="some/repo",
            trust_remote_code=True,
            token="hf_my_secret_token",
        )

    mock_get.assert_not_called()
    from sentence_transformers.models import Pooling

    assert cls is Pooling


def test_import_module_class_local_path_without_trust_warns_about_v6(tmp_path):
    """Loading a non-ST custom class from a local path currently succeeds without
    `trust_remote_code=True` via the `os.path.exists` short-circuit. That implicit trust is being
    removed in v6.0, so a FutureWarning must telegraph the change and point at `trust_remote_code=True`.
    """
    fake_class = type("FakeClass", (), {})
    with patch("transformers.dynamic_module_utils.get_class_from_dynamic_module", return_value=fake_class):
        with pytest.warns(FutureWarning, match="trust_remote_code"):
            cls = import_module_class(
                "modeling_custom.CustomTransformer",
                model_name_or_path=str(tmp_path),
                trust_remote_code=False,
            )

    assert cls is fake_class


def test_import_module_class_local_path_with_trust_does_not_warn(tmp_path):
    """Passing `trust_remote_code=True` is the explicit opt-in that stays valid in v6.0, so the
    local-path deprecation warning must not fire.
    """
    fake_class = type("FakeClass", (), {})
    with patch("transformers.dynamic_module_utils.get_class_from_dynamic_module", return_value=fake_class):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cls = import_module_class(
                "modeling_custom.CustomTransformer",
                model_name_or_path=str(tmp_path),
                trust_remote_code=True,
            )

    assert cls is fake_class
