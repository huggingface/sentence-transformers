from __future__ import annotations

from unittest.mock import patch

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
