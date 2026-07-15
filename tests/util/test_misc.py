from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from sentence_transformers.util.misc import append_to_last_row, import_module_class


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


def test_import_module_class_untrusted_hub_ref_warns_about_v6(monkeypatch):
    """A non-ST class ref for a Hub repo (not a local dir) without `trust_remote_code` still imports for
    now, but must emit a FutureWarning that v6.0 will require `trust_remote_code=True`. Until then this is
    the (deprecated) path that resolves an arbitrary dotted path from a model's `modules.json` `type`.
    """
    dynamic_loading_attempted = False

    def spy_get_class(*args, **kwargs):
        nonlocal dynamic_loading_attempted
        dynamic_loading_attempted = True
        raise OSError("get_class_from_dynamic_module must not be reached for an untrusted, non-local ref")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", spy_get_class)
    with pytest.warns(FutureWarning, match="trust_remote_code"):
        cls = import_module_class(
            "collections.OrderedDict",
            model_name_or_path="attacker/evil-embeddings",
            trust_remote_code=False,
        )

    # An untrusted, non-local ref must resolve via the direct-import fallback without ever attempting remote
    # (dynamic) loading. Asserting this pins the security guard and keeps the test hermetic (no Hub request).
    assert not dynamic_loading_attempted

    from collections import OrderedDict

    assert cls is OrderedDict


@pytest.mark.parametrize(
    ("use_local_path", "trust_remote_code"),
    [(True, False), (False, True)],
)
def test_import_module_class_trusted_fallback_does_not_warn(tmp_path, monkeypatch, use_local_path, trust_remote_code):
    """A trusted source whose dynamic load fails falls through to the direct import without emitting the
    untrusted-ref FutureWarning. Covers both trust sources: a local dir (implicitly trusted until v6.0) and
    an explicit `trust_remote_code=True`.
    """

    def raise_oserror(*args, **kwargs):
        raise OSError("no modeling file in repo")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", raise_oserror)
    model_name_or_path = str(tmp_path) if use_local_path else "some/repo"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cls = import_module_class(
            "collections.OrderedDict",
            model_name_or_path=model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    from collections import OrderedDict

    assert cls is OrderedDict


def test_import_module_class_local_file_collision_is_not_trusted(tmp_path, monkeypatch):
    """The local-trust carve-out requires a directory (`os.path.isdir`), not merely an existing path. A file
    whose name collides with a Hub repo id must not be treated as a trusted local model, otherwise the
    untrusted-ref gate is bypassed while `modules.json` is still fetched from the Hub.
    """
    collision = tmp_path / "repo"
    collision.write_text("not a model")

    def fail_get_class(*args, **kwargs):
        raise AssertionError("dynamic loading must not be attempted for a non-directory path")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", fail_get_class)
    with pytest.warns(FutureWarning, match="trust_remote_code"):
        cls = import_module_class(
            "collections.OrderedDict",
            model_name_or_path=str(collision),
            trust_remote_code=False,
        )

    from collections import OrderedDict

    assert cls is OrderedDict


def test_import_module_class_without_model_name_does_not_warn():
    """With no model involved (`model_name_or_path=None`), the untrusted-ref warning must not fire: Path B
    already skips a None path, and the warning text would otherwise reference a nonexistent model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cls = import_module_class("collections.OrderedDict")

    from collections import OrderedDict

    assert cls is OrderedDict


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


@pytest.mark.parametrize(
    ("content", "additional_data", "expected_return", "expected_content"),
    [
        pytest.param(
            # Two data rows, so this pins the *last* one getting extended, not the first.
            b"epoch,score\r\n0,0.5\r\n1,0.7\r\n",
            ["0.9"],
            True,
            b"epoch,score\r\n0,0.5\r\n1,0.7,0.9\r\n",
            id="appends-to-the-last-data-row",
        ),
        pytest.param(
            # The sparse evaluators pass `sparsity_stats.values()`: a dict_values of floats, not a list of strings.
            b"epoch,steps,accuracy\r\n-1,-1,0.83\r\n",
            {"active_dims": 64.5, "sparsity_ratio": 0.99}.values(),
            True,
            b"epoch,steps,accuracy\r\n-1,-1,0.83,64.5,0.99\r\n",
            id="appends-every-dict-value",
        ),
        # A header on its own isn't a data row, so there's nothing to append to.
        pytest.param(b"epoch,score\r\n", ["0.9"], False, b"epoch,score\r\n", id="noop-on-header-only"),
        pytest.param(b"", ["0.9"], False, b"", id="noop-on-empty-file"),
    ],
)
def test_append_to_last_row(tmp_path, content, additional_data, expected_return, expected_content):
    """Only writes when there's a header plus at least one data row, and then appends every value
    to that row. Otherwise it no-ops and leaves the file untouched.
    """
    csv_path = tmp_path / "results.csv"
    csv_path.write_bytes(content)

    assert append_to_last_row(str(csv_path), additional_data) is expected_return
    assert csv_path.read_bytes() == expected_content
