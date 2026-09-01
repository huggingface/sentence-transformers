from __future__ import annotations

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


@pytest.mark.parametrize("trust_remote_code", [False, None])
def test_import_module_class_local_path_without_trust_raises(tmp_path, monkeypatch, trust_remote_code):
    """A local model directory is no longer implicitly trusted (#3801): a non-ST custom class requires
    `trust_remote_code=True` regardless of where the model lives, and dynamic loading is never attempted
    without it. A forwarded None stays falsy rather than triggering transformers' interactive prompt.
    """

    def fail_get_class(*args, **kwargs):
        raise AssertionError("dynamic loading must not be attempted without trust_remote_code=True")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", fail_get_class)
    with pytest.raises(ValueError, match="trust_remote_code"):
        import_module_class(
            "modeling_custom.CustomTransformer",
            model_name_or_path=str(tmp_path),
            trust_remote_code=trust_remote_code,
        )


def test_import_module_class_untrusted_hub_ref_raises(monkeypatch):
    """A non-ST class ref without `trust_remote_code=True` is refused outright: no dynamic loading, and
    no direct-import fallback that would resolve an arbitrary dotted path from a model's `modules.json`
    `type`. The refusal also keeps the test hermetic (no Hub request).
    """

    def fail_get_class(*args, **kwargs):
        raise AssertionError("dynamic loading must not be attempted without trust_remote_code=True")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", fail_get_class)
    with pytest.raises(ValueError, match="trust_remote_code"):
        import_module_class(
            "collections.OrderedDict",
            model_name_or_path="attacker/evil-embeddings",
            trust_remote_code=False,
        )


@pytest.mark.parametrize("use_local_path", [True, False])
def test_import_module_class_trusted_fallback_imports_directly(tmp_path, monkeypatch, use_local_path):
    """A trusted ref whose dynamic load fails falls through to the direct import, for local directories
    and Hub repos alike.
    """

    def raise_oserror(*args, **kwargs):
        raise OSError("no modeling file in repo")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", raise_oserror)
    model_name_or_path = str(tmp_path) if use_local_path else "some/repo"
    cls = import_module_class(
        "collections.OrderedDict",
        model_name_or_path=model_name_or_path,
        trust_remote_code=True,
    )

    from collections import OrderedDict

    assert cls is OrderedDict


def test_import_module_class_without_model_name_needs_no_trust():
    """The trust gate only applies when a model is involved: a bare programmatic import
    (`model_name_or_path=None`) is the caller's own code choosing a class, so it needs no flag.
    """
    cls = import_module_class("collections.OrderedDict")

    from collections import OrderedDict

    assert cls is OrderedDict


def test_import_module_class_local_path_with_trust_loads(tmp_path, monkeypatch):
    """Passing `trust_remote_code=True` is the explicit opt-in that v6.0 requires, so the dynamic
    load runs and its class is returned.
    """
    fake_class = type("FakeClass", (), {})
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", lambda *args, **kwargs: fake_class
    )
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


class TestSimilarityFctName:
    """``similarity_fct_name`` is what puts a loss's scoring callable into the model card. Every
    other test reaches it through a loss's ``get_config_dict``, so the fallbacks are unpinned."""

    def test_renders_plain_functions_configured_objects_and_partials(self) -> None:
        from functools import partial

        from sentence_transformers.multi_vector_encoder.scoring import XTRScores, colbert_scores
        from sentence_transformers.util import similarity_fct_name

        assert similarity_fct_name(colbert_scores) == "colbert_scores"
        # Objects exposing get_config_dict carry their settings into the name.
        assert similarity_fct_name(XTRScores(top_k=8)) == "XTRScores(top_k=8, chunk_elements=None)"
        # A keyword-bound partial reads as a call, and a bare one collapses to the function name.
        assert similarity_fct_name(partial(colbert_scores, length_normalize=True)) == (
            "colbert_scores(length_normalize=True)"
        )
        assert similarity_fct_name(partial(colbert_scores)) == "colbert_scores"

    def test_positionally_bound_partials_lose_their_settings(self) -> None:
        """Known limitation: only ``keywords`` are rendered, so a positional binding is invisible in
        the model card. Harmless for the scoring signature, where the leading positionals are the
        embeddings, but pinned so a change is deliberate."""
        from functools import partial

        from sentence_transformers.util import maxsim, similarity_fct_name

        assert similarity_fct_name(partial(maxsim, "queries")) == "maxsim"

    def test_falls_back_to_the_type_name(self) -> None:
        from sentence_transformers.util import similarity_fct_name

        class _Scorer:
            def __call__(self, a, b): ...

        assert similarity_fct_name(_Scorer()) == "_Scorer"
