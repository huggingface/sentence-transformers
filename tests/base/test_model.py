from __future__ import annotations

import json
import logging
import os
import warnings
from collections import UserDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from torch import Tensor, nn

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.evaluation import SentenceEvaluator


class BaseModelPreprocessTest:
    def create_dense_text_model(
        self, request: pytest.FixtureRequest
    ) -> Any:  # pragma: no cover - to be implemented by subclasses
        """Return a dense text model instance for the concrete test class.

        Subclasses should use ``request.getfixturevalue(...)`` to obtain the
        concrete model fixture, e.g. ``stsb_bert_tiny_model`` or
        ``splade_bert_tiny_model``.
        """
        raise NotImplementedError

    def create_text_inputs(self) -> list:  # pragma: no cover - to be implemented by subclasses
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def _setup_model(self, request: pytest.FixtureRequest) -> None:
        self.model = self.create_dense_text_model(request)

    @pytest.fixture(autouse=True)
    def _setup_inputs(self) -> None:
        self.inputs = self.create_text_inputs()

    def test_preprocess(self) -> None:
        model = self.model
        inputs = self.inputs
        input_length = len(inputs)
        features = model.preprocess(inputs)

        assert isinstance(features, (dict, UserDict))

        for key, value in features.items():
            if isinstance(value, Tensor):
                assert value.size(0) == input_length

    def test_preprocess_accepts_prompt(self) -> None:
        model = self.model
        inputs = self.inputs

        features_without_prompt = model.preprocess(inputs)
        features_with_prompt = model.preprocess(inputs, prompt="Instruction: ")

        assert isinstance(features_without_prompt, (dict, UserDict))
        assert isinstance(features_with_prompt, (dict, UserDict))
        assert set(features_without_prompt.keys()).issubset(features_with_prompt.keys())
        assert any(
            not torch.equal(features_without_prompt[key], features_with_prompt[key])
            for key in features_without_prompt.keys()
            if isinstance(features_without_prompt[key], Tensor)
        )


class TestSentenceTransformerPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> SentenceTransformer:
        return request.getfixturevalue("stsb_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return ["This is a test.", "Another test sentence."]


class TestCrossEncoderPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> CrossEncoder:
        return request.getfixturevalue("reranker_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return [("This is a test.", "This is a test."), ("Another test sentence.", "Yet another sentence.")]


class TestSparseEncoderPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> SparseEncoder:
        return request.getfixturevalue("splade_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return ["This is a test.", "Another test sentence."]


def test_preprocess_rejects_unsupported_modality(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess() should raise ValueError when the inferred modality is not supported."""
    monkeypatch.setattr(
        "sentence_transformers.base.model.infer_batch_modality",
        lambda inputs: "image",
    )
    with pytest.raises(ValueError, match="Modality 'image' is not supported"):
        stsb_bert_tiny_model.preprocess(["dummy text"])


def test_preprocess_rejects_multimodal_without_message_support(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess() should raise ValueError for combined modalities when the model supports
    individual modalities (e.g. text and image) but not 'message' format.

    This mirrors the behavior of architectures like blip-2, sam3, and flava that handle
    text and image separately but cannot combine them in a single forward pass.
    """
    # Pretend the model supports text and image individually, but not message format
    monkeypatch.setattr(type(stsb_bert_tiny_model), "modalities", property(lambda self: ["text", "image"]))
    # Pretend the inferred modality is a combined tuple
    monkeypatch.setattr(
        "sentence_transformers.base.model.infer_batch_modality",
        lambda inputs: ("text", "image"),
    )
    with pytest.raises(
        ValueError, match=r"This model supports text and image individually, but not in the same input"
    ):
        stsb_bert_tiny_model.preprocess(["dummy text"])


def test_preprocess_passes_supported_modality(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """preprocess() should succeed with text inputs on a text-only model."""
    features = stsb_bert_tiny_model.preprocess(["Hello world"])
    assert isinstance(features, (dict, UserDict))
    assert "input_ids" in features


def test_preprocess_skips_modality_check_for_empty_inputs(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess() should skip the modality check when inputs is empty."""

    def fail_if_called(inputs):
        raise AssertionError("infer_batch_modality should not be called for empty inputs")

    monkeypatch.setattr("sentence_transformers.base.model.infer_batch_modality", fail_if_called)
    with pytest.raises((IndexError, Exception)):
        stsb_bert_tiny_model.preprocess([])


def test_sentence_transformer(stsb_bert_tiny_model: SentenceTransformer) -> None:
    assert stsb_bert_tiny_model.supports("text")
    assert not stsb_bert_tiny_model.supports("image")
    assert not stsb_bert_tiny_model.supports("audio")
    assert not stsb_bert_tiny_model.supports("video")
    assert not stsb_bert_tiny_model.supports("message")
    assert not stsb_bert_tiny_model.supports(("image", "text"))


def test_sparse_encoder(splade_bert_tiny_model: SparseEncoder) -> None:
    assert splade_bert_tiny_model.supports("text")
    assert not splade_bert_tiny_model.supports("image")
    assert not splade_bert_tiny_model.supports(("image", "text"))


def test_cross_encoder(reranker_bert_tiny_model: CrossEncoder) -> None:
    assert reranker_bert_tiny_model.supports("text")
    assert not reranker_bert_tiny_model.supports("image")
    assert not reranker_bert_tiny_model.supports(("image", "text"))


@pytest.mark.parametrize(
    "modalities, modality, expected",
    [
        (["text"], "text", True),
        (["text"], "image", False),
        (["text", "image"], "text", True),
        (["text", "image"], "image", True),
        (["text", "image"], "audio", False),
        (["text", "message"], "image", False),
        (["text", ("image", "text")], ("image", "text"), True),
        (["text", "image"], ("image", "text"), False),
        (["text", "audio"], ("audio", "text"), False),
        (["text", "image", "message"], ("image", "text"), True),
        (["text", "audio", "message"], ("audio", "text"), True),
        (["text", "image", "audio", "message"], ("image", "text"), True),
        (["text", "image", "audio", "message"], ("audio", "text"), True),
        (["text", "image", "audio", "message"], ("audio", "image"), True),
        (["text", "image", "audio", "message"], ("audio", "image", "text"), True),
        (["text", "message"], ("image", "text"), False),
        (["text", "image", "message"], ("audio", "text"), False),
        (["text", "image", "message"], ("video", "text"), False),
        (["text", "image", "audio", "video", "message"], ("video", "audio", "image", "text"), True),
        (["text", "image", "audio", "video", "message"], "video", True),
        (["text", "image", "audio", "video", "message"], "message", True),
        (["text", "message"], "message", True),
        (["text"], "message", False),
    ],
    ids=[
        "text_only-supports_text",
        "text_only-rejects_image",
        "text_image-supports_text",
        "text_image-supports_image",
        "text_image-rejects_audio",
        "text_message-rejects_image",
        "explicit_tuple-supports_image_text",
        "text_image-rejects_tuple_without_message",
        "text_audio-rejects_tuple_without_message",
        "text_image_message-supports_image_text_tuple",
        "text_audio_message-supports_audio_text_tuple",
        "text_image_audio_message-supports_image_text_tuple",
        "text_image_audio_message-supports_audio_text_tuple",
        "text_image_audio_message-supports_audio_image_tuple",
        "text_image_audio_message-supports_triple_tuple",
        "text_message-rejects_tuple_with_missing_part",
        "text_image_message-rejects_audio_text_tuple",
        "text_image_message-rejects_video_text_tuple",
        "all_modalities-supports_quad_tuple",
        "all_modalities-supports_video",
        "all_modalities-supports_message",
        "text_message-supports_message",
        "text_only-rejects_message",
    ],
)
def test_supports_parametrized(
    monkeypatch: pytest.MonkeyPatch,
    stsb_bert_tiny_model: SentenceTransformer,
    modalities: list,
    modality: str | tuple,
    expected: bool,
) -> None:
    monkeypatch.setattr(type(stsb_bert_tiny_model), "modalities", property(lambda self: modalities))
    assert stsb_bert_tiny_model.supports(modality) == expected


def test_use_auth_token_warns() -> None:
    """Passing use_auth_token should emit a FutureWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SentenceTransformer(modules=[], use_auth_token="fake-token")
        future_warnings = [
            x for x in w if issubclass(x.category, FutureWarning) and "use_auth_token" in str(x.message)
        ]
        assert len(future_warnings) == 1


def test_use_auth_token_and_token_raises() -> None:
    """Passing both token and use_auth_token should raise ValueError."""
    with pytest.raises(ValueError, match="Both `token` and `use_auth_token`"):
        SentenceTransformer(modules=[], token="tok", use_auth_token="tok2")


def test_cache_folder_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cache_folder is None, it should fall back to SENTENCE_TRANSFORMERS_HOME env var."""
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/tmp/fake_cache")
    model = SentenceTransformer(modules=[])
    assert model is not None


def test_save_creates_expected_files(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """save() should create config_sentence_transformers.json and modules.json."""
    stsb_bert_tiny_model.save(str(tmp_path))

    config_path = tmp_path / "config_sentence_transformers.json"
    modules_path = tmp_path / "modules.json"
    assert config_path.exists()
    assert modules_path.exists()

    config = json.loads(config_path.read_text(encoding="utf8"))
    assert "__version__" in config

    modules_config = json.loads(modules_path.read_text(encoding="utf8"))
    assert isinstance(modules_config, list)
    assert len(modules_config) > 0
    for module_entry in modules_config:
        assert "idx" in module_entry
        assert "name" in module_entry
        assert "path" in module_entry
        assert "type" in module_entry


def test_save_none_path_is_noop(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """save(None) should return immediately without error."""
    stsb_bert_tiny_model.save(None)  # type: ignore[arg-type]


def test_save_pretrained_delegates_to_save(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """save_pretrained() should produce the same files as save()."""
    stsb_bert_tiny_model.save_pretrained(str(tmp_path))
    assert (tmp_path / "config_sentence_transformers.json").exists()
    assert (tmp_path / "modules.json").exists()


def test_save_without_model_card(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """save(create_model_card=False) should not create README.md."""
    stsb_bert_tiny_model.save(str(tmp_path), create_model_card=False)
    assert not (tmp_path / "README.md").exists()
    assert (tmp_path / "modules.json").exists()


def test_save_with_safe_serialization_fallback(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a module's save() doesn't accept safe_serialization, fall back to save without it."""
    original_save = type(stsb_bert_tiny_model[0]).save
    call_log = []

    def patched_save(self, path, safe_serialization=None):
        if safe_serialization is not None:
            call_log.append("safe")
            raise TypeError("unexpected keyword argument 'safe_serialization'")
        call_log.append("fallback")
        return original_save(self, path)

    monkeypatch.setattr(type(stsb_bert_tiny_model[0]), "save", patched_save)

    stsb_bert_tiny_model.save(str(tmp_path))
    assert "safe" in call_log
    assert "fallback" in call_log


def test_save_load_roundtrip(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """A model saved and re-loaded should produce the same embeddings."""
    sentences = ["Hello world", "Testing roundtrip"]
    original_embeddings = stsb_bert_tiny_model.encode(sentences)

    stsb_bert_tiny_model.save(str(tmp_path))
    loaded_model = SentenceTransformer(str(tmp_path))
    loaded_embeddings = loaded_model.encode(sentences)

    np.testing.assert_allclose(original_embeddings, loaded_embeddings, atol=1e-5)


def test_save_load_roundtrip_sparse_encoder(splade_bert_tiny_model: SparseEncoder, tmp_path: Path) -> None:
    """A SparseEncoder model saved and re-loaded should produce the same embeddings."""
    sentences = ["Hello world", "Testing roundtrip"]
    encode_kwargs = {"convert_to_sparse_tensor": False, "save_to_cpu": True}
    original_embeddings = splade_bert_tiny_model.encode(sentences, **encode_kwargs)

    splade_bert_tiny_model.save(str(tmp_path))
    loaded_model = SparseEncoder(str(tmp_path))
    loaded_embeddings = loaded_model.encode(sentences, **encode_kwargs)

    np.testing.assert_allclose(original_embeddings, loaded_embeddings, atol=1e-5)


def test_save_load_roundtrip_cross_encoder(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path) -> None:
    """A CrossEncoder model saved and re-loaded should produce the same predictions."""
    pairs = [("Hello", "World"), ("Testing", "Roundtrip")]
    original_preds = reranker_bert_tiny_model.predict(pairs)

    reranker_bert_tiny_model.save(str(tmp_path))
    loaded_model = CrossEncoder(str(tmp_path))
    loaded_preds = loaded_model.predict(pairs)

    np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-5)


class _FakeEvaluator(SentenceEvaluator):
    def __init__(self, result: dict[str, float]) -> None:
        super().__init__()
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        self.calls.append((model, output_path))
        return self.result


def test_evaluate_calls_evaluator(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """evaluate() should call the evaluator and return its result."""
    evaluator = _FakeEvaluator({"score": 0.95})

    result = stsb_bert_tiny_model.evaluate(evaluator)
    assert result == {"score": 0.95}
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0] == (stsb_bert_tiny_model, None)


def test_evaluate_creates_output_dir(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """evaluate() should create output_path directory if it doesn't exist."""
    evaluator = _FakeEvaluator({"score": 0.9})
    output_path = str(tmp_path / "eval_output" / "nested")

    result = stsb_bert_tiny_model.evaluate(evaluator, output_path=output_path)

    assert result == {"score": 0.9}
    assert os.path.isdir(output_path)
    assert evaluator.calls[0] == (stsb_bert_tiny_model, output_path)


def test_load_module_class_from_ref_sentence_transformers(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A sentence_transformers.* class ref should be imported directly."""
    from sentence_transformers.sentence_transformer.modules import Pooling

    cls = stsb_bert_tiny_model._load_module_class_from_ref(
        "sentence_transformers.sentence_transformer.modules.Pooling",
        model_name_or_path="unused",
        trust_remote_code=False,
        revision=None,
        model_kwargs=None,
    )
    assert cls is Pooling


def test_load_module_class_from_ref_fallback_to_import(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A non-sentence_transformers class ref without trust_remote_code should fall back to import_from_string."""
    cls = stsb_bert_tiny_model._load_module_class_from_ref(
        "torch.nn.Linear",
        model_name_or_path="nonexistent_path_12345",
        trust_remote_code=False,
        revision=None,
        model_kwargs=None,
    )
    assert cls is nn.Linear


def test_load_module_class_from_ref_trust_remote_code_fallback(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With trust_remote_code=True, should try dynamic import first, then fall back on error."""

    def mock_get_class(class_ref, model_name_or_path, **kwargs):
        raise OSError("not found")

    monkeypatch.setattr(
        "sentence_transformers.base.model.get_class_from_dynamic_module",
        mock_get_class,
    )

    cls = stsb_bert_tiny_model._load_module_class_from_ref(
        "torch.nn.ReLU",
        model_name_or_path="nonexistent_path_12345",
        trust_remote_code=True,
        revision=None,
        model_kwargs=None,
    )
    assert cls is nn.ReLU


def test_load_module_class_from_ref_code_revision(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With trust_remote_code=True and code_revision in model_kwargs, it should pop code_revision."""
    captured_kwargs = {}

    def mock_get_class(class_ref, model_name_or_path, **kwargs):
        captured_kwargs.update(kwargs)
        raise ValueError("not found")

    monkeypatch.setattr(
        "sentence_transformers.base.model.get_class_from_dynamic_module",
        mock_get_class,
    )

    model_kwargs = {"code_revision": "v1.0", "other": "value"}
    stsb_bert_tiny_model._load_module_class_from_ref(
        "torch.nn.ReLU",
        model_name_or_path="nonexistent_path_12345",
        trust_remote_code=True,
        revision="main",
        model_kwargs=model_kwargs,
    )
    assert "code_revision" not in model_kwargs
    assert captured_kwargs.get("code_revision") == "v1.0"


def _setup_hub_mocks(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Set up common mocks for push_to_hub tests. Returns a dict to capture upload_folder kwargs."""
    mock_upload_folder_kwargs = {}

    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    def mock_upload_folder(self, **kwargs):
        mock_upload_folder_kwargs.update(kwargs)
        commit_hash = "123456" if kwargs.get("revision") is None else "678901"
        commit_info_kwargs = {
            "commit_url": f"https://huggingface.co/{kwargs.get('repo_id')}/commit/{commit_hash}",
            "commit_message": "commit_message",
            "commit_description": "commit_description",
            "oid": "oid",
            "pr_url": f"https://huggingface.co/{kwargs.get('repo_id')}/discussions/123",
        }
        try:
            return CommitInfo(**commit_info_kwargs)
        except TypeError:
            return CommitInfo(**commit_info_kwargs, _endpoint=None)

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    return mock_upload_folder_kwargs


def test_push_to_hub_sparse_encoder(splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_kwargs = _setup_hub_mocks(monkeypatch)
    model = splade_bert_tiny_model

    url = model.push_to_hub("sparse-encoder-testing/splade-bert-tiny-nq")
    assert mock_kwargs["repo_id"] == "sparse-encoder-testing/splade-bert-tiny-nq"
    assert url == "https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq/commit/123456"
    mock_kwargs.clear()

    url = model.push_to_hub("sparse-encoder-testing/splade-bert-tiny-nq", create_pr=True)
    assert url == "https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq/discussions/123"
    mock_kwargs.clear()

    url = model.push_to_hub("sparse-encoder-testing/splade-bert-tiny-nq", revision="test-branch")
    assert mock_kwargs["revision"] == "test-branch"
    assert url == "https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq/commit/678901"


def test_save_to_hub_deprecation_warning(
    splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """save_to_hub should log a deprecation warning."""
    _setup_hub_mocks(monkeypatch)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = splade_bert_tiny_model.save_to_hub("sparse-encoder-testing/splade-bert-tiny-nq")
        assert url == "https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq/commit/123456"
        assert any("save_to_hub" in record.message and "deprecated" in record.message for record in caplog.records)


def test_save_to_hub_organization_conflict_raises(
    splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_to_hub with conflicting organization in repo_id should raise ValueError."""
    _setup_hub_mocks(monkeypatch)

    with pytest.raises(ValueError, match="Providing an `organization` to `save_to_hub` is deprecated"):
        splade_bert_tiny_model.save_to_hub("org1/model-name", organization="different-org")


def test_tokenize_deprecation_warning(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """tokenize() should emit a FutureWarning directing to preprocess()."""
    with pytest.warns(FutureWarning, match="tokenize.*deprecated.*preprocess"):
        result = stsb_bert_tiny_model.tokenize(["Hello world"])
    assert isinstance(result, (dict, UserDict))


def test_gradient_checkpointing_enable(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """gradient_checkpointing_enable should propagate without error."""
    stsb_bert_tiny_model.gradient_checkpointing_enable()


def test_device_property(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """device property should return a torch.device."""
    dev = stsb_bert_tiny_model.device
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


def test_get_max_seq_length(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """get_max_seq_length should return an int for transformer-based models."""
    max_seq = stsb_bert_tiny_model.get_max_seq_length()
    assert isinstance(max_seq, int)
    assert max_seq > 0
