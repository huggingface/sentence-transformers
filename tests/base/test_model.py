from __future__ import annotations

import json
import logging
import os
import tempfile
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
from sentence_transformers.base.evaluation import BaseEvaluator


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


# A 3D array is inferred as the "image" modality, a 1D array as "audio"; values are never read
# because preprocess() raises before processing, so reusing these constants across cases is safe.
_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)
_AUDIO = np.zeros((16000,), dtype=np.float32)


@pytest.mark.parametrize(
    ("modalities", "inputs", "expected", "forbidden"),
    [
        # Genuinely unsupported modality -> "Modality 'X' is not supported".
        pytest.param(
            ["text"],
            [_IMAGE],
            ["Modality 'image' is not supported", "Supported modalities: text"],
            ["mixes", "individually", "chat-style"],
            id="unsupported-image-on-text-only",
        ),
        pytest.param(
            ["text"],
            [{"text": "a cat", "image": _IMAGE}],
            ["Modality 'image+text' is not supported", "Supported modalities: text"],
            ["individually", "mixes", "chat-style"],
            id="unsupported-combined-dict-on-text-only",
        ),
        # Combinable but not combined: every part is supported, the model just cannot fuse them
        # without 'message' support, so the guidance is to encode each modality separately.
        pytest.param(
            ["text", "image"],
            [{"text": "a cat", "image": _IMAGE}],
            [
                "supports image and text individually",
                "cannot combine them in a single input",
                "Encode each modality separately",
            ],
            ["is not supported", "mixes", "does not support"],
            id="combinable-single-dict-on-text-image",
        ),
        pytest.param(
            ["text", "image"],
            ["a sentence", _IMAGE],
            ["mixes multiple modalities (image, text)", "Encode each modality separately"],
            ["does not support", "is not supported", "chat-style"],
            id="combinable-mixed-batch-on-text-image",
        ),
        pytest.param(
            ["text", "image"],
            ["a sentence", {"text": "a cat", "image": _IMAGE}],
            ["mixes multiple modalities (image+text, text)", "Encode each modality separately"],
            ["does not support", "is not supported"],
            id="combinable-mixed-batch-with-dict-on-text-image",
        ),
        # Mixed batch where a base modality is genuinely unsupported -> name the unsupported part(s).
        pytest.param(
            ["text"],
            ["a sentence", _IMAGE],
            ["mixes multiple modalities (image, text)", "does not support image", "Supported modalities: text"],
            ["Encode each modality separately", "chat-style"],
            id="mixed-batch-unsupported-image-on-text-only",
        ),
        pytest.param(
            ["text"],
            [{"text": "a cat", "image": _IMAGE}, {"text": "a dog"}],
            ["mixes multiple modalities (image+text, text)", "does not support image"],
            ["Encode each modality separately"],
            id="mixed-batch-dict-and-text-on-text-only",
        ),
        pytest.param(
            ["text"],
            ["a sentence", _AUDIO, _IMAGE],
            ["mixes multiple modalities (audio, image, text)", "does not support audio, image"],
            ["Encode each modality separately"],
            id="mixed-batch-multiple-unsupported-on-text-only",
        ),
        # Explicit chat-style message inputs on a model without 'message' support.
        pytest.param(
            ["text"],
            [{"role": "user", "content": "hi"}],
            ["does not support chat-style 'message' inputs", "Supported modalities: text"],
            ["mixes", "Modality '"],
            id="explicit-messages-on-text-only",
        ),
        pytest.param(
            ["text", "image"],
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            ["does not support chat-style 'message' inputs", "Supported modalities: text, image"],
            ["mixes", "does not support image"],
            id="explicit-messages-on-text-image",
        ),
        # Chat-style messages mixed with other inputs: the model lacks 'message' support, so the
        # chat-style inputs are the blocking issue. "message" must not be listed as a content modality.
        pytest.param(
            ["text"],
            [{"role": "user", "content": "hi"}, _IMAGE],
            ["does not support chat-style 'message' inputs", "mixes with other modalities"],
            ["multiple modalities", ", message", "does not support image"],
            id="messages-mixed-with-image-on-text-only",
        ),
        pytest.param(
            ["text"],
            [{"role": "user", "content": "hi"}, "a plain sentence"],
            ["does not support chat-style 'message' inputs", "mixes with other modalities"],
            ["multiple modalities", ", message", "does not support text"],
            id="messages-mixed-with-text-on-text-only",
        ),
    ],
)
def test_preprocess_modality_error_messages(
    stsb_bert_tiny_model: SentenceTransformer,
    monkeypatch: pytest.MonkeyPatch,
    modalities: list[str],
    inputs: list,
    expected: list[str],
    forbidden: list[str],
) -> None:
    """preprocess() should raise a clear, accurate ValueError for each unsupported-modality scenario.

    The model's supported modalities are simulated via ``modalities``; inputs use real inference
    (numpy arrays for image/audio, dicts for combined/chat inputs) so the full path is exercised.
    ``forbidden`` guards against misleading wording, e.g. claiming a modality is unsupported when
    every part is actually supported.
    """
    monkeypatch.setattr(type(stsb_bert_tiny_model), "modalities", property(lambda self: modalities))

    with pytest.raises(ValueError) as exc_info:
        stsb_bert_tiny_model.preprocess(inputs)

    message = str(exc_info.value)
    for substring in expected:
        assert substring in message, f"expected {substring!r} in error message, got: {message!r}"
    for substring in forbidden:
        assert substring not in message, f"did not expect {substring!r} in error message, got: {message!r}"


def test_preprocess_passes_supported_modality(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """preprocess() should succeed with text inputs on a text-only model."""
    features = stsb_bert_tiny_model.preprocess(["Hello world"])
    assert isinstance(features, (dict, UserDict))
    assert "input_ids" in features


def test_preprocess_skips_modality_check_for_empty_inputs(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess() should skip the modality check when inputs is empty."""

    def fail_if_called(inputs, **kwargs):
        raise AssertionError("infer_batch_modality should not be called for empty inputs")

    monkeypatch.setattr("sentence_transformers.base.model.infer_batch_modality", fail_if_called)
    features = stsb_bert_tiny_model.preprocess([])
    assert isinstance(features, dict)


def test_config_delegates_to_underlying_transformers_model(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """`model.config` should return the underlying transformers PretrainedConfig so
    integrations like Deepspeed and transformers.Trainer can read `hidden_size` etc.
    """
    assert stsb_bert_tiny_model.config is not None
    assert hasattr(stsb_bert_tiny_model.config, "hidden_size")
    assert stsb_bert_tiny_model.config.hidden_size == stsb_bert_tiny_model.transformers_model.config.hidden_size


def test_config_returns_none_when_no_underlying_transformers_model(
    static_embedding_model: SentenceTransformer,
) -> None:
    """If the model has no underlying transformers PreTrainedModel (e.g. StaticEmbedding-only),
    `model.config` returns `None` instead of raising.
    """
    assert static_embedding_model.config is None


def test_config_is_settable(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Assigning ``model.config`` should not raise and should round-trip through the
    underlying transformers model (fixes optimum ONNX export regression in v5.5.0).
    """
    original_config = stsb_bert_tiny_model.config
    assert original_config is not None
    new_config = type(original_config)(**original_config.to_dict())
    stsb_bert_tiny_model.config = new_config
    assert stsb_bert_tiny_model.config is new_config
    assert stsb_bert_tiny_model.transformers_model.config is new_config


def test_config_setter_on_model_without_underlying_transformers_model(
    static_embedding_model: SentenceTransformer,
) -> None:
    """Assigning ``model.config`` on a StaticEmbedding-only model (no underlying transformers
    model) should be silently ignored, i.e. ``model.config`` keeps returning ``None``.
    """
    assert static_embedding_model.config is None
    static_embedding_model.config = object()
    assert static_embedding_model.config is None


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
        SentenceTransformer(modules=[torch.nn.Linear(10, 10)], use_auth_token="fake-token")
        future_warnings = [
            x for x in w if issubclass(x.category, FutureWarning) and "use_auth_token" in str(x.message)
        ]
        assert len(future_warnings) == 1


def test_use_auth_token_and_token_raises() -> None:
    """Passing both token and use_auth_token should raise ValueError."""
    with pytest.raises(ValueError, match="Both `token` and `use_auth_token`"):
        SentenceTransformer(modules=[torch.nn.Linear(10, 10)], token="tok", use_auth_token="tok2")


def test_empty_modules_raises() -> None:
    """Passing an empty modules list should raise ValueError."""
    with pytest.raises(ValueError, match="An empty modules list was passed"):
        SentenceTransformer(modules=[])


def test_cache_folder_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cache_folder is None, it should fall back to SENTENCE_TRANSFORMERS_HOME env var."""
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/tmp/fake_cache")
    model = SentenceTransformer(modules=[torch.nn.Linear(10, 10)])
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


def test_save_load_roundtrip(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path, caplog) -> None:
    """A model saved and re-loaded should produce the same embeddings."""
    sentences = ["Hello world", "Testing roundtrip"]
    original_embeddings = stsb_bert_tiny_model.encode(sentences)

    stsb_bert_tiny_model.save(str(tmp_path))
    with caplog.at_level(logging.INFO, logger="sentence_transformers.base.model"):
        loaded_model = SentenceTransformer(str(tmp_path))
    loaded_embeddings = loaded_model.encode(sentences)

    np.testing.assert_allclose(original_embeddings, loaded_embeddings, atol=1e-5)
    assert f"Loading SentenceTransformer model from {tmp_path}" in caplog.text


def test_save_load_roundtrip_sparse_encoder(splade_bert_tiny_model: SparseEncoder, tmp_path: Path, caplog) -> None:
    """A SparseEncoder model saved and re-loaded should produce the same embeddings."""
    sentences = ["Hello world", "Testing roundtrip"]
    encode_kwargs = {"convert_to_sparse_tensor": False, "save_to_cpu": True}
    original_embeddings = splade_bert_tiny_model.encode(sentences, **encode_kwargs)

    splade_bert_tiny_model.save(str(tmp_path))
    with caplog.at_level(logging.INFO, logger="sentence_transformers.base.model"):
        loaded_model = SparseEncoder(str(tmp_path))
    loaded_embeddings = loaded_model.encode(sentences, **encode_kwargs)

    np.testing.assert_allclose(original_embeddings, loaded_embeddings, atol=1e-5)
    assert f"Loading SparseEncoder model from {tmp_path}" in caplog.text


def test_save_load_roundtrip_cross_encoder(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, caplog) -> None:
    """A CrossEncoder model saved and re-loaded should produce the same predictions."""
    pairs = [("Hello", "World"), ("Testing", "Roundtrip")]
    original_preds = reranker_bert_tiny_model.predict(pairs)

    reranker_bert_tiny_model.save(str(tmp_path))
    with caplog.at_level(logging.INFO, logger="sentence_transformers.base.model"):
        loaded_model = CrossEncoder(str(tmp_path))
    loaded_preds = loaded_model.predict(pairs)

    np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-5)
    assert f"Loading CrossEncoder model from {tmp_path}" in caplog.text


def test_load_logs_no_modules_json(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path, caplog) -> None:
    """Loading a plain HF model (no modules.json) should log the initializing message."""
    stsb_bert_tiny_model.save(str(tmp_path))
    (tmp_path / "modules.json").unlink()

    with caplog.at_level(logging.INFO, logger="sentence_transformers.base.model"):
        SentenceTransformer(str(tmp_path))

    assert f"No modules.json found for {tmp_path}" in caplog.text
    assert "initializing a new SentenceTransformer model" in caplog.text


def test_load_logs_converting_model_type(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path, caplog) -> None:
    """Loading a SentenceTransformer model as a SparseEncoder should log the converting message."""
    stsb_bert_tiny_model.save(str(tmp_path))

    with caplog.at_level(logging.INFO, logger="sentence_transformers.base.model"):
        SparseEncoder(str(tmp_path))

    assert "Converting SentenceTransformer" in caplog.text
    assert "to SparseEncoder" in caplog.text


class _FakeEvaluator(BaseEvaluator):
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


def test_load_module_class_from_ref_untrusted_ref_warns_about_v6(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-sentence_transformers class ref from an untrusted, non-local model still resolves for now, but
    must emit a FutureWarning that v6.0 will require `trust_remote_code=True` (the import runs
    repository-chosen code)."""
    dynamic_loading_attempted = False

    def mock_get_class(class_ref, model_name_or_path, **kwargs):
        nonlocal dynamic_loading_attempted
        dynamic_loading_attempted = True
        raise OSError("must not be reached for an untrusted, non-local ref")

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        mock_get_class,
    )

    with pytest.warns(FutureWarning, match="trust_remote_code"):
        cls = stsb_bert_tiny_model._load_module_class_from_ref(
            "torch.nn.Linear",
            model_name_or_path="nonexistent_path_12345",
            trust_remote_code=False,
            revision=None,
            model_kwargs=None,
        )
    # Untrusted + non-local must not attempt remote (dynamic) loading. The ref resolves via direct import.
    assert not dynamic_loading_attempted
    assert cls is nn.Linear


def test_load_module_class_from_ref_trust_remote_code_fallback(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With trust_remote_code=True, should try dynamic import first, then fall back on error."""

    def mock_get_class(class_ref, model_name_or_path, **kwargs):
        raise OSError("not found")

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
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
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
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


@pytest.mark.parametrize("model_class", [SentenceTransformer, CrossEncoder, SparseEncoder])
def test_model_path_pointing_to_a_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_class: type
) -> None:
    """A path that exists but is not a directory is not a local model, and must be rejected before any
    loading happens. `load_file_path` only skips its Hub fallback for real directories, so accepting a file
    would source `modules.json` (and the module class it names) from the Hub while the path still counts as
    local. A file whose name collides with a repo id would then silently load that repo (#3801).
    """
    collision = tmp_path / "bert-base-uncased"
    collision.write_text("not a model")

    def fail_hub_request(*args, **kwargs):
        raise AssertionError("no Hub request may be made for a path that points to a file")

    monkeypatch.setattr("sentence_transformers.util.file_io.hf_hub_download", fail_hub_request)
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", fail_hub_request)

    with pytest.raises(NotADirectoryError, match="is a file, not a directory"):
        model_class(str(collision))


def _setup_hub_mocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict | list]:
    """Set up common mocks for push_to_hub tests.

    Returns a dict with:
      - "upload_folder": dict capturing the most recent upload_folder kwargs
      - "create_branch_calls": list of (repo_id, branch) tuples
    """
    state = {"upload_folder": {}, "create_branch_calls": []}

    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    def mock_upload_folder(self, **kwargs):
        state["upload_folder"].update(kwargs)
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
        state["create_branch_calls"].append((repo_id, branch))
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    return state


def test_push_to_hub_sparse_encoder(splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]
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


def test_push_to_hub_create_pr_commit_description_sentence_transformer(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]
    model = stsb_bert_tiny_model
    repo_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    model.push_to_hub(repo_id, create_pr=True)
    desc = mock_kwargs["commit_description"]
    assert "automatically generated to add SentenceTransformer compatibility" in desc
    assert "push_to_hub" in desc
    assert "Full Model Architecture" in desc
    assert str(model) in desc
    assert "model.encode(" in desc
    assert "model.similarity(" in desc
    assert "from sentence_transformers import SentenceTransformer" in desc
    assert f'"{repo_id}"' in desc
    assert 'backend="torch"' in desc


def test_push_to_hub_create_pr_commit_description_cross_encoder(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]
    model = reranker_bert_tiny_model
    repo_id = "cross-encoder-testing/reranker-bert-tiny"

    model.push_to_hub(repo_id, create_pr=True)
    desc = mock_kwargs["commit_description"]
    assert "automatically generated to add CrossEncoder compatibility" in desc
    assert "Full Model Architecture" in desc
    assert "model.predict(" in desc
    assert "model.rank(" in desc
    assert "from sentence_transformers import CrossEncoder" in desc
    assert f'"{repo_id}"' in desc


def test_push_to_hub_create_pr_commit_description_sparse_encoder(
    splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]
    model = splade_bert_tiny_model
    repo_id = "sparse-encoder-testing/splade-bert-tiny-nq"

    model.push_to_hub(repo_id, create_pr=True)
    desc = mock_kwargs["commit_description"]
    assert "automatically generated to add SparseEncoder compatibility" in desc
    assert "Full Model Architecture" in desc
    assert "model.encode(" in desc
    assert "model.similarity(" in desc
    assert "from sentence_transformers import SparseEncoder" in desc
    assert f'"{repo_id}"' in desc


def test_push_to_hub_no_commit_description_without_create_pr(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]

    stsb_bert_tiny_model.push_to_hub("test-org/test-model")
    assert mock_kwargs["commit_description"] is None


def test_push_to_hub_commit_message_includes_backend(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)
    mock_kwargs = state["upload_folder"]

    stsb_bert_tiny_model.push_to_hub("test-org/test-model")
    assert mock_kwargs["commit_message"] == "Add new SentenceTransformer model"


def test_push_to_hub_create_branch_on_revision(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup_hub_mocks(monkeypatch)

    stsb_bert_tiny_model.push_to_hub("test-org/test-model")
    assert state["create_branch_calls"] == []

    stsb_bert_tiny_model.push_to_hub("test-org/test-model", revision="my-branch")
    assert ("test-org/test-model", "my-branch") in state["create_branch_calls"]


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


def test_tokenize_deprecation_warning(stsb_bert_tiny_model: SentenceTransformer, caplog) -> None:
    """tokenize() should emit a warning directing to preprocess()."""
    with caplog.at_level(logging.WARNING, logger="sentence_transformers.base.model"):
        result = stsb_bert_tiny_model.tokenize(["Hello world"])
    assert "tokenize" in caplog.text and "deprecated" in caplog.text and "preprocess" in caplog.text
    assert isinstance(result, (dict, UserDict))


def test_gradient_checkpointing_enable(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """gradient_checkpointing_enable should propagate without error."""
    stsb_bert_tiny_model.gradient_checkpointing_enable()


def test_device_property(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """device property should return a torch.device."""
    dev = stsb_bert_tiny_model.device
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


@pytest.mark.parametrize(
    "model_class, model_id",
    [
        (SentenceTransformer, "sentence-transformers-testing/stsb-bert-tiny-safetensors"),
        (CrossEncoder, "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"),
        (SparseEncoder, "sparse-encoder-testing/inference-free-splade-bert-tiny-nq"),
    ],
)
def test_device_map_controls_placement(model_class: type, model_id: str) -> None:
    """A ``device_map`` in ``model_kwargs`` must control placement and not be overridden by the
    auto-detected ``device``. Previously the unconditional ``self.to(device)`` would e.g. pull a
    ``device_map="cuda:1"`` model back onto ``cuda:0``. The fix lives in the shared base, so this is
    checked for every model family. See #3822."""
    # device_map pins the backbone to CPU. Even when a GPU is present (so the auto-detected device
    # would be cuda), the device_map must win and nothing should be moved to cuda.
    model = model_class(model_id, model_kwargs={"device_map": {"": "cpu"}})
    assert model.transformers_model.device.type == "cpu"
    assert all(param.device.type == "cpu" for param in model.parameters())


def test_device_argument_warns_when_device_map_present(caplog: pytest.LogCaptureFixture) -> None:
    """Passing both ``device`` and a ``device_map`` should warn that ``device_map`` wins. See #3822."""
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    with caplog.at_level(logging.WARNING, logger="sentence_transformers.base.model"):
        # device="cuda" is inert here (device_map controls placement), so this is safe without a GPU.
        model = SentenceTransformer(model_id, device="cuda", model_kwargs={"device_map": {"": "cpu"}})
    assert model.device.type == "cpu"
    assert "device_map" in caplog.text and "ignored" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_map_aligns_auxiliary_modules_to_backbone() -> None:
    """With a ``device_map`` placing the backbone on a GPU, separately-loaded modules (e.g. Dense,
    which load on CPU) must be moved onto the backbone's device, else the forward pass mismatches."""
    from sentence_transformers.sentence_transformer.modules import Dense, Pooling, Transformer

    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    transformer = Transformer(model_id)
    dim = transformer.get_embedding_dimension()
    model = SentenceTransformer(modules=[transformer, Pooling(dim), Dense(dim, 64)], device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        reloaded = SentenceTransformer(tmp_dir, model_kwargs={"device_map": {"": 0}})

    for param in reloaded.parameters():
        assert param.device.type == "cuda"
    # A cross-device mismatch would raise here.
    assert reloaded.encode("hello world").shape == (64,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_map_ignored_for_prebuilt_modules() -> None:
    """``device_map`` only applies when loading from a name (via ``from_pretrained``). With pre-built
    ``modules`` it is inert, so it must not suppress the usual ``self.to(device)`` and strand the model."""
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    transformer = Transformer(model_id)  # loads on CPU
    pooling = Pooling(transformer.get_embedding_dimension())
    # device defaults to the auto-detected GPU. The stray device_map must not prevent the move there.
    model = SentenceTransformer(modules=[transformer, pooling], model_kwargs={"device_map": {"": "cpu"}})
    assert model.device.type == "cuda"


def test_get_max_seq_length(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """get_max_seq_length should return an int for transformer-based models."""
    max_seq = stsb_bert_tiny_model.get_max_seq_length()
    assert isinstance(max_seq, int)
    assert max_seq > 0


def test_can_flatten_inputs_transformer_model(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_can_flatten_inputs should delegate to the Transformer module's can_flatten_inputs attribute."""
    from sentence_transformers.base.modules import Transformer

    assert isinstance(stsb_bert_tiny_model[0], Transformer)
    # Force can_flatten_inputs on the Transformer to True and check propagation
    monkeypatch.setattr(stsb_bert_tiny_model[0], "can_flatten_inputs", True)
    assert stsb_bert_tiny_model._can_flatten_inputs() is True
    # Force it to False and check
    monkeypatch.setattr(stsb_bert_tiny_model[0], "can_flatten_inputs", False)
    assert stsb_bert_tiny_model._can_flatten_inputs() is False


def test_can_flatten_inputs_static_model(static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """A StaticEmbedding model (non-Transformer first module) should not support flattened inputs."""
    from sentence_transformers.base.modules import Transformer

    assert not isinstance(static_retrieval_mrl_en_v1_model[0], Transformer)
    assert static_retrieval_mrl_en_v1_model._can_flatten_inputs() is False


def test_dtype_property(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """dtype should return a torch.dtype for a model with parameters."""
    dtype = stsb_bert_tiny_model.dtype
    assert isinstance(dtype, torch.dtype)


def test_dtype_no_parameters() -> None:
    """dtype should return None for a model with no parameters."""
    model = SentenceTransformer(modules=[torch.nn.Module()])
    assert model.dtype is None


def test_is_singular_input_string(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A string input should be singular."""
    assert stsb_bert_tiny_model.is_singular_input("hello") is True


def test_is_singular_input_list(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A list input should be a batch (not singular)."""
    assert stsb_bert_tiny_model.is_singular_input(["hello", "world"]) is False


def test_is_singular_input_tuple(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A tuple input should be a batch (not singular)."""
    assert stsb_bert_tiny_model.is_singular_input(("hello", "world")) is False


def test_is_singular_input_numpy(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A numeric numpy array should be singular (treated as an audio waveform)."""
    assert stsb_bert_tiny_model.is_singular_input(np.array([1, 2, 3])) is True


def test_is_singular_input_tensor(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A torch tensor should be singular (not a list type)."""
    assert stsb_bert_tiny_model.is_singular_input(torch.tensor([1, 2, 3])) is True


def test_is_singular_input_numpy_1d_strings(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A 1D numpy string array is a batch of texts (e.g. from np.unique), not a single audio input."""
    assert stsb_bert_tiny_model.is_singular_input(np.array(["hello", "world"])) is False
    assert stsb_bert_tiny_model.is_singular_input(np.unique(["a", "b", "a"])) is False


def test_is_singular_input_numpy_2d_strings(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A 2D numpy string array is a batch of text pairs."""
    assert stsb_bert_tiny_model.is_singular_input(np.array([["q1", "d1"], ["q2", "d2"]])) is False


def test_is_singular_input_numpy_bytes(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A numpy byte-string array is not treated as a text batch (downstream modality inference
    does not handle Python ``bytes``), so it falls through to the default singular interpretation."""
    assert stsb_bert_tiny_model.is_singular_input(np.array([b"hello", b"world"])) is True


def test_is_singular_input_numpy_object(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A numpy object array is a batch (no valid single-sample type is an object ndarray)."""
    assert stsb_bert_tiny_model.is_singular_input(np.array(["hello", "world"], dtype=object)) is False


def test_is_singular_input_numpy_0d_string(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A 0-dim numpy string array represents a single text."""
    assert stsb_bert_tiny_model.is_singular_input(np.array("hello")) is True


def test_encode_numpy_1d_string_array(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Regression test for #3718: encoding a 1D numpy string array should not raise and
    should produce one embedding per element."""
    texts = np.array(["Access Management", "Press Coordination", "Financial Reports"])
    embeddings = stsb_bert_tiny_model.encode(texts, show_progress_bar=False)
    expected = stsb_bert_tiny_model.encode(texts.tolist(), show_progress_bar=False)
    assert embeddings.shape[0] == 3
    assert np.allclose(embeddings, expected)


def test_encode_numpy_2d_string_array(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Encoding a 2D numpy string array should match encoding the equivalent nested list."""
    pairs = np.array([["what is AI?", "AI is artificial intelligence."], ["what is ML?", "ML is machine learning."]])
    embeddings = stsb_bert_tiny_model.encode(pairs, show_progress_bar=False)
    expected = stsb_bert_tiny_model.encode(pairs.tolist(), show_progress_bar=False)
    assert embeddings.shape[0] == 2
    assert np.allclose(embeddings, expected)


def test_encode_numpy_empty(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Encoding an empty string ndarray should return an empty result, like ``encode([])``."""
    embeddings = stsb_bert_tiny_model.encode(np.array([], dtype=str), show_progress_bar=False)
    expected = stsb_bert_tiny_model.encode([], show_progress_bar=False)
    assert np.array_equal(embeddings, expected)


@pytest.mark.parametrize(
    "initial_prompts, config_prompts, expected_prompts",
    [
        ({}, {"query": "Search: ", "passage": "Passage: "}, {"query": "Search: ", "passage": "Passage: "}),
        (
            {"query": "My custom: "},
            {"query": "Search: ", "passage": "Passage: "},
            {"query": "My custom: ", "passage": "Passage: "},
        ),
        ({"query": ""}, {"query": "Search: "}, {"query": ""}),
        ({"query": None}, {"query": "Search: "}, {"query": "Search: "}),
    ],
    ids=[
        "saved_prompts_fill_empty_slots",
        "user_prompts_not_overwritten",
        "empty_string_is_intentional",
        "none_placeholder_is_overwritten",
    ],
)
def test_parse_model_config_prompts(
    stsb_bert_tiny_model: SentenceTransformer, initial_prompts, config_prompts, expected_prompts
) -> None:
    stsb_bert_tiny_model.prompts = initial_prompts
    stsb_bert_tiny_model.default_prompt_name = None
    stsb_bert_tiny_model._parse_model_config({"prompts": config_prompts})
    assert stsb_bert_tiny_model.prompts == expected_prompts


@pytest.mark.parametrize(
    "initial_default, config_default, expected_default",
    [
        (None, "query", "query"),
        ("my_prompt", "query", "my_prompt"),
    ],
    ids=["from_config_when_none", "not_overwritten_when_set"],
)
def test_parse_model_config_default_prompt_name(
    stsb_bert_tiny_model: SentenceTransformer, initial_default, config_default, expected_default
) -> None:
    stsb_bert_tiny_model.prompts = {}
    stsb_bert_tiny_model.default_prompt_name = initial_default
    stsb_bert_tiny_model._parse_model_config({"default_prompt_name": config_default})
    assert stsb_bert_tiny_model.default_prompt_name == expected_default


def test_get_model_type_no_config_defaults_to_sentence_transformer(
    stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path
) -> None:
    """When config_sentence_transformers.json is missing, _get_model_type defaults to 'SentenceTransformer'."""
    # Empty directory, no config file
    result = stsb_bert_tiny_model._get_model_type(str(tmp_path), token=None, cache_folder=None, local_files_only=True)
    assert result == "SentenceTransformer"


def test_get_model_type_missing_key_defaults_to_sentence_transformer(
    stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path
) -> None:
    """A config without 'model_type' key should default to 'SentenceTransformer' (backward compat)."""
    config_path = tmp_path / "config_sentence_transformers.json"
    config_path.write_text(json.dumps({"__version__": "1.0.0"}), encoding="utf8")

    result = stsb_bert_tiny_model._get_model_type(str(tmp_path), token=None, cache_folder=None, local_files_only=True)
    assert result == "SentenceTransformer"


def test_get_model_type_reads_model_type(stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path) -> None:
    """_get_model_type should return the model_type from config when present."""
    config_path = tmp_path / "config_sentence_transformers.json"
    config_path.write_text(json.dumps({"__version__": "1.0.0", "model_type": "SparseEncoder"}), encoding="utf8")

    result = stsb_bert_tiny_model._get_model_type(str(tmp_path), token=None, cache_folder=None, local_files_only=True)
    assert result == "SparseEncoder"
