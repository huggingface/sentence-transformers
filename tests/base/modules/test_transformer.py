from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from packaging.version import Version
from packaging.version import parse as parse_version
from tokenizers.normalizers import NFC, Lowercase, Sequence
from transformers import AutoModel, AutoProcessor
from transformers import __version__ as transformers_version
from transformers.utils import is_torchvision_available, is_vision_available

from sentence_transformers.base.modules import Transformer
from sentence_transformers.base.modules.transformer import (
    TRANSFORMER_TASK_DEFAULTS,
    _count_media_per_sample,
    set_temporary_class_attrs,
)
from sentence_transformers.util import batch_to_device

transformer_module = sys.modules[Transformer.__module__]

TINY_BERT = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_LLAVA = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"

# Uneven lengths, so that padding is actually applied and the padded side is observable.
RAGGED_BATCH = ["hi", "a considerably longer sentence than the other one"]


@pytest.fixture()
def bert_tiny_transformer(stsb_bert_tiny_model) -> Transformer:
    """A lightweight BERT Transformer for reuse across tests."""
    return stsb_bert_tiny_model[0]


def test_preprocess_unsupported_modality_uses_shared_error(bert_tiny_transformer):
    """A text-only Transformer module rejects unsupported modalities with the same module-scoped
    guidance that BaseModel.preprocess produces (both via ``raise_unsupported_modality_error``)."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=r"Modality 'image' is not supported by this Transformer module\. Supported modalities: text",
    ):
        bert_tiny_transformer.preprocess([image])

    with pytest.raises(
        ValueError,
        match=r"mixes multiple modalities \(image, text\), but this Transformer module does not support image",
    ):
        bert_tiny_transformer.preprocess(["a sentence", image])


class TestSetTemporaryClassAttrs:
    def test_sets_and_restores(self):
        class Dummy:
            x = 1
            y = 2

        with set_temporary_class_attrs(Dummy, x=10, y=20):
            assert Dummy.x == 10
            assert Dummy.y == 20
        assert Dummy.x == 1
        assert Dummy.y == 2

    def test_restores_on_exception(self):
        class Dummy:
            x = 1

        with pytest.raises(RuntimeError):
            with set_temporary_class_attrs(Dummy, x=99):
                raise RuntimeError("boom")
        assert Dummy.x == 1


class TestTransformerInit:
    def test_invalid_transformer_task(self):
        with pytest.raises(ValueError, match="Unsupported transformer_task"):
            Transformer(TINY_BERT, transformer_task="nonexistent-task")

    def test_sequence_classification_default_num_labels(self):
        """When loading a non-SeqCls model with task='sequence-classification', num_labels defaults to 1."""
        transformer = Transformer(TINY_BERT, transformer_task="sequence-classification")
        assert transformer.config.num_labels == 1

    def test_do_lower_case(self):
        """do_lower_case should add exactly one Lowercase normalizer and actually lowercase tokens."""
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        assert transformer.do_lower_case is True

        # Normalizer should be a Sequence containing exactly one Lowercase
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        assert sum(1 for n in normalizer if isinstance(n, Lowercase)) == 1

        # Tokens should be lowercased, and different cases should produce the same IDs
        tokens = transformer.tokenizer.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens)
        features_lower = transformer.preprocess(["hello world"])
        features_upper = transformer.preprocess(["HELLO WORLD"])
        assert torch.equal(features_lower["input_ids"], features_upper["input_ids"])

    def test_do_lower_case_noop_when_lowercase_already_present(self, monkeypatch):
        """If the normalizer already has Lowercase, do_lower_case should not add another."""
        # Pre-set the normalizer to already contain Lowercase before __init__ applies it
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = Sequence([Lowercase()])
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        assert sum(1 for n in normalizer if isinstance(n, Lowercase)) == 1

    def test_do_lower_case_prepends_to_existing_sequence(self, monkeypatch):
        """When normalizer is a Sequence without Lowercase, Lowercase should be prepended."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = Sequence([NFC()])
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert isinstance(normalizer_list[0], Lowercase)
        assert any(isinstance(n, NFC) for n in normalizer_list)

    def test_do_lower_case_wraps_single_normalizer(self, monkeypatch):
        """When normalizer is a single non-Sequence normalizer, it should be wrapped with Lowercase."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = NFC()
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert isinstance(normalizer_list[0], Lowercase)
        assert isinstance(normalizer_list[1], NFC)

    def test_do_lower_case_with_none_normalizer(self, monkeypatch):
        """When normalizer is None, do_lower_case should create a Sequence with just Lowercase."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = None
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert len(normalizer_list) == 1
        assert isinstance(normalizer_list[0], Lowercase)

    def test_do_lower_case_false_does_not_modify_normalizer(self):
        """do_lower_case=False should not modify the tokenizer normalizer."""
        transformer_default = Transformer(TINY_BERT, do_lower_case=False)
        transformer_none = Transformer(TINY_BERT)
        norm_default = transformer_default.tokenizer.backend_tokenizer.normalizer
        norm_none = transformer_none.tokenizer.backend_tokenizer.normalizer
        if norm_default is None:
            assert norm_none is None
        else:
            assert str(norm_default) == str(norm_none)

    @pytest.mark.skipif(
        parse_version(transformers_version) >= parse_version("5.0.0"),
        reason="Transformers v5 only has fast tokenizers",
    )
    def test_do_lower_case_slow_tokenizer_fallback(self):
        """For slow tokenizers, do_lower_case should set processor.do_lower_case."""
        transformer = Transformer(TINY_BERT, do_lower_case=True, processor_kwargs={"use_fast": False})
        assert transformer.tokenizer.is_fast is False
        assert transformer.processor.do_lower_case is True

    def test_do_lower_case_tokenizer_persisted_after_save_load(self, tmp_path):
        """The Lowercase normalizer added to the tokenizer should persist after save/load."""
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        tokens_before = transformer.tokenizer.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens_before)

        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)
        reloaded = Transformer.load(save_dir)

        # The tokenizer normalizer is saved with the tokenizer, so lowercasing still works
        tokens_after = reloaded.tokenizer.tokenize("Hello WORLD")
        assert tokens_before == tokens_after
        assert all(t == t.lower() for t in tokens_after)

    def test_processing_kwargs_default_empty(self):
        """processing_kwargs should default to an empty dict when not provided."""
        transformer = Transformer(TINY_BERT)
        assert transformer.processing_kwargs == {}

    def test_processing_kwargs_stored(self):
        """processing_kwargs passed to __init__ should be stored on the instance."""
        kwargs = {"text": {"truncation": "only_first"}, "chat_template": {"add_generation_prompt": True}}
        transformer = Transformer(TINY_BERT, processing_kwargs=kwargs)
        assert transformer.processing_kwargs == kwargs

    def test_processing_kwargs_unknown_keys_warning(self, caplog):
        """Unknown top-level keys in processing_kwargs should emit a warning."""
        with caplog.at_level(logging.WARNING):
            Transformer(TINY_BERT, processing_kwargs={"padding_side": {"value": "left"}, "text": {"truncation": True}})
        assert any("Unknown keys" in record.message and "padding_side" in record.message for record in caplog.records)

    def test_processing_kwargs_valid_keys_no_warning(self, caplog):
        """Valid processing_kwargs keys should not emit a warning."""
        with caplog.at_level(logging.WARNING):
            Transformer(
                TINY_BERT, processing_kwargs={"common": {"return_tensors": "pt"}, "text": {"truncation": True}}
            )
        assert not any("Unknown keys" in record.message for record in caplog.records)

    def test_tokenizer_name_or_path_warning(self, caplog):
        """tokenizer_name_or_path should emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, tokenizer_name_or_path=TINY_BERT)
        assert any(
            "tokenizer_name_or_path" in record.message and "deprecated" in record.message for record in caplog.records
        )
        assert transformer is not None


class TestWarnOnUnsupportedAttentionConfig:
    """Warn when the config requests attention settings not supported by the installed transformers version."""

    _BIDIR = ("use_bidirectional_attention", True, "_TRANSFORMERS_SUPPORTS_USE_BIDIRECTIONAL_ATTENTION", "4.56.2")
    _CAUSAL = ("is_causal", False, "_TRANSFORMERS_SUPPORTS_IS_CAUSAL_FALSE", "5.2.0")

    @pytest.mark.parametrize(
        ("param", "value", "support_flag", "min_version", "set_on", "supports", "expect_warn"),
        [
            # Warns when flag is set on the main config and version is unsupported
            (*_BIDIR, "main", False, True),
            (*_CAUSAL, "main", False, True),
            # Warns when flag is set on a sub-config (e.g. multimodal text_config)
            (*_BIDIR, "sub", False, True),
            (*_CAUSAL, "sub", False, True),
            # No warning when version supports the flag
            (*_BIDIR, "main", True, False),
            (*_CAUSAL, "main", True, False),
            # No warning when flag is absent, even if version is unsupported
            (*_BIDIR, None, False, False),
            (*_CAUSAL, None, False, False),
        ],
    )
    def test_warn_on_unsupported_attention_config(
        self,
        caplog,
        monkeypatch,
        bert_tiny_transformer,
        param,
        value,
        support_flag,
        min_version,
        set_on,
        supports,
        expect_warn,
    ):
        monkeypatch.setattr(transformer_module, support_flag, supports)
        config = bert_tiny_transformer.config
        if set_on == "main":
            setattr(config, param, value)
        elif set_on == "sub":
            sub = deepcopy(config)
            setattr(sub, param, value)
            config.sub_configs = {"text_config": type(sub)}
            config.text_config = sub

        with caplog.at_level(logging.WARNING):
            Transformer._warn_on_unsupported_attention_config(config)

        matching = [
            r
            for r in caplog.records
            if f"{param}={value}" in r.message and f"transformers>={min_version}" in r.message
        ]
        assert bool(matching) is expect_warn


class TestTransformerMaxSeqLength:
    """Test the max_seq_length property for both tokenizer-based and config-based models."""

    def test_max_seq_length_from_tokenizer(self, bert_tiny_transformer):
        model = bert_tiny_transformer
        assert model.max_seq_length is not None
        assert model.max_seq_length == model.tokenizer.model_max_length

    def test_max_seq_length_setter(self, bert_tiny_transformer):
        model = bert_tiny_transformer
        original = model.max_seq_length
        model.max_seq_length = 64
        assert model.max_seq_length == 64
        assert model.tokenizer.model_max_length == 64
        model.max_seq_length = original

    def test_max_seq_length_init_kwarg(self):
        transformer = Transformer(TINY_BERT, max_seq_length=42)
        assert transformer.max_seq_length == 42

    def test_max_seq_length_capped_by_max_position_embeddings(self, bert_tiny_transformer):
        model = bert_tiny_transformer
        if hasattr(model.config, "max_position_embeddings"):
            assert model.max_seq_length <= model.config.max_position_embeddings

    def test_max_seq_length_fallback_to_config(self, bert_tiny_transformer, monkeypatch):
        """When tokenizer is None, max_seq_length should fall back to config.max_position_embeddings."""
        model = bert_tiny_transformer
        monkeypatch.setattr(type(model), "tokenizer", property(lambda self: None))
        seq_len = model.max_seq_length
        if hasattr(model.config, "max_position_embeddings"):
            assert seq_len == model.config.max_position_embeddings

    def test_max_seq_length_setter_noop_without_tokenizer(self, bert_tiny_transformer, monkeypatch):
        """Setting max_seq_length when tokenizer is None should be a no-op."""
        model = bert_tiny_transformer
        monkeypatch.setattr(type(model), "tokenizer", property(lambda self: None))
        model.max_seq_length = 42  # should not raise

    def test_max_seq_length_truncation(self):
        """Inputs longer than max_seq_length should be truncated during preprocessing."""
        transformer = Transformer(TINY_BERT, max_seq_length=5)
        features = transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert features["input_ids"].shape[1] == 5
        assert transformer.tokenizer.model_max_length == 5


class TestTransformerDeprecatedKwargs:
    def test_model_args_deprecated(self, caplog):
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, model_args={})
        assert any("model_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_tokenizer_args_deprecated(self, caplog):
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, tokenizer_args={})
        assert any("tokenizer_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_config_args_deprecated(self, caplog):
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, config_args={})
        assert any("config_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_new_kwargs_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, model_kwargs={}, processor_kwargs={}, config_kwargs={})
        assert not any("deprecated" in r.message for r in caplog.records)
        assert transformer is not None


class TestTransformerModalityConfigValidation:
    def test_valid_modality_config(self):
        transformer = Transformer(
            TINY_BERT,
            modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
            module_output_name="token_embeddings",
        )
        assert transformer.modality_config == {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"}
        }

    def test_invalid_modality_config_missing_method(self):
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                TINY_BERT,
                modality_config={"text": {"method_output_name": "last_hidden_state"}},
                module_output_name="token_embeddings",
            )

    def test_invalid_modality_config_missing_output_name(self):
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                TINY_BERT, modality_config={"text": {"method": "forward"}}, module_output_name="token_embeddings"
            )

    def test_modality_config_requires_module_output_name(self):
        with pytest.raises(ValueError, match="module_output_name"):
            Transformer(
                TINY_BERT, modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
            )


class TestPreprocess:
    def test_unsupported_modality_error(self, bert_tiny_transformer, monkeypatch):
        """Passing an unsupported modality should raise ValueError."""
        model = bert_tiny_transformer
        monkeypatch.setattr(model.input_formatter, "parse_inputs", lambda *a, **kw: ("video", {"video": []}, {}))
        with pytest.raises(ValueError, match="not supported"):
            model.preprocess(["test"])

    def test_preprocess_with_text_prompt(self, bert_tiny_transformer):
        """preprocess should incorporate a text prompt and set prompt_length."""
        result = bert_tiny_transformer.preprocess(["hello world"], prompt="search query: ")
        assert "prompt_length" in result
        assert isinstance(result["prompt_length"], int)
        assert result["prompt_length"] > 0

    def test_preprocess_without_prompt(self, bert_tiny_transformer):
        """preprocess without a prompt should not include prompt_length."""
        result = bert_tiny_transformer.preprocess(["hello world"])
        assert "prompt_length" not in result

    def test_preprocess_empty_inputs(self, bert_tiny_transformer):
        """preprocess with an empty list should return an empty dict without raising."""
        result = bert_tiny_transformer.preprocess([])
        assert result == {}

    def test_preprocess_processing_kwargs_text_override(self):
        """processing_kwargs should override default text preprocessing kwargs."""
        # With max_length=5 and truncation, the output should be truncated
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        result = transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert result["input_ids"].shape[1] == 5

    def test_preprocess_processing_kwargs_common_override(self):
        """processing_kwargs 'common' should override common_kwargs like return_tensors."""
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"common": {"return_tensors": "np"}},
        )
        result = transformer.preprocess(["hello world"])
        assert isinstance(result["input_ids"], np.ndarray)

    def test_preprocess_per_call_processing_kwargs_text_override(self, bert_tiny_transformer):
        """Per-call processing_kwargs should override defaults for that call only."""
        result = bert_tiny_transformer.preprocess(
            ["this is a longer sentence that should get truncated"],
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        assert result["input_ids"].shape[1] == 5

        # Without per-call overrides, the default behaviour is restored
        next_result = bert_tiny_transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert next_result["input_ids"].shape[1] > 5

    def test_preprocess_per_call_processing_kwargs_common_override(self, bert_tiny_transformer):
        """Per-call processing_kwargs 'common' should override return_tensors for that call only."""
        result = bert_tiny_transformer.preprocess(
            ["hello world"],
            processing_kwargs={"common": {"return_tensors": "np"}},
        )
        assert isinstance(result["input_ids"], np.ndarray)

        next_result = bert_tiny_transformer.preprocess(["hello world"])
        assert isinstance(next_result["input_ids"], torch.Tensor)

    def test_preprocess_per_call_merges_with_instance_processing_kwargs(self):
        """Per-call values override matching instance values; non-overridden settings are preserved."""
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        # Per-call only changes max_length; truncation from the instance must still apply
        result = transformer.preprocess(
            ["this is a longer sentence that should get truncated"],
            processing_kwargs={"text": {"max_length": 8}},
        )
        assert result["input_ids"].shape[1] == 8

    def test_preprocess_per_call_processing_kwargs_unknown_keys_warning(self, bert_tiny_transformer, caplog):
        """Unknown top-level keys in per-call processing_kwargs should emit a warning."""
        with caplog.at_level(logging.WARNING):
            bert_tiny_transformer.preprocess(
                ["hello world"],
                processing_kwargs={"texts": {"max_length": 5}},
            )
        assert "Unknown keys in per-call `processing_kwargs`" in caplog.text

    def test_preprocess_per_call_processing_kwargs_does_not_mutate_instance(self, bert_tiny_transformer):
        """Per-call overrides must not leak into self.processing_kwargs."""
        before = deepcopy(bert_tiny_transformer.processing_kwargs)
        bert_tiny_transformer.preprocess(
            ["hello world"],
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        assert bert_tiny_transformer.processing_kwargs == before

    def test_causal_task_left_pads_by_default(self):
        """The module sets padding_side to "left" on load, so a ragged batch ends in real tokens."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        features = transformer.preprocess(RAGGED_BATCH)
        assert transformer.processor.padding_side == "left"
        assert features["attention_mask"][:, -1].all()

    def test_causal_right_padding_via_attribute_raises(self):
        """padding_side is set to "left" on load, so reaching this means it was mutated afterwards."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        transformer.processor.padding_side = "right"
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH)
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert 'processor.padding_side = "left"' in message

    def test_causal_right_padding_via_processing_kwargs_raises(self):
        """processing_kwargs overrides the attribute at call time, so the attribute alone can't be trusted:
        it still reads "left" here while the processor pads right."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH, processing_kwargs={"text": {"padding_side": "right"}})
        assert transformer.processor.padding_side == "left"
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert "processing_kwargs" in message

    @pytest.mark.parametrize("bucket", ["text", "common"])
    def test_causal_right_padding_via_processing_kwargs_raises_without_message_modality(self, bucket):
        """TINY_LLAMA has a chat template, so text normally routes through apply_chat_template, which only
        forwards the "text" bucket. Without a "message" modality the tokenizer is called directly and the
        "common" bucket reaches it as well, so both buckets must be caught."""
        transformer = Transformer(
            TINY_LLAMA,
            transformer_task="text-generation",
            modality_config={"text": {"method": "forward", "method_output_name": "logits"}},
            module_output_name="causal_logits",
        )
        assert "message" not in transformer.modality_config
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH, processing_kwargs={bucket: {"padding_side": "right"}})
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert "processing_kwargs" in message

    def test_causal_padding_check_ignores_unpadded_batches(self):
        """A batch with nothing to pad has no last-token problem, whatever padding_side says."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        transformer.processor.padding_side = "right"
        transformer.preprocess(["same length here"])

    @pytest.mark.skipif(
        Version(transformers_version) < Version("5.0.0"), reason="any-to-any requires transformers v5+"
    )
    @pytest.mark.skipif(
        not (is_vision_available() and is_torchvision_available()),
        reason="Loading the Llava processor requires Pillow and torchvision",
    )
    def test_causal_right_padding_via_multimodal_tokenizer_raises(self):
        """A multimodal processor pads via its tokenizer and has no padding_side of its own, so the
        tokenizer attribute is the only one that can be wrong, and the only one worth reporting."""
        transformer = Transformer(TINY_LLAVA, transformer_task="any-to-any")
        assert not hasattr(transformer.processor, "padding_side")
        assert transformer.processor.tokenizer.padding_side == "left"

        transformer.processor.tokenizer.padding_side = "right"
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH)
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert 'processor.tokenizer.padding_side = "left"' in message


class TestForward:
    def test_missing_method_error(self, bert_tiny_transformer):
        """forward should raise ValueError when the model doesn't have the requested method."""
        model = bert_tiny_transformer
        model.modality_config["text"]["method"] = "nonexistent_method"
        features = model.preprocess(["test"])
        with pytest.raises(ValueError, match="does not have the requested"):
            model.forward(features)

    def test_4d_embedding_reshape(self, bert_tiny_transformer, monkeypatch):
        """4D model outputs should be flattened to 3D."""
        model = bert_tiny_transformer
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        original_forward = model.model.forward

        def mock_forward(**kwargs):
            out = original_forward(**kwargs)
            b = out.last_hidden_state.shape[0]
            fake_4d = torch.randn(b, 3, 4, 4)

            class Fake4DOutput:
                def __getitem__(self, key):
                    return fake_4d if key == "last_hidden_state" else None

                def __contains__(self, key):
                    return key == "last_hidden_state"

            return Fake4DOutput()

        monkeypatch.setattr(model.model, "forward", mock_forward)
        with torch.no_grad():
            result = model.forward(features)
        emb = result[model.module_output_name]
        assert emb.ndim == 3

    def test_attribute_fallback_for_output(self, bert_tiny_transformer, monkeypatch):
        """forward should fall back to getattr when dict indexing fails on model output."""
        model = bert_tiny_transformer

        class FakeOutput:
            def __init__(self, tensor):
                self.last_hidden_state = tensor

            def __getitem__(self, key):
                raise KeyError(key)

        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        original_forward = model.model.forward

        def mock_forward(**kwargs):
            out = original_forward(**kwargs)
            return FakeOutput(out.last_hidden_state)

        monkeypatch.setattr(model.model, "forward", mock_forward)
        with torch.no_grad():
            result = model.forward(features)
        assert model.module_output_name in result

    def test_output_hidden_states(self, bert_tiny_transformer):
        """When output_hidden_states is True, all_layer_embeddings should appear in features."""
        model = bert_tiny_transformer
        model.model.config.output_hidden_states = True
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        with torch.no_grad():
            result = model.forward(features)
        assert "all_layer_embeddings" in result

    def test_list_method_output_name_hidden_states(self, bert_tiny_transformer):
        """A list method_output_name like ["hidden_states", -1] should auto-enable output_hidden_states
        and traverse the output path to return the correct hidden layer embedding."""
        model = bert_tiny_transformer
        model.model.config.output_hidden_states = False
        model.modality_config["text"]["method_output_name"] = ["hidden_states", -1]
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        with torch.no_grad():
            result = model.forward(features)
        embedding = result[model.module_output_name]
        assert embedding.ndim == 3
        assert embedding.shape[-1] == model.config.hidden_size


class TestGetEmbeddingDimension:
    def test_standard_hidden_size(self, bert_tiny_transformer):
        dim = bert_tiny_transformer.get_embedding_dimension()
        assert dim == bert_tiny_transformer.config.hidden_size

    def test_hidden_sizes_list(self, bert_tiny_transformer, monkeypatch):
        """Models with hidden_sizes (list) should return the last element."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        model.config.hidden_sizes = [64, 128, 256]
        assert model.get_embedding_dimension() == 256

    def test_hidden_dim(self, bert_tiny_transformer, monkeypatch):
        """Models with hidden_dim should return it."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        model.config.hidden_dim = 384
        assert model.get_embedding_dimension() == 384

    def test_raises_when_no_dimension_found(self, bert_tiny_transformer, monkeypatch):
        """Should raise ValueError when no dimension attribute is found."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        with pytest.raises(ValueError, match="Could not determine embedding dimension"):
            model.get_embedding_dimension()

    def test_projection_dim_for_sentence_embedding(self, bert_tiny_transformer):
        """When module_output_name is 'sentence_embedding', projection_dim takes priority."""
        model = bert_tiny_transformer
        model.module_output_name = "sentence_embedding"
        model.config.projection_dim = 512
        assert model.get_embedding_dimension() == 512

    def test_text_config_fallback(self, bert_tiny_transformer):
        """Should fall back to text_config.hidden_size when main config lacks it."""
        model = bert_tiny_transformer
        del model.config.hidden_size

        class FakeTextConfig:
            hidden_size = 768

        model.config.text_config = FakeTextConfig()
        assert model.get_embedding_dimension() == 768


class TestGetPromptLength:
    def test_prompt_length_cached(self, bert_tiny_transformer):
        """Second call with the same prompt should use the cache."""
        model = bert_tiny_transformer
        model._prompt_length_mapping.clear()
        len1 = model._get_prompt_length("search query: ")
        len2 = model._get_prompt_length("search query: ")
        assert len1 == len2
        assert len(model._prompt_length_mapping) == 1

    def test_prompt_length_cache_distinguishes_kwarg_keys(self, bert_tiny_transformer):
        """Different kwarg keys with the same values should produce separate cache entries."""
        model = bert_tiny_transformer
        model._prompt_length_mapping.clear()
        model._get_prompt_length("search query: ", task="foo")
        model._get_prompt_length("search query: ", mode="foo")
        assert len(model._prompt_length_mapping) == 2

    def test_prompt_length_excludes_special_tokens(self, bert_tiny_transformer):
        """Prompt length should exclude trailing special tokens like [SEP]."""
        model = bert_tiny_transformer
        prompt = "query: "
        tokenized = model.tokenizer(prompt, add_special_tokens=True)
        raw_length = len(tokenized["input_ids"])
        prompt_length = model._get_prompt_length(prompt)
        assert prompt_length < raw_length


class TestCallProcessor:
    def test_single_modality_tokenizer(self, bert_tiny_transformer):
        """Single-modality tokenizer path should work for text."""
        model = bert_tiny_transformer
        result = model._call_processor(
            modality="text",
            processor_inputs={"text": ["hello world"]},
            modality_kwargs={
                "text": {"padding": True, "truncation": "longest_first"},
                "audio": {},
                "image": {},
                "video": {},
            },
            common_kwargs={"return_tensors": "pt"},
        )
        assert "input_ids" in result


def make_char_chat_template_mock(suffix, captured_kwargs=None):
    """Chat-template mock: one token per content character plus ``suffix``, honoring truncation/max_length
    and return_tensors ("pt", "np", or ragged lists)."""

    def mock_apply_chat_template(messages, **kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        text = "".join(str(message["content"]) for message in messages[0])
        ids = [ord(char) for char in text] + suffix
        # Like real tokenizers, max_length truncates unless truncation is explicitly disabled.
        if kwargs.get("max_length") and kwargs.get("truncation") not in (False, "do_not_truncate"):
            ids = ids[: kwargs["max_length"]]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if kwargs.get("return_tensors") == "np":
            return {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        return {"input_ids": [ids]}

    return mock_apply_chat_template


def make_modality_kwargs(**text_kwargs):
    """Modality kwargs for _process_chat_messages with only the text bucket populated."""
    return {"text": text_kwargs, "audio": {}, "image": {}, "video": {}}


class TestProcessChatMessages:
    def test_unsupported_message_modality(self, bert_tiny_transformer):
        """Should raise ValueError when 'message' modality is not in modality_config."""
        model = bert_tiny_transformer
        with pytest.raises(ValueError, match="does not support 'message' modality"):
            model._process_chat_messages(
                messages=[[{"role": "user", "content": "test"}]],
                modality_kwargs={"text": {}, "audio": {}, "image": {}, "video": {}},
                common_kwargs={},
            )

    def test_processing_kwargs_chat_template_passed_through(self, bert_tiny_transformer, monkeypatch):
        """Instance ``processing_kwargs['chat_template']`` should reach ``apply_chat_template`` via preprocess."""
        model = bert_tiny_transformer
        model.processing_kwargs = {"chat_template": {"add_generation_prompt": True, "continue_final_message": False}}
        model.modality_config["message"] = {"method": "forward", "method_output_name": "last_hidden_state"}

        captured_kwargs = {}

        def mock_apply_chat_template(messages, **kwargs):
            captured_kwargs.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        model.preprocess([[{"role": "user", "content": "test"}]])
        assert captured_kwargs["add_generation_prompt"] is True
        assert captured_kwargs["continue_final_message"] is False

    def test_per_call_chat_template_overrides_instance_via_preprocess(self, bert_tiny_transformer, monkeypatch):
        """Per-call ``processing_kwargs={'chat_template': ...}`` should override instance-level via preprocess."""
        model = bert_tiny_transformer
        model.processing_kwargs = {"chat_template": {"add_generation_prompt": False}}
        model.modality_config["message"] = {"method": "forward", "method_output_name": "last_hidden_state"}

        captured_kwargs = {}

        def mock_apply_chat_template(messages, **kwargs):
            captured_kwargs.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        model.preprocess(
            [[{"role": "user", "content": "test"}]],
            processing_kwargs={"chat_template": {"add_generation_prompt": True}},
        )
        assert captured_kwargs["add_generation_prompt"] is True

    def test_chat_truncation_restores_chat_template_suffix(self, bert_tiny_transformer, monkeypatch):
        model = bert_tiny_transformer
        # Suffix preservation is task-independent: the default feature-extraction task confirms it benefits
        # embedders (e.g. last-token pooling), not just causal-LM rerankers.
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}

        # A chat template appends a fixed scoring suffix (e.g. the assistant prefill) that right-truncation
        # cuts off first. The mock mimics this: one token per content character, then ``suffix``.
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))

        def run(content, max_length, truncation=True):
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": content}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation=truncation, max_length=max_length),
                common_kwargs={"return_tensors": "pt"},
            )

        # The mock tokenizes one token per character, so this 13-token content is truncated at max_length=8,
        # dropping the suffix. The restore puts it back on the tail (head kept), with an auto-derived length.
        long_content = "abcdefghijklm"
        ids = run(long_content, max_length=8)["input_ids"].tolist()[0]
        assert len(ids) == 8  # max_length is respected
        assert ids[-3:] == suffix  # the scoring suffix is back at the tail
        assert ids[:5] == [ord(char) for char in long_content[:5]]  # the head is preserved

        # Short content fits, so no truncation happened and the row is returned untouched (no-op restore).
        assert run("ab", max_length=8)["input_ids"].tolist()[0] == [ord("a"), ord("b")] + suffix

        # Escape hatch: with truncation disabled the full sequence (suffix intact) is returned as-is.
        for truncation in (False, "do_not_truncate"):
            assert (
                run(long_content, max_length=8, truncation=truncation)["input_ids"].tolist()[0]
                == [ord(char) for char in long_content] + suffix
            )

        # Flattening path (feature-extraction + can_flatten_inputs): return_tensors is dropped, so the
        # processor returns ragged unpadded lists. The list branch restores the suffix the same way.
        list_ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": long_content}]],
            modality_kwargs=make_modality_kwargs(truncation=True, max_length=8),
            common_kwargs={},
        )["input_ids"][0]
        assert len(list_ids) == 8 and list_ids[-3:] == suffix and list_ids[:5] == [ord(c) for c in long_content[:5]]

    def test_chat_truncation_restore_can_be_disabled(self, bert_tiny_transformer, monkeypatch):
        # The restore is on by default, but ``chat_template_kwargs["restore_suffix"]=False`` (set via
        # ``processing_kwargs["chat_template"]``) turns it off. The flag is a Sentence Transformers concept,
        # so it must be popped and never forwarded to apply_chat_template.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [201, 202, 203]
        forwarded_kwargs = []
        monkeypatch.setattr(
            model.processor, "apply_chat_template", make_char_chat_template_mock(suffix, forwarded_kwargs)
        )

        def run(restore_suffix):
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": "abcdefghij"}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation=True, max_length=8),
                common_kwargs={"return_tensors": "pt"},
                chat_template_kwargs={"restore_suffix": restore_suffix},
            )["input_ids"].tolist()[0]

        # Default behaviour (restore on): the template suffix is put back at the tail.
        assert run(True)[-3:] == suffix
        # Disabled: the raw truncated tail (content tokens, no suffix) is left untouched.
        assert run(False) == [ord(char) for char in "abcdefghij"][:8]
        # restore_suffix is a ST-only flag, so it is never leaked into the apply_chat_template call(s).
        assert forwarded_kwargs and all("restore_suffix" not in kwargs for kwargs in forwarded_kwargs)

    def test_chat_truncation_restore_leaves_non_torch_tensors_untouched(self, bert_tiny_transformer, monkeypatch):
        # The restore only supports torch tensors and token lists. Other containers (e.g. numpy arrays from
        # ``return_tensors="np"``) must be returned untouched, not crash on ndarray truthiness.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}

        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([101, 102, 103]))
        features = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True, truncation=True, max_length=8),
            common_kwargs={"return_tensors": "np"},
        )
        # The truncated numpy row keeps its raw tail (content tokens, no suffix): numpy is not restored.
        assert features["input_ids"].tolist() == [[ord(char) for char in "abcdefghijklm"][:8]]

    def test_chat_truncation_restore_reads_chat_template_bucket_kwargs(self, bert_tiny_transformer, monkeypatch):
        # truncation/max_length can also ride in the chat_template bucket (apply_chat_template accepts them
        # top level). The restore gate must see them there, and the skeleton renders must not inherit them.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True),
            common_kwargs={"return_tensors": "pt"},
            chat_template_kwargs={"truncation": True, "max_length": 8},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

        # The chat bucket also wins over an existing text value (preprocess always sets a default text
        # truncation, which previously collided with an explicit chat-bucket one as a duplicate kwarg).
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True, truncation="longest_first"),
            common_kwargs={"return_tensors": "pt"},
            chat_template_kwargs={"truncation": True, "max_length": 8},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

    def test_chat_truncation_restore_handles_implicit_truncation(self, bert_tiny_transformer, monkeypatch):
        # Tokenizers truncate when max_length is set even if truncation was merely left unset (None), so the
        # restore gate must treat that as truncation-on.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=False, truncation=None, max_length=8),
            common_kwargs={"return_tensors": "pt"},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

    def test_restore_chat_template_suffix_handles_both_padding_sides(self, bert_tiny_transformer, monkeypatch):
        # The restore must locate the real tail via the attention mask (not a fixed [-keep:] slice), so it is
        # correct whether the batch is left- or right-padded. In each batch row 0 was truncated (full width,
        # suffix cut) and row 1 fits (real tokens end with the suffix, the remainder is padding).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *args, **kwargs: suffix)
        messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]

        # Right padding: real tokens left-aligned, pad on the right. The fit row's pad must stay untouched.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 101, 102, 103, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [20, 21, 101, 102, 103, 0]]

        # Left padding: pad on the left, real tokens right-aligned.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [0, 20, 21, 101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [0, 20, 21, 101, 102, 103]]

    def test_restore_chat_template_suffix_when_truncated_row_is_padded(self, bert_tiny_transformer, monkeypatch):
        # pad_to_multiple_of can pad a truncated row past max_length, so a truncated row may itself carry
        # padding. The real tail is read from the mask, so the suffix still lands on real tokens (pad intact).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *args, **kwargs: suffix)
        messages = [[{"role": "user", "content": "x"}]]

        # Right padding: 5 real tokens (suffix cut by truncation) then 3 trailing pads.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 101, 102, 103, 0, 0, 0]]

        # Left padding: 3 leading pads then the 5 real tokens.
        features = {
            "input_ids": torch.tensor([[0, 0, 0, 10, 11, 12, 13, 14]]),
            "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[0, 0, 0, 10, 11, 101, 102, 103]]

    def test_restore_skipped_when_nothing_reached_max_length(self, bert_tiny_transformer, monkeypatch):
        # Rows shorter than max_length were not truncated, so they are skipped without deriving any suffix.
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [1, 2, 3])
        messages = [[{"role": "user", "content": "x"}]]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),  # real length 3
        }
        # real length (3) < max_length (8): not truncated -> skipped, no derivation, untouched.
        model._restore_chat_template_suffix(features, messages, {}, {}, max_length=8)
        assert derived == [] and features["input_ids"].tolist() == [[10, 11, 12, 0, 0, 0]]

    def test_restore_only_touches_rows_that_reached_max_length(self, bert_tiny_transformer, monkeypatch):
        # The truncation check is per row: in a batch where one row was truncated (real length equals
        # max_length), shorter rows kept their tail and must be skipped entirely (no derivation, no write).
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(
            model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [101, 102, 103]
        )
        messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {}, max_length=6)
        # Row 0 (length 6 == max_length) is restored, row 1 (length 3) is left untouched and underived.
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [20, 21, 22, 0, 0, 0]]
        assert len(derived) == 1

    def test_restore_skips_tensor_batches_without_attention_mask(self, bert_tiny_transformer, monkeypatch):
        # Without a mask, real tokens can't be told apart from padding, so the restore must leave the batch
        # untouched rather than stamp the suffix over what may be padding.
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [101, 102])
        features = {"input_ids": torch.tensor([[10, 11, 12, 13]])}
        model._restore_chat_template_suffix(features, [[{"role": "user", "content": "x"}]], {}, {})
        assert derived == [] and features["input_ids"].tolist() == [[10, 11, 12, 13]]

    def test_restore_handles_float_attention_mask(self, bert_tiny_transformer, monkeypatch):
        # Some processors emit float masks: the real lengths must still be ints for the tensor slicing.
        model = bert_tiny_transformer
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: [101, 102, 103])
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
            "attention_mask": torch.ones(1, 6),  # float32
        }
        model._restore_chat_template_suffix(features, [[{"role": "user", "content": "a"}]], {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103]]

    def test_chat_template_suffix_ids_tolerates_unhashable_kwargs(self, bert_tiny_transformer, monkeypatch):
        # chat_template_kwargs may carry unhashable values (e.g. a ``tools`` list). repr() in the cache key
        # keeps the lookup from raising TypeError, so derivation still runs (and caches).
        model = bert_tiny_transformer
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([99]))
        suffix = model._chat_template_suffix_ids([{"role": "user", "content": "x"}], {"tools": [{"name": "f"}]}, {})
        assert suffix == [99]  # no crash from the unhashable kwarg; the fixed marker is the derived suffix

    def test_chat_template_suffix_ids_skips_uninspectable_messages(self, bert_tiny_transformer, monkeypatch):
        # Some templates accept shapes the text-only check can't inspect (a non-dict message, or a content
        # list of plain strings). The restore must skip them (no suffix), not raise AttributeError.
        model = bert_tiny_transformer
        monkeypatch.setattr(
            model.processor, "apply_chat_template", lambda messages, **kwargs: {"input_ids": [[1, 2, 3]]}
        )
        assert model._chat_template_suffix_ids([{"role": "user", "content": "x"}, "not a message"], {}, {}) == []
        assert model._chat_template_suffix_ids([{"role": "user", "content": ["chunk a", "chunk b"]}], {}, {}) == []

    def test_chat_template_suffix_ids_returns_empty_without_variable_content(self, bert_tiny_transformer, monkeypatch):
        # With only kept (system) content there's nothing to vary, so the two renders are identical and there
        # is no reliable suffix boundary -> return [] rather than treating the whole render as the suffix.
        model = bert_tiny_transformer
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([]))
        assert model._chat_template_suffix_ids([{"role": "system", "content": "only a prompt"}], {}, {}) == []

    def test_chat_template_suffix_cache_is_bounded(self, bert_tiny_transformer, monkeypatch):
        # The cache key includes the prompt, so a workload with many distinct prompts must stay bounded.
        model = bert_tiny_transformer
        monkeypatch.setattr("sentence_transformers.base.modules.transformer._CHAT_TEMPLATE_SUFFIX_CACHE_SIZE", 3)
        monkeypatch.setattr(
            model.processor, "apply_chat_template", lambda messages, **kwargs: {"input_ids": [[1, 2, 3]]}
        )
        for index in range(10):
            model._chat_template_suffix_ids([{"role": "system", "content": f"prompt {index}"}], {}, {})
        assert len(model._chat_template_suffix_cache) == 3

    def test_chat_template_suffix_ids_captures_prompt_rendered_in_tail(self, bert_tiny_transformer, monkeypatch):
        # When the template renders the system prompt in the tail, keeping it fixed (not fillered) lets the
        # diff capture it, so a truncated trailing prompt can be restored. Different prompts -> different keys.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # Render the (fillered) non-system content, then the kept system prompt, then a fixed marker (99).
            body = "".join(str(m["content"]) for m in messages[0] if m.get("role") != "system")
            prompt = "".join(str(m["content"]) for m in messages[0] if m.get("role") == "system")
            return {"input_ids": [[ord(c) for c in body] + [ord(c) for c in prompt] + [99]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        # "PQ" (80, 81) is kept fixed and lands in the tail, so it plus the marker (99) form the suffix.
        assert model._chat_template_suffix_ids(
            [{"role": "system", "content": "PQ"}, {"role": "user", "content": "x"}], {}, {}
        ) == [80, 81, 99]
        # A different prompt gets its own cache entry and its own suffix.
        assert model._chat_template_suffix_ids(
            [{"role": "system", "content": "RS"}, {"role": "user", "content": "x"}], {}, {}
        ) == [82, 83, 99]

    def test_chat_template_suffix_ids_keys_on_non_content_fields(self, bert_tiny_transformer, monkeypatch):
        # The skeleton keeps non-content message fields (e.g. ``name``, ``tool_calls``) verbatim and a
        # template may render them, so they are part of the cache key: the same role layout with different
        # fields must not share a cached suffix.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # Render the (fillered) content, then the kept ``name`` field, then a fixed marker (99).
            body = "".join(str(m["content"]) for m in messages[0])
            name = "".join(str(m.get("name", "")) for m in messages[0])
            return {"input_ids": [[ord(c) for c in body] + [ord(c) for c in name] + [99]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        conv_a = [{"role": "user", "content": "x", "name": "AB"}]
        conv_b = [{"role": "user", "content": "x", "name": "CD"}]
        assert model._chat_template_suffix_ids(conv_a, {}, {}) == [65, 66, 99]  # "AB" + marker
        assert model._chat_template_suffix_ids(conv_b, {}, {}) == [67, 68, 99]  # "CD", not A's cached suffix

    def test_chat_template_suffix_ids_mirrors_common_kwargs(self, bert_tiny_transformer, monkeypatch):
        # Common kwargs can change tokenization (a ProcessorMixin spreads them into the text kwargs), so the
        # skeleton renders must receive them and the cache must key on them.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # Emit a different fixed tail depending on a tokenization-affecting common kwarg.
            marker = 98 if kwargs.get("common_kwargs", {}).get("add_special_tokens") is False else 99
            body = "".join(str(m["content"]) for m in messages[0])
            return {"input_ids": [[ord(c) for c in body] + [marker]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        conversation = [{"role": "user", "content": "x"}]
        assert model._chat_template_suffix_ids(conversation, {}, {}, {"add_special_tokens": False}) == [98]
        assert model._chat_template_suffix_ids(conversation, {}, {}, {}) == [99]

    def test_restore_chat_template_suffix_is_per_row_and_skips_multimodal(self, bert_tiny_transformer, monkeypatch):
        # A heterogeneous batch: the text-only row is restored per-row, while the multimodal row is left
        # untouched (its trailing suffix can't be isolated from a text skeleton, so it returns []).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]

        def mock_apply_chat_template(messages, **kwargs):
            # Skeleton content is the filler string, so emit one content token plus the fixed suffix.
            text = "".join(str(message["content"]) for message in messages[0])
            return {"input_ids": [[ord(text[0])] + suffix]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        messages = [
            [{"role": "user", "content": "text"}],
            [{"role": "user", "content": [{"type": "text", "text": "t"}, {"type": "image", "image": "img"}]}],
        ]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"][0].tolist() == [10, 11, 12, 101, 102, 103]  # text row restored
        assert features["input_ids"][1].tolist() == [20, 21, 22, 23, 24, 25]  # multimodal row untouched

    @pytest.mark.slow
    def test_chat_truncation_restores_suffix_with_real_chat_template(self):
        # Integration check against a real chat-template model (the other suffix tests mock the tokenizer), so
        # it exercises the real apply_chat_template dispatch and two-filler derivation. The Llama-2 template
        # ends the user turn with `[/INST]`, which truncation drops and the restore must put back.
        model = Transformer(TINY_LLAMA, transformer_task="feature-extraction")
        tokenizer = model.tokenizer

        def process(max_length, restore_suffix=True):
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": "word " * 50}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation="longest_first", max_length=max_length),
                common_kwargs={"return_tensors": "pt"},
                chat_template_kwargs={"add_generation_prompt": True, "restore_suffix": restore_suffix},
            )["input_ids"][0].tolist()

        # The conversation overflows max_length=16, so `[/INST]` is truncated off and then restored.
        restored = process(16)
        decoded = tokenizer.decode(restored)
        assert len(restored) == 16  # truncation is still respected
        assert decoded.rstrip().endswith("[/INST]")  # the template suffix is back at the tail
        assert "word" in decoded  # real content is kept at the head, not overwritten wholesale

        # With the restore disabled the same input ends on a content token: the bug this feature fixes.
        assert not tokenizer.decode(process(16, restore_suffix=False)).rstrip().endswith("[/INST]")


class TestModelLoading:
    def test_invalid_backend_error(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            Transformer(TINY_BERT, backend="invalid_backend")

    def test_peft_seq_classification_no_architectures(self, monkeypatch):
        """PeftConfig has no 'architectures' attr; sequence-classification init should not crash."""

        class FakePeftConfig:
            """Minimal stand-in for PeftConfig that intentionally lacks 'architectures'."""

            base_model_name_or_path = TINY_BERT

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        monkeypatch.setattr(transformer_module, "find_adapter_config_file", lambda *a, **kw: "some_file")
        monkeypatch.setattr(transformer_module, "is_peft_available", lambda: True)

        import peft

        monkeypatch.setattr(peft, "PeftConfig", FakePeftConfig)

        # PeftConfig lacks 'architectures', so the sequence-classification guard
        # (which accesses config.architectures) will raise AttributeError.
        with pytest.raises(AttributeError):
            Transformer(TINY_BERT, transformer_task="sequence-classification")

    def test_peft_non_torch_backend_error(self, monkeypatch):
        """PEFT models should raise an error for non-torch backends."""
        monkeypatch.setattr(transformer_module, "find_adapter_config_file", lambda *a, **kw: "some_file")
        with pytest.raises(ValueError, match="PEFT models can currently only be loaded"):
            Transformer(TINY_BERT, backend="onnx")


class TestModalityInference:
    def test_infer_modalities_edge_cases_returns_none_for_unknown(self, bert_tiny_transformer):
        """Should return None for model types not in _EDGE_CASE_MODALITY_CONFIGS."""
        model = bert_tiny_transformer
        result = model.infer_modalities_edge_cases(model.model, model.processor)
        assert result is None

    def test_infer_modalities_from_processor_text(self, bert_tiny_transformer):
        """Should identify 'text' modality for a tokenizer-based processor."""
        model = bert_tiny_transformer
        modalities = model.infer_modalities_from_processor(model.processor)
        assert modalities == ["text"]


class TestGetMethodOutputFields:
    def test_with_model_output_return_type(self):
        """Should extract field names from a ModelOutput return type, and validate _infer_method_output_name."""
        model = AutoModel.from_pretrained(TINY_BERT)
        fields = Transformer._get_method_output_fields(model.forward)
        assert fields is not None
        assert "last_hidden_state" in fields

        # _infer_method_output_name: valid name should be returned, invalid should return None
        assert Transformer._infer_method_output_name("last_hidden_state", model.forward) == "last_hidden_state"
        assert Transformer._infer_method_output_name("nonexistent_field", model.forward) is None

    def test_with_no_return_annotation(self):
        """Should return None when method has no return type annotation."""

        def plain_func(x):
            return x

        assert Transformer._get_method_output_fields(plain_func) is None

    def test_with_exception_in_get_type_hints(self, monkeypatch):
        """Should return None when get_type_hints raises."""

        def raise_boom(*a, **kw):
            raise Exception("boom")

        monkeypatch.setattr(transformer_module, "get_type_hints", raise_boom)
        model = AutoModel.from_pretrained(TINY_BERT)
        assert Transformer._get_method_output_fields(model.forward) is None


class TestGetProcessorAttributes:
    def test_returns_none_for_tokenizer(self, bert_tiny_transformer):
        """Tokenizer-only processors don't have get_attributes or attributes."""
        result = bert_tiny_transformer._get_processor_attributes()
        assert result is None or isinstance(result, (dict, list))


class TestSerialization:
    def test_save_and_load_roundtrip(self, bert_tiny_transformer, tmp_path):
        """save() then load() should produce an equivalent Transformer with identical outputs."""
        model = bert_tiny_transformer
        texts = ["hello world", "goodbye world"]
        features = batch_to_device(model.preprocess(texts), model.model.device)
        with torch.no_grad():
            out1 = model.forward(features)

        save_dir = str(tmp_path / "model")
        model.save(save_dir)
        reloaded = Transformer.load(save_dir)

        assert type(reloaded.auto_model).__name__ == type(model.auto_model).__name__
        assert reloaded.module_output_name == model.module_output_name
        assert reloaded.modality_config == model.modality_config

        features = batch_to_device(reloaded.preprocess(texts), reloaded.model.device)
        with torch.no_grad():
            out2 = reloaded.forward(features)

        for key in out1:
            v1, v2 = out1[key], out2[key]
            if isinstance(v1, torch.Tensor):
                assert torch.allclose(v1.cpu(), v2.cpu(), atol=1e-5), f"Output '{key}' differs after save/load"
            else:
                assert v1 == v2

    def test_max_seq_length_save_and_load(self, bert_tiny_transformer, tmp_path):
        """A custom max_seq_length should be preserved after save/load."""
        model = bert_tiny_transformer
        assert model.tokenizer.model_max_length != 42
        model.max_seq_length = 42
        assert model.tokenizer.model_max_length == 42
        save_dir = str(tmp_path / "model")
        model.save(save_dir)
        reloaded = Transformer.load(save_dir)
        assert reloaded.max_seq_length == 42
        assert reloaded.tokenizer.model_max_length == 42

    def test_processing_kwargs_save_load_roundtrip(self, tmp_path):
        """processing_kwargs should be persisted to config JSON and restored on load."""
        processing_kwargs = {
            "text": {"truncation": "only_first"},
            "chat_template": {"add_generation_prompt": True},
        }
        transformer = Transformer(TINY_BERT, processing_kwargs=processing_kwargs)
        assert transformer.processing_kwargs == processing_kwargs

        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)

        # Verify the JSON file contains processing_kwargs
        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert config["processing_kwargs"] == processing_kwargs

        # Verify the reloaded Transformer has the same processing_kwargs
        reloaded = Transformer.load(save_dir)
        assert reloaded.processing_kwargs == processing_kwargs

    def test_processing_kwargs_omitted_from_config_when_empty(self, bert_tiny_transformer, tmp_path):
        """Empty processing_kwargs should not appear in the saved config JSON."""
        assert bert_tiny_transformer.processing_kwargs == {}
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert "processing_kwargs" not in config

    def test_unpad_inputs_omitted_from_config_when_inferred(self, bert_tiny_transformer, tmp_path):
        """Inferred unpad_inputs (None) should not appear in the saved config JSON."""
        assert bert_tiny_transformer.unpad_inputs is None
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert "unpad_inputs" not in config

    @pytest.mark.parametrize("unpad_inputs", [True, False])
    def test_unpad_inputs_save_load_roundtrip(self, tmp_path, unpad_inputs):
        """Explicitly set unpad_inputs should appear in the saved config and survive a roundtrip."""
        transformer = Transformer(TINY_BERT, unpad_inputs=unpad_inputs)
        assert transformer.unpad_inputs is unpad_inputs
        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert config["unpad_inputs"] is unpad_inputs

        reloaded = Transformer.load(save_dir)
        assert reloaded.unpad_inputs is unpad_inputs

    def test_get_config_dict(self, bert_tiny_transformer):
        config = bert_tiny_transformer.get_config_dict()
        assert "modality_config" in config
        assert isinstance(config["modality_config"], dict)

    def test_get_config_dict_tuple_keys_serialized(self, bert_tiny_transformer):
        """Tuple modality keys should be serialized to plus-separated strings."""
        model = bert_tiny_transformer
        model.modality_config[("image", "text")] = {
            "method": "forward",
            "method_output_name": "last_hidden_state",
        }
        config = model.get_config_dict()
        assert "image+text" in config["modality_config"]

    def test_repr(self, bert_tiny_transformer):
        r = repr(bert_tiny_transformer)
        assert "Transformer(" in r
        assert "architecture" in r

    def test_load_config_deserializes_tuple_keys(self, bert_tiny_transformer, tmp_path):
        """load_config should deserialize plus-separated keys back to tuples."""
        model = bert_tiny_transformer
        save_dir = str(tmp_path / "model")
        model.modality_config[("image", "text")] = {
            "method": "forward",
            "method_output_name": "last_hidden_state",
        }
        model.save(save_dir)

        config = Transformer.load_config(save_dir)
        assert ("image", "text") in config["modality_config"]

    def test_load_config_strips_trust_remote_code(self, bert_tiny_transformer, tmp_path):
        """load_config should remove trust_remote_code from sub-dicts."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        assert config_path.exists(), "save() must always create sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        config.setdefault("model_args", {})["trust_remote_code"] = True
        config_path.write_text(json.dumps(config))

        loaded = Transformer.load_config(save_dir)
        assert "trust_remote_code" not in loaded.get("model_args", {})

    def test_load_config_default_modality_config_for_old_models(self, bert_tiny_transformer, tmp_path):
        """Models saved without modality_config should get the default text-only config."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        assert config_path.exists(), "save() must always create sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        config.pop("modality_config", None)
        config.pop("module_output_name", None)
        config_path.write_text(json.dumps(config))

        loaded = Transformer.load_config(save_dir)
        expected_config, expected_output = TRANSFORMER_TASK_DEFAULTS["feature-extraction"]
        assert loaded["modality_config"] == expected_config
        assert loaded["module_output_name"] == expected_output


class TestGetDefaultModalityConfig:
    def test_default_and_explicit_tasks(self):
        """Dict lookup should work for all tasks, and missing key should default to feature-extraction."""
        for task in ("feature-extraction", "sequence-classification", "text-generation", "fill-mask"):
            assert (
                Transformer._get_default_modality_config({"transformer_task": task}) == TRANSFORMER_TASK_DEFAULTS[task]
            )
        assert Transformer._get_default_modality_config({}) == TRANSFORMER_TASK_DEFAULTS["feature-extraction"]


class TestLoadInitKwargs:
    def test_merges_config_and_overrides(self, bert_tiny_transformer, tmp_path):
        """_load_init_kwargs should merge config, hub_kwargs, user overrides, and pass cache_dir for the decorator."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        kwargs = Transformer._load_init_kwargs(
            model_name_or_path=save_dir,
            model_kwargs={"torch_dtype": "float16"},
            cache_folder="/tmp/cache",
        )
        assert "model_kwargs" in kwargs
        assert kwargs["model_kwargs"]["torch_dtype"] == "float16"
        # cache_folder is distributed into each kwargs dict directly
        assert kwargs["model_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["processor_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["config_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["backend"] == "torch"


class TestEncoderOnlySaveLoadRoundtrip:
    """Test save/load roundtrip for encoder-only models extracted from encoder-decoder architectures.

    Each encoder-decoder architecture in ``_ENCODER_ONLY_MODELS`` (plus the T5Gemma special cases)
    should produce the correct encoder class, and outputs should be identical after save/load.
    """

    # (model_name, expected_class_name, extra_kwargs)
    # extra_kwargs may contain config_kwargs and is_audio
    ENCODER_ONLY_TEXT_MODELS = [
        ("hf-internal-testing/tiny-random-T5Model", "T5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-mt5", "MT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-UMT5ForTokenClassification", "UMT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-LongT5Model", "LongT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-ProphetNetModel", "ProphetNetEncoder", {}),
        ("hf-internal-testing/tiny-random-SwitchTransformersModel", "SwitchTransformersEncoderModel", {}),
        ("hf-internal-testing/tiny-random-BlenderbotModel", "BlenderbotEncoder", {}),
        ("hf-internal-testing/tiny-random-BlenderbotSmallModel", "BlenderbotSmallEncoder", {}),
        ("hf-internal-testing/tiny-random-M2M100Model", "M2M100Encoder", {}),
        ("hf-internal-testing/tiny-random-PegasusModel", "PegasusEncoder", {}),
        ("hf-internal-testing/tiny-random-PegasusXModel", "PegasusXEncoder", {}),
        (
            "hf-internal-testing/tiny-random-MarianModel",
            "MarianEncoder",
            # The default pad_token is at idx 58100, but embeddings were reduced to 99 in the tiny model
            {"config_kwargs": {"pad_token_id": 1}},
        ),
    ]

    ENCODER_ONLY_AUDIO_MODELS = [
        (
            "hf-internal-testing/tiny-random-WhisperModel",
            "WhisperEncoder",
            {"config_kwargs": {"max_source_positions": 1500}},
        ),
        ("hf-internal-testing/tiny-random-MoonshineForConditionalGeneration", "MoonshineEncoder", {}),
    ]

    ENCODER_ONLY_LARGE_MODELS = [
        ("google/t5gemma-s-s-prefixlm", "T5GemmaEncoderModel", {}),
        ("google/t5gemma-2-270m-270m", "T5Gemma2Encoder", {}),
    ]

    @staticmethod
    def _load_transformer(model_name: str, extra_kwargs: dict) -> Transformer:
        config_kwargs = extra_kwargs.get("config_kwargs", {})
        return Transformer(
            model_name_or_path=model_name,
            model_kwargs={"ignore_mismatched_sizes": True},
            config_kwargs=config_kwargs,
        )

    @staticmethod
    def _get_inputs(transformer: Transformer, is_audio: bool) -> list:
        if is_audio:
            return [np.random.randn(16000).astype(np.float32) for _ in range(2)]
        return ["hello world", "goodbye world"]

    @staticmethod
    def _assert_outputs_match(out1: dict, out2: dict) -> None:
        for key in out1:
            v1, v2 = out1[key], out2[key]
            if isinstance(v1, torch.Tensor):
                assert torch.allclose(v1, v2, atol=1e-5), f"Outputs for key {key!r} differ after save/load"
            else:
                assert v1 == v2, f"Outputs for key {key!r} differ after save/load"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        [(TINY_BERT, "BertModel", {})] + ENCODER_ONLY_TEXT_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_text_model_roundtrip(self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict):
        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        inputs = self._get_inputs(transformer, is_audio=False)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        ENCODER_ONLY_AUDIO_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_audio_model_roundtrip(
        self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict
    ):
        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        inputs = self._get_inputs(transformer, is_audio=True)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        ENCODER_ONLY_LARGE_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_large_model_roundtrip(
        self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict
    ):
        if parse_version(transformers_version) < parse_version("5.0.0") and expected_class_name == "T5Gemma2Encoder":
            pytest.skip("T5Gemma2Encoder requires transformers>=5.0.0")
        if (
            parse_version(transformers_version) < parse_version("4.54.1")
            and expected_class_name == "T5GemmaEncoderModel"
        ):
            pytest.skip("T5GemmaEncoderModel requires transformers>=4.54.1")

        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        if expected_class_name == "T5Gemma2Encoder":
            transformer.auto_model.config._attn_implementation = "eager"

        inputs = self._get_inputs(transformer, is_audio=False)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        if expected_class_name == "T5Gemma2Encoder":
            reloaded.auto_model.config._attn_implementation = "eager"

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)


class TestCanFlattenInputs:
    def test_false_for_non_feature_extraction_task(self):
        """Flattening requires the feature-extraction task."""
        transformer = Transformer(TINY_BERT, transformer_task="sequence-classification")
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_for_non_torch_backend(self):
        """Flattening requires the torch backend."""
        try:
            transformer = Transformer(TINY_BERT, backend="onnx")
        except (ImportError, Exception):
            pytest.skip("ONNX backend not available")
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_when_no_flash_attention(self):
        """Without flash attention, flattening should be disabled."""
        transformer = Transformer(TINY_BERT)
        # Default attn_implementation is not flash_attention_2
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_when_backend_incompatible(self, monkeypatch):
        """If the model's auto_model reports backend incompatibility, flattening is disabled."""
        transformer = Transformer(TINY_BERT)
        monkeypatch.setattr(transformer.model, "is_backend_compatible", lambda: False)
        assert transformer._can_flatten_inputs() is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_true_with_flash_attention(self):
        """With flash attention and a compatible model, flattening should be enabled."""
        try:
            import kernels  # noqa: F401
        except ImportError:
            pytest.skip("kernels library not available")

        transformer = Transformer(
            TINY_BERT,
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        )
        assert transformer._can_flatten_inputs() is True
        assert transformer.can_flatten_inputs is True
        assert transformer.data_collator is not None

    def test_unpad_inputs_false_forces_padding(self):
        """Setting unpad_inputs=False should disable flattening regardless of prerequisites."""
        transformer = Transformer(TINY_BERT, unpad_inputs=False)
        assert transformer.unpad_inputs is False
        assert transformer.can_flatten_inputs is False

    def test_unpad_inputs_true_warns_when_prerequisites_not_met(self, caplog):
        """Setting unpad_inputs=True should warn if prerequisites are not met."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, unpad_inputs=True)
        assert transformer.can_flatten_inputs is False
        assert "unpad_inputs=True was set" in caplog.text

    def test_unpad_inputs_none_auto_detects(self):
        """Setting unpad_inputs=None (default) should auto-detect."""
        transformer = Transformer(TINY_BERT)
        assert transformer.unpad_inputs is None
        # Without flash attention, auto-detect should disable flattening
        assert transformer.can_flatten_inputs is False

    def test_unpad_inputs_setter_re_evaluates(self):
        """Setting unpad_inputs after init should re-evaluate can_flatten_inputs."""
        transformer = Transformer(TINY_BERT)
        assert transformer.can_flatten_inputs is False

        # Setting to False explicitly should keep it disabled
        transformer.unpad_inputs = False
        assert transformer.unpad_inputs is False
        assert transformer.can_flatten_inputs is False

        # Setting back to None should re-evaluate (still False without flash attention)
        transformer.unpad_inputs = None
        assert transformer.unpad_inputs is None
        assert transformer.can_flatten_inputs is False


class TestConditionalFlattening:
    """Test that preprocess only flattens text-only inputs, even when can_flatten_inputs is True."""

    @pytest.fixture()
    def flatten_ready_transformer(self):
        """A BERT transformer with can_flatten_inputs=True and a mock data_collator."""
        transformer = Transformer(TINY_BERT)
        transformer.can_flatten_inputs = True
        transformer.data_collator = MagicMock(
            return_value={"input_ids": torch.tensor([1, 2, 3]), "position_ids": torch.tensor([0, 1, 2])}
        )
        return transformer

    def test_text_inputs_are_flattened(self, flatten_ready_transformer):
        """Text inputs should be flattened when can_flatten_inputs is True."""
        flatten_ready_transformer.preprocess(["hello world", "another sentence"])
        flatten_ready_transformer.data_collator.assert_called_once()

    def test_text_inputs_not_flattened_when_disabled(self):
        """Text inputs should NOT be flattened when can_flatten_inputs is False."""
        transformer = Transformer(TINY_BERT)
        assert transformer.can_flatten_inputs is False
        transformer.data_collator = MagicMock()
        transformer.preprocess(["hello world"])
        transformer.data_collator.assert_not_called()

    def test_multimodal_messages_skip_flattening(self, flatten_ready_transformer, monkeypatch):
        """Messages with non-text content should NOT be flattened even when can_flatten_inputs is True."""
        transformer = flatten_ready_transformer
        transformer.modality_config["message"] = {"method": "forward"}

        multimodal_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://example.com/img.jpg"},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        ]
        monkeypatch.setattr(
            transformer.input_formatter,
            "parse_inputs",
            lambda inputs: ("message", {"message": multimodal_messages}, {}),
        )
        monkeypatch.setattr(
            transformer,
            "_call_processor",
            lambda modality, inputs, modality_kwargs, common_kwargs, **_: {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        )

        result = transformer.preprocess(["dummy"])
        transformer.data_collator.assert_not_called()
        assert "attention_mask" in result

    def test_text_only_messages_are_flattened(self, flatten_ready_transformer, monkeypatch):
        """Messages with only text content should be flattened when can_flatten_inputs is True."""
        transformer = flatten_ready_transformer
        transformer.modality_config["message"] = {"method": "forward"}

        text_messages = [[{"role": "user", "content": "hello world"}]]
        monkeypatch.setattr(
            transformer.input_formatter,
            "parse_inputs",
            lambda inputs: ("message", {"message": text_messages}, {}),
        )
        monkeypatch.setattr(
            transformer,
            "_call_processor",
            lambda modality, inputs, modality_kwargs, common_kwargs, **_: {"input_ids": [[1, 2, 3]]},
        )

        transformer.preprocess(["dummy"])
        transformer.data_collator.assert_called_once()


@pytest.mark.skipif(
    Version(transformers_version) >= Version("5.0.0"),
    reason="Test only applies to transformers v4",
)
def test_any_to_any_requires_transformers_v5():
    with pytest.raises(ImportError, match="transformers v5"):
        Transformer("hf-internal-testing/tiny-random-LlamaForCausalLM", transformer_task="any-to-any")


class TestCountMediaPerSample:
    def test_multiple_images_per_sample(self):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "a"},
                        {"type": "image", "image": "b"},
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
            [{"role": "user", "content": [{"type": "image", "image": "c"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [2, 1, 0]
        assert num_videos == [0, 0, 0]

    def test_videos(self):
        messages = [
            [{"role": "user", "content": [{"type": "video", "video": "v1"}, {"type": "text", "text": "describe"}]}],
            [{"role": "user", "content": [{"type": "video", "video": "v2"}, {"type": "video", "video": "v3"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [0, 0]
        assert num_videos == [1, 2]

    def test_mixed_images_and_videos(self):
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": "i"}, {"type": "video", "video": "v"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [1]
        assert num_videos == [1]

    def test_text_only(self):
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "user", "content": "plain string"}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [0, 0]
        assert num_videos == [0, 0]

    def test_multi_turn_messages(self):
        messages = [
            [
                {"role": "user", "content": [{"type": "image", "image": "a"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]},
                {"role": "user", "content": [{"type": "image", "image": "b"}, {"type": "text", "text": "and this?"}]},
            ],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [2]
        assert num_videos == [0]

    def test_empty_batch(self):
        assert _count_media_per_sample([]) == ([], [])


class TestTransformerSaveLoadRoundtrip:
    def test_save_load_produces_same_output(self, bert_tiny_transformer: Transformer, tmp_path):
        """Test that a Transformer can be saved and loaded, producing identical output."""
        save_dir = str(tmp_path / "transformer_roundtrip")
        bert_tiny_transformer.save(save_dir)

        # Verify config.json exists and has expected keys
        config_path = Path(save_dir) / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config_data = json.load(f)
        assert "model_type" in config_data
        assert "architectures" in config_data

        # Verify sentence_bert_config.json exists
        st_config_path = Path(save_dir) / Transformer.config_file_name
        assert st_config_path.exists()

        # Load and compare outputs on CPU to avoid cross-device precision issues
        reloaded = Transformer.load(save_dir)
        bert_tiny_transformer.cpu()
        reloaded.cpu()

        features_original = bert_tiny_transformer.preprocess(["Hello world"])
        features_reloaded = reloaded.preprocess(["Hello world"])

        with torch.no_grad():
            out_original = bert_tiny_transformer.forward(features_original)
            out_reloaded = reloaded.forward(features_reloaded)

        assert torch.equal(out_original["token_embeddings"], out_reloaded["token_embeddings"])


class TestTransformerForwardEmptyBatch:
    def test_forward_empty_inputs(self, bert_tiny_transformer: Transformer):
        """Test that forward with an empty batch (zero-length tensors) either raises a RuntimeError
        or returns empty outputs, depending on the PyTorch/transformers version."""
        features = {
            "input_ids": torch.zeros(0, 5, dtype=torch.long),
            "attention_mask": torch.zeros(0, 5, dtype=torch.long),
        }
        features = batch_to_device(features, bert_tiny_transformer.model.device)
        try:
            output = bert_tiny_transformer.forward(features)
        except RuntimeError:
            pass
        else:
            assert output["token_embeddings"].shape[0] == 0
