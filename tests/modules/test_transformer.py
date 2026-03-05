from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
import torch
from packaging.version import parse as parse_version
from transformers import AutoModel
from transformers import __version__ as transformers_version

from sentence_transformers.base.modules.Transformer import TRANSFORMER_TASK_DEFAULTS, set_temporary_class_attrs
from sentence_transformers.modules import Transformer
from sentence_transformers.util import batch_to_device

transformer_module = sys.modules[Transformer.__module__]

TINY_BERT = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture()
def bert_tiny_transformer(stsb_bert_tiny_model) -> Transformer:
    """A lightweight BERT Transformer for reuse across tests."""
    return stsb_bert_tiny_model[0]


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
        """do_lower_case should add Lowercase normalization to the tokenizer."""
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        assert transformer.do_lower_case is True
        tokens = transformer.tokenizer.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_tokenizer_name_or_path_warning(self, caplog):
        """tokenizer_name_or_path should emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, tokenizer_name_or_path=TINY_BERT)
        assert any(
            "tokenizer_name_or_path" in record.message and "deprecated" in record.message for record in caplog.records
        )
        assert transformer is not None


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


class TestGetWordEmbeddingDimension:
    def test_standard_hidden_size(self, bert_tiny_transformer):
        dim = bert_tiny_transformer.get_word_embedding_dimension()
        assert dim == bert_tiny_transformer.config.hidden_size

    def test_hidden_sizes_list(self, bert_tiny_transformer):
        """Models with hidden_sizes (list) should return the last element."""
        model = bert_tiny_transformer
        del model.config.hidden_size
        model.config.hidden_sizes = [64, 128, 256]
        assert model.get_word_embedding_dimension() == 256

    def test_hidden_dim(self, bert_tiny_transformer):
        """Models with hidden_dim should return it."""
        model = bert_tiny_transformer
        del model.config.hidden_size
        model.config.hidden_dim = 384
        assert model.get_word_embedding_dimension() == 384

    def test_raises_when_no_dimension_found(self, bert_tiny_transformer):
        """Should raise ValueError when no dimension attribute is found."""
        model = bert_tiny_transformer
        del model.config.hidden_size
        with pytest.raises(ValueError, match="Could not determine embedding dimension"):
            model.get_word_embedding_dimension()

    def test_projection_dim_for_sentence_embedding(self, bert_tiny_transformer):
        """When module_output_name is 'sentence_embedding', projection_dim takes priority."""
        model = bert_tiny_transformer
        model.module_output_name = "sentence_embedding"
        model.config.projection_dim = 512
        assert model.get_word_embedding_dimension() == 512

    def test_text_config_fallback(self, bert_tiny_transformer):
        """Should fall back to text_config.hidden_size when main config lacks it."""
        model = bert_tiny_transformer
        del model.config.hidden_size

        class FakeTextConfig:
            hidden_size = 768

        model.config.text_config = FakeTextConfig()
        assert model.get_word_embedding_dimension() == 768


class TestGetPromptLength:
    def test_prompt_length_cached(self, bert_tiny_transformer):
        """Second call with the same prompt should use the cache."""
        model = bert_tiny_transformer
        model._prompt_length_mapping.clear()
        len1 = model._get_prompt_length("search query: ")
        len2 = model._get_prompt_length("search query: ")
        assert len1 == len2
        assert len(model._prompt_length_mapping) == 1

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


class TestModelLoading:
    def test_invalid_backend_error(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            Transformer(TINY_BERT, backend="invalid_backend")

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
        modalities = model.infer_modalities_from_processor(model.model, model.processor)
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

    def test_get_config_dict(self, bert_tiny_transformer):
        config = bert_tiny_transformer.get_config_dict()
        assert "modality_config" in config
        assert isinstance(config["modality_config"], dict)

    def test_get_config_dict_tuple_keys_serialized(self, bert_tiny_transformer):
        """Tuple modality keys should be serialized to comma-separated strings."""
        model = bert_tiny_transformer
        model.modality_config[("image", "text")] = {
            "method": "forward",
            "method_output_name": "last_hidden_state",
        }
        config = model.get_config_dict()
        assert "image,text" in config["modality_config"]

    def test_repr(self, bert_tiny_transformer):
        r = repr(bert_tiny_transformer)
        assert "Transformer(" in r
        assert "architecture" in r

    def test_load_config_deserializes_tuple_keys(self, bert_tiny_transformer, tmp_path):
        """load_config should deserialize comma-separated keys back to tuples."""
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
        """_load_init_kwargs should merge config, hub_kwargs, user overrides, and map cache_folder."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        kwargs = Transformer._load_init_kwargs(
            model_name_or_path=save_dir,
            model_kwargs={"torch_dtype": "float16"},
            cache_folder="/tmp/cache",
        )
        assert "model_args" in kwargs
        assert kwargs["model_args"]["torch_dtype"] == "float16"
        assert kwargs["cache_dir"] == "/tmp/cache"
        assert kwargs["backend"] == "torch"


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, expected_class_name",
    [
        (TINY_BERT, "BertModel"),
        ("hf-internal-testing/tiny-random-t5", "T5EncoderModel"),
        ("hf-internal-testing/tiny-random-mt5", "MT5EncoderModel"),
        ("google/t5gemma-s-s-prefixlm", "T5GemmaEncoderModel"),
        ("google/t5gemma-2-270m-270m", "T5Gemma2Encoder"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available() and torch.cuda.device_count() == 0, reason="Requires torch backend")
def test_transformer_load_save_roundtrip(tmp_path: Path, model_name: str, expected_class_name: str):
    if parse_version(transformers_version) < parse_version("5.0.0") and expected_class_name == "T5Gemma2Encoder":
        pytest.skip("T5Gemma2Encoder requires transformers>=5.0.0")
    if parse_version(transformers_version) < parse_version("4.54.1") and expected_class_name == "T5GemmaEncoderModel":
        pytest.skip("T5GemmaEncoderModel requires transformers>=4.54.1")

    transformer = Transformer(model_name_or_path=model_name)
    actual_class_name = type(transformer.auto_model).__name__
    assert actual_class_name == expected_class_name

    if expected_class_name == "T5Gemma2Encoder":
        transformer.auto_model.config._attn_implementation = "eager"

    texts = ["hello world", "goodbye world"]
    features = transformer.tokenize(texts)
    with torch.no_grad():
        out1 = transformer(features)

    save_dir = tmp_path / "model"
    transformer.save(str(save_dir))
    reloaded = Transformer.load(str(save_dir))

    actual_class_name = type(reloaded.auto_model).__name__
    assert actual_class_name == expected_class_name

    if expected_class_name == "T5Gemma2Encoder":
        reloaded.auto_model.config._attn_implementation = "eager"

    features = reloaded.tokenize(texts)
    with torch.no_grad():
        out2 = reloaded(features)

    for key in out1.keys():
        value1 = out1[key]
        value2 = out2[key]
        if isinstance(value1, torch.Tensor):
            assert torch.allclose(value1, value2), f"Outputs for key {key} differ after save/load"
        else:
            assert value1 == value2, f"Outputs for key {key} differ after save/load"
