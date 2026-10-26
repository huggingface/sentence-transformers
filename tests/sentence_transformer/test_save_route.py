from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Dense, Normalize, Router
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def static_embedding(scale=1.0):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "cat": 1, "dog": 2, "query": 3, "doc": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    weights = (
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 1.0, 2.0]]) * scale
    )
    return StaticEmbedding(tokenizer, embedding_weights=weights)


def make_model(router_first):
    if router_first:
        router = Router.for_query_document([static_embedding(), Dense(3, 3)], [static_embedding(2), Dense(3, 3)])
        modules = [router, Normalize()]
    else:
        router = Router.for_query_document([Dense(3, 3)], [Dense(3, 3), Normalize()])
        modules = [static_embedding(), router, Dense(3, 3)]
    model = SentenceTransformer(
        modules=modules,
        prompts={"query": "query ", "document": "doc ", "other": "dog "},
        default_prompt_name="other",
        similarity_fn_name="dot",
        truncate_dim=2,
        device="cpu",
    )
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.mark.parametrize("router_first", [True, False])
@pytest.mark.parametrize("route", ["query", "document"])
@pytest.mark.parametrize("safe_serialization", [True, False])
def test_save_route_round_trip(tmp_path, router_first, route, safe_serialization):
    model = make_model(router_first)
    method = "encode_query" if route == "query" else "encode_document"
    sentences = ["cat", "dog", "cat dog"]
    expected = getattr(model, method)(sentences)
    original_modules = list(model.children())
    original_card_dir = model.model_card_data.save_dir
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    model.save_route(str(tmp_path), route, safe_serialization=safe_serialization, create_model_card=False)
    loaded = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)

    assert not any(isinstance(module, Router) for module in loaded.modules())
    np.testing.assert_allclose(getattr(loaded, method)(sentences), expected, rtol=1e-6, atol=1e-7)
    assert loaded.prompts == model.prompts
    assert loaded.default_prompt_name == model.default_prompt_name
    assert loaded.similarity_fn_name == "dot"
    assert loaded.truncate_dim == 2
    assert list(model.children()) == original_modules
    assert model.model_card_data.model is model
    assert model.model_card_data.save_dir == original_card_dir
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
    config = json.loads((tmp_path / "modules.json").read_text())
    assert len(config) == (4 if not router_first and route == "document" else 3)
    assert not list(tmp_path.rglob("*Router*"))


def test_preserve_forward_kwargs_when_renumbering(tmp_path):
    model = SentenceTransformer(
        modules=OrderedDict(
            [
                ("encoder", static_embedding()),
                ("router", Router.for_query_document([Normalize()], [Dense(3, 3)])),
                ("last", Normalize()),
            ]
        ),
        device="cpu",
    )
    model.module_kwargs = {"last": ["custom_argument"]}
    model.save_route(str(tmp_path), "document", create_model_card=False)
    config = json.loads((tmp_path / "modules.json").read_text())
    assert config[-1]["kwargs"] == ["custom_argument"]
    assert model.module_kwargs == {"last": ["custom_argument"]}


def test_model_card_uses_exported_architecture(tmp_path):
    model = make_model(True)
    model._model_card_text = "An obsolete card for the full Router model."
    model.save_route(str(tmp_path), "query")
    card = (tmp_path / "README.md").read_text()
    assert "obsolete card" not in card
    assert "Router(" not in card
    assert model._model_card_text == "An obsolete card for the full Router model."
    assert model.model_card_data.model is model


@pytest.mark.parametrize("case", ["no_router", "multiple", "unknown", "nested"])
def test_invalid_route_export_does_not_write(tmp_path, case):
    router = Router.for_query_document([Normalize()], [Normalize()])
    if case == "no_router":
        modules = [static_embedding()]
        match = "exactly one"
    elif case == "multiple":
        modules = [static_embedding(), router, Router.for_query_document([Normalize()], [Normalize()])]
        match = "exactly one"
    elif case == "nested":
        modules = [static_embedding(), Router({"query": [router]})]
        match = "nested Router"
    else:
        modules = [static_embedding(), router]
        match = "Unknown route"
    model = SentenceTransformer(modules=modules, device="cpu")
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match=match):
        model.save_route(str(destination), "missing" if case == "unknown" else "query")
    assert not destination.exists()


@pytest.mark.parametrize("route, length", [("query", 4), ("document", 7)])
def test_export_preserves_task_specific_transformer_preprocessing(tmp_path, route, length):
    from tokenizers import processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    from sentence_transformers.base.modules import Transformer
    from sentence_transformers.sentence_transformer.modules import Pooling

    source = tmp_path / "bert"
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "cat": 4, "dog": 5}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
    fast.save_pretrained(source)
    BertModel(
        BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).save_pretrained(source)
    backbone = Transformer(str(source), query_length=4, document_length=7)
    router = Router.for_query_document([backbone, Pooling(8)], [backbone, Pooling(8)])
    model = SentenceTransformer(modules=[router, Normalize()], device="cpu")
    texts = ["cat dog cat dog cat dog cat dog"]
    method = f"encode_{route}"
    expected = getattr(model, method)(texts)
    destination = tmp_path / "export"
    model.save_route(str(destination), route, create_model_card=False)
    loaded = SentenceTransformer(str(destination), device="cpu", local_files_only=True)
    assert loaded.preprocess(texts, task=route)["input_ids"].shape[1] == length
    np.testing.assert_allclose(getattr(loaded, method)(texts), expected, rtol=1e-6, atol=1e-7)
