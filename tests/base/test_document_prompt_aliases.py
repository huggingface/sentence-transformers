"""Offline regression tests using real model construction, loaders and prompt resolution."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from sentence_transformers import MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.sentence_transformer.modules import BoW

ENCODERS = (SentenceTransformer, SparseEncoder, MultiVectorEncoder)
CASES = [
    ({"passage": "p "}, None, {}, "p "),
    ({"corpus": "c "}, None, {}, "c "),
    ({"passage": "p ", "corpus": "c "}, None, {}, "p "),
    ({"document": "d ", "passage": "p "}, None, {}, "d "),
    ({"document": "", "passage": "p ", "corpus": "c "}, None, {}, ""),
    ({"document": "d "}, None, {"prompt": ""}, ""),
    ({"document": "d "}, None, {"prompt": "", "prompt_name": "document"}, ""),
    ({"query": "q "}, "query", {}, None),
    ({"passage": "", "corpus": "c "}, None, {}, ""),
    ({"passage": "p "}, "document", {}, "p "),
    ({"passage": "p "}, None, {"prompt_name": "document"}, None),
    ({"passage": "p "}, None, {"prompt": "custom "}, "custom "),
]


def make_model(cls, prompts=None, default_prompt_name=None):
    return cls(
        modules=[BoW(["p", "c", "d", "q"])],
        prompts=prompts,
        default_prompt_name=default_prompt_name,
        device="cpu",
    )


def resolved(model, **kwargs):
    def spy(**call):
        assert call["task"] == "document"
        return model._resolve_prompt(call.get("prompt"), call.get("prompt_name"))

    with patch.object(model, "encode", side_effect=spy):
        return model.encode_document(["d"], **kwargs)


@pytest.mark.parametrize("cls", ENCODERS)
@pytest.mark.parametrize("prompts,default,kwargs,expected", CASES)
def test_selection(cls, prompts, default, kwargs, expected):
    actual = resolved(make_model(cls, prompts, default), **kwargs)
    assert (actual or "") == (expected or "")


@pytest.mark.parametrize("cls", ENCODERS)
def test_empty_argument_forwarded_unchanged(cls):
    model = make_model(cls, {"document": "d "})
    with patch.object(model, "encode", return_value=None) as encode:
        model.encode_document(["d"], prompt="")
    assert encode.call_args.kwargs["prompt"] == ""
    assert encode.call_args.kwargs["prompt_name"] is None


@pytest.mark.parametrize("cls", ENCODERS)
@pytest.mark.parametrize("explicit_document", (False, True))
def test_save_load_and_alias_override(cls, explicit_document, tmp_path):
    prompts = {"passage": "p "}
    if explicit_document:
        prompts["document"] = ""
    model = make_model(cls, prompts)
    model.save(str(tmp_path), create_model_card=False)
    loaded = cls(str(tmp_path), local_files_only=True, device="cpu")
    assert (resolved(loaded) or "") == ("" if explicit_document else "p ")
    overridden = cls(str(tmp_path), prompts={"passage": "new "}, local_files_only=True, device="cpu")
    assert (resolved(overridden) or "") == ("" if explicit_document else "new ")


@pytest.mark.parametrize("cls", ENCODERS)
def test_runtime_explicit_empty(cls):
    model = make_model(cls, {"passage": "p "})
    model.prompts["document"] = ""
    assert resolved(model) == ""


def test_real_dense_encoding():
    model = make_model(SentenceTransformer, {"passage": "p "})
    actual = model.encode_document(["d"], convert_to_tensor=True)
    expected = model.encode(["d"], prompt="p ", convert_to_tensor=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cls", ENCODERS)
def test_no_prompts_model_card_and_fallback(cls):
    model = make_model(cls)
    assert model.prompts["document"] is None
    assert resolved(model) is None


@pytest.mark.parametrize("cls", ENCODERS)
def test_old_saved_empty_document_stays_authoritative(cls, tmp_path):
    model = make_model(cls, {"passage": "p "})
    model.save(str(tmp_path), create_model_card=False)
    config_path = next(tmp_path.glob("config_*encoder.json"), None)
    if config_path is None:
        config_path = tmp_path / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config["prompts"]["document"] = ""
    config_path.write_text(json.dumps(config))
    loaded = cls(str(tmp_path), local_files_only=True, device="cpu")
    assert resolved(loaded) == ""


@pytest.mark.parametrize("cls", ENCODERS)
def test_constructor_empty_document_overrides_saved_nonempty(cls, tmp_path):
    model = make_model(cls, {"document": "d ", "passage": "p "})
    model.save(str(tmp_path), create_model_card=False)
    loaded = cls(str(tmp_path), prompts={"document": ""}, local_files_only=True, device="cpu")
    assert resolved(loaded) == ""


def test_ir_warning_matches_selected_alias():
    from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
    from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
        InformationRetrievalEvaluator,
    )

    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries={"q": "q"}, corpus={"d": "d"}, relevant_docs={"q": {"d"}}, corpus_prompt="custom "
    )
    model = make_model(MultiVectorEncoder, {"passage": "p "})
    with (
        patch.object(InformationRetrievalEvaluator, "__call__", return_value={}),
        patch(
            "sentence_transformers.multi_vector_encoder.evaluation.information_retrieval.logger.warning_once"
        ) as warning,
    ):
        evaluator(model)
        assert "registered 'passage' prompt" in warning.call_args.args[0]
        warning.reset_mock()
        model.prompts["document"] = ""
        evaluator(model)
        warning.assert_not_called()
