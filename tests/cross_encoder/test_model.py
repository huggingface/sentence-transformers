from __future__ import annotations

import functools
import json
import logging
import pickle
import re
import sys
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from packaging.version import Version, parse
from PIL import Image
from pytest import FixtureRequest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from transformers import __version__ as transformers_version

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util.decorators import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from tests.utils import GENERIC_INSTRUCT_TEMPLATE, RERANKER_TEMPLATE, skip_bfloat16_cpu_crash


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"config_kwargs": None},
        {"config_kwargs": {"classifier_dropout": 0.1}},
        {"config_args": {"classifier_dropout": 0.1}},
    ],
)
def test_classifier_dropout_is_set(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, kwargs: dict) -> None:
    reranker_bert_tiny_model.save_pretrained(tmp_path)
    original_kwargs = deepcopy(kwargs)
    model = CrossEncoder(str(tmp_path), classifier_dropout=0.1234, local_files_only=True, device="cpu", **kwargs)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.dropout.p == 0.1234
    assert kwargs == original_kwargs


def test_classifier_dropout_default_value(reranker_bert_tiny_model: CrossEncoder) -> None:
    model = reranker_bert_tiny_model
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None


def test_init_does_not_mutate_kwargs(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path) -> None:
    reranker_bert_tiny_model.save_pretrained(tmp_path)
    config_kwargs = {"output_attentions": False, "num_labels": 2}
    processor_kwargs = {"model_max_length": 64}
    model = CrossEncoder(
        str(tmp_path),
        num_labels=1,
        max_length=32,
        config_kwargs=config_kwargs,
        processor_kwargs=processor_kwargs,
        local_files_only=True,
        device="cpu",
    )
    assert model.num_labels == 1
    assert model.max_seq_length == 32
    assert config_kwargs == {"output_attentions": False, "num_labels": 2}
    assert processor_kwargs == {"model_max_length": 64}
    original_model = CrossEncoder(
        str(tmp_path), processor_kwargs=processor_kwargs, local_files_only=True, device="cpu"
    )
    assert original_model.max_seq_length == 64


def test_load_with_revision() -> None:
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    main_model = CrossEncoder(model_name, num_labels=1, revision="main")
    latest_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="f3cb857cba53019a20df283396bcca179cf051a4",
    )
    older_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24",
    )

    # Set the classifier.bias and classifier.weight equal among models. This
    # is needed because the AutoModelForSequenceClassification randomly initializes
    # the classifier.bias and classifier.weight for each (model) initialization.
    # The test is only possible if all models have the same classifier.bias
    # and classifier.weight parameters.
    latest_model.model.classifier.bias = main_model.model.classifier.bias
    latest_model.model.classifier.weight = main_model.model.classifier.weight
    older_model.model.classifier.bias = main_model.model.classifier.bias
    older_model.model.classifier.weight = main_model.model.classifier.weight

    test_sentences = [["Hello there!", "Hello, World!"]]
    main_prob = main_model.predict(test_sentences, convert_to_tensor=True)
    assert torch.equal(main_prob, latest_model.predict(test_sentences, convert_to_tensor=True))
    assert not torch.equal(main_prob, older_model.predict(test_sentences, convert_to_tensor=True))


@pytest.mark.parametrize(
    argnames="return_documents",
    argvalues=[True, False],
    ids=["return-docs", "no-return-docs"],
)
def test_rank(return_documents: bool, reranker_bert_tiny_model: CrossEncoder) -> None:
    model = reranker_bert_tiny_model
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    scores = model.predict([(query, document) for document in corpus])
    expected_ranking = [1, 0, 5, 4, 2, 6, 8, 7, 3]

    # 1. We rank all sentences in the corpus for the query
    ranks = model.rank(query=query, documents=corpus, return_documents=return_documents)
    if return_documents:
        assert [rank["text"] for rank in ranks] == [corpus[index] for index in expected_ranking]
    else:
        assert all("text" not in rank for rank in ranks)

    pred_ranking = [rank["corpus_id"] for rank in ranks]
    assert pred_ranking == expected_ranking
    np.testing.assert_allclose([rank["score"] for rank in ranks], scores[expected_ranking])


def test_rank_scores_are_floats(reranker_bert_tiny_model: CrossEncoder) -> None:
    ranks = reranker_bert_tiny_model.rank("A man is eating pasta.", ["A man is eating food.", "A monkey is playing."])
    assert [type(rank["score"]) for rank in ranks] == [float, float]
    json.dumps(ranks)


@pytest.mark.parametrize("kwarg", ["convert_to_numpy", "convert_to_tensor"])
def test_rank_convert_kwargs_deprecated(reranker_bert_tiny_model: CrossEncoder, kwarg: str, caplog) -> None:
    # Annotated as Any so type checkers don't match the removed kwarg against every remaining parameter
    deprecated_kwargs: dict[str, Any] = {kwarg: True}
    with caplog.at_level(logging.WARNING):
        ranks = reranker_bert_tiny_model.rank("A man is eating pasta.", ["A man is eating food."], **deprecated_kwargs)

    assert kwarg in caplog.text
    assert type(ranks[0]["score"]) is float


@pytest.mark.parametrize("apply_softmax", [True, False])
def test_rank_apply_softmax_deprecated(reranker_bert_tiny_model: CrossEncoder, apply_softmax: bool, caplog) -> None:
    model = reranker_bert_tiny_model
    query = "How many people live in Berlin?"
    documents = ["Pasta is cooked in boiling water.", "Berlin has a population of 3.5 million."]
    expected = model.rank(query, documents)
    deprecated_kwargs: dict[str, Any] = {"apply_softmax": apply_softmax}
    with caplog.at_level(logging.WARNING):
        ranks = model.rank(query, documents, **deprecated_kwargs)

    assert "CrossEncoder.rank `apply_softmax` argument is deprecated" in caplog.text
    assert ranks == expected


def test_rank_multiple_labels(multiclass_bert_tiny_model: CrossEncoder):
    model = multiclass_bert_tiny_model
    with pytest.raises(
        ValueError,
        match=re.escape(
            "CrossEncoder.rank() only works for models with num_labels=1. "
            "Consider using CrossEncoder.predict() with input pairs instead."
        ),
    ):
        model.rank(
            query="A man is eating pasta.",
            documents=[
                "A man is eating food.",
                "A man is eating a piece of bread.",
                "The girl is carrying a baby.",
            ],
        )


def test_predict_softmax(multiclass_bert_tiny_model: CrossEncoder, caplog):
    model = multiclass_bert_tiny_model
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
    ]
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=True, convert_to_tensor=True)
    assert torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=False, convert_to_tensor=True)
    assert not torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()
    assert "`apply_softmax` argument is deprecated" not in caplog.text


def test_predict_softmax_single_label(reranker_bert_tiny_model: CrossEncoder):
    model = reranker_bert_tiny_model
    query = "How many people live in Berlin?"
    corpus = [
        "Berlin has a population of 3.5 million.",
        "Pasta is cooked in boiling water.",
        "The capital of France is Paris.",
    ]
    scores = model.predict([(query, doc) for doc in corpus], convert_to_tensor=True)
    softmax_scores = model.predict([(query, doc) for doc in corpus], apply_softmax=True, convert_to_tensor=True)
    assert torch.allclose(softmax_scores, scores)


@skip_bfloat16_cpu_crash
def test_predict_low_precision_logits_upcast_to_float32(reranker_bert_tiny_model: CrossEncoder) -> None:
    """The activation over cross-encoder logits must run in float32 for low-precision models (HPS).

    Applying sigmoid/softmax to float16/bfloat16 logits buckets the (0, 1) probability range
    coarsely, collapsing distinct relevance scores into spurious ties.
    """
    model = reranker_bert_tiny_model
    if model.device.type == "cpu" and Version(torch.__version__) <= Version("2.5.0"):
        pytest.xfail("bfloat16 CPU inference is unreliable on older torch")
    pairs = [
        ("A man is eating pasta.", "A man is eating food."),
        ("A man is eating pasta.", "The girl is carrying a baby."),
    ]
    reference = model.predict(pairs, convert_to_tensor=True)
    assert reference.dtype == torch.float32

    model.model.to(torch.bfloat16)
    scores = model.predict(pairs, convert_to_tensor=True)

    assert scores.dtype == torch.float32
    assert torch.allclose(scores, reference, atol=1e-2)


@pytest.mark.parametrize("model_fixture", ["reranker_bert_tiny_model", "multiclass_bert_tiny_model"])
def test_predict_single_input(model_fixture: str, request: FixtureRequest):
    model = request.getfixturevalue(model_fixture)
    nested_pair_score = model.predict([["A man is eating pasta.", "A man is eating food."]])
    assert isinstance(nested_pair_score, np.ndarray)
    if model.num_labels == 1:
        assert nested_pair_score.shape == (1,)
    else:
        assert nested_pair_score.shape == (1, model.num_labels)

    pair_score = model.predict(["A man is eating pasta.", "A man is eating food."])
    if model.num_labels == 1:
        assert isinstance(pair_score, np.float32)
    else:
        assert isinstance(pair_score, np.ndarray)
        assert pair_score.shape == (model.num_labels,)


def test_is_singular_input_numpy_1d_pair(reranker_bert_tiny_model: CrossEncoder) -> None:
    """A 1D numpy string array represents a single (query, document) pair."""
    assert reranker_bert_tiny_model.is_singular_input(np.array(["query", "document"])) is True


def test_is_singular_input_numpy_2d_pairs(reranker_bert_tiny_model: CrossEncoder) -> None:
    """A 2D numpy string array is a batch of pairs."""
    assert reranker_bert_tiny_model.is_singular_input(np.array([["q1", "d1"], ["q2", "d2"]])) is False


def test_is_singular_input_numpy_empty(reranker_bert_tiny_model: CrossEncoder) -> None:
    """An empty 1D string ndarray is an empty batch, not a singular pair, matching ``predict([])``."""
    assert reranker_bert_tiny_model.is_singular_input(np.array([], dtype=str)) is False


def test_predict_numpy_empty(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Predicting on an empty string ndarray should return an empty array, like ``predict([])``."""
    scores = reranker_bert_tiny_model.predict(np.array([], dtype=str), show_progress_bar=False)
    expected = reranker_bert_tiny_model.predict([], show_progress_bar=False)
    assert scores.shape == (0,)
    assert np.array_equal(scores, expected)


def test_predict_numpy_1d_pair(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Predicting on a 1D numpy string array (a single pair) should match the tuple equivalent
    and return a scalar score. Exercises the singular-branch .tolist() conversion."""
    model = reranker_bert_tiny_model
    pair = np.array(["what is AI?", "AI is artificial intelligence."])
    score = model.predict(pair, show_progress_bar=False)
    expected = model.predict(tuple(pair.tolist()), show_progress_bar=False)
    assert isinstance(score, np.float32)
    assert np.allclose(score, expected)


def test_predict_numpy_2d_pairs(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Predicting on a 2D numpy string array should match predicting on the equivalent nested list."""
    pairs = np.array([["what is AI?", "AI is artificial intelligence."], ["what is ML?", "ML is machine learning."]])
    scores = reranker_bert_tiny_model.predict(pairs, show_progress_bar=False)
    expected = reranker_bert_tiny_model.predict(pairs.tolist(), show_progress_bar=False)
    assert scores.shape == (2,)
    assert np.allclose(scores, expected)


def test_predict_batch_size_1(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Regression test: batch_size=1 with num_labels=1 used to fail because squeeze produced a 0-d tensor.

    Some models (e.g. jinaai/jina-reranker-m0) return scores with shape [batch_size] instead of [batch_size, 1].
    With batch_size=1, squeeze(-1) would collapse [1] to a 0-d scalar, causing .extend() to fail.
    We mock forward to reproduce this by stripping the trailing dimension.
    """
    model = reranker_bert_tiny_model
    pairs = [
        ["A man is eating pasta.", "A man is eating food."],
        ["The girl is carrying a baby.", "A man is riding a horse."],
    ]

    original_forward = model.forward

    def forward_without_trailing_dim(features, **kwargs):
        out = original_forward(features, **kwargs)
        # Simulate models that return [batch_size] instead of [batch_size, 1]
        out["scores"] = out["scores"].squeeze(-1)
        return out

    model.forward = forward_without_trailing_dim

    scores = model.predict(pairs, batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)


@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
def test_empty_predict(reranker_bert_tiny_model: CrossEncoder, convert_to_numpy: bool, convert_to_tensor: bool):
    model = reranker_bert_tiny_model
    result = model.predict([], convert_to_numpy=convert_to_numpy, convert_to_tensor=convert_to_tensor)

    if convert_to_tensor:
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 0
        assert result.device == model.model.device
    elif convert_to_numpy:
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    else:
        assert result == []


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
def test_predict_output_types(
    reranker_bert_tiny_model: CrossEncoder, convert_to_tensor: bool, convert_to_numpy: bool
) -> None:
    model = reranker_bert_tiny_model
    embeddings = model.predict(
        [["One sentence", "Another sentence"]],
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )
    if convert_to_tensor:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, torch.Tensor)
    elif convert_to_numpy:
        assert embeddings[0].dtype == np.float32
        assert isinstance(embeddings, np.ndarray)
    else:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, list)


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(reranker_bert_tiny_model: CrossEncoder, safe_serialization: bool) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        model = reranker_bert_tiny_model
        if safe_serialization:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save_pretrained(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            # For transformers v5.0, safe_serialization is quietly ignored
            if parse(transformers_version) < Version("5.0.0dev0"):
                model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
                model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
                assert 1 == len(model_files)


def test_bfloat16() -> None:
    if sys.platform == "win32" and not torch.cuda.is_available():
        pytest.skip(
            "bfloat16 CPU matmul can hard-crash (0xc000001d) on some Windows machines. Skipping to avoid CI failures."
        )
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", automodel_args={"torch_dtype": torch.bfloat16}
    )
    score = model.predict([["Hello there!", "Hello, World!"]])
    assert isinstance(score, np.ndarray)

    ranking = model.rank("Hello there!", ["Hello, World!", "Heya!"])
    assert isinstance(ranking, list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_assignment(device):
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device=device)
    assert model.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_device_switching():
    # test assignment using .to
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"
    assert model.model.device.type == "cpu"

    model.to("cuda")
    assert model.device.type == "cuda"
    assert model.model.device.type == "cuda"

    del model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_target_device_backwards_compat():
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"

    assert model._target_device.type == "cpu"
    model._target_device = "cuda"
    assert model.device.type == "cuda"


def test_num_labels_fresh_model():
    model = CrossEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert model.num_labels == 1


# The transformer-less stacks below aren't runnable end-to-end (no non-Transformer InputModule
# currently preprocesses CrossEncoder pairs), but we technically allow them so the property logic
# (config/model/num_labels/activation_fn) is exercised in isolation. It's what makes stacks like
# Transformer + Pooling + Dense(scores) usable as rerankers.
@pytest.mark.parametrize("out_features", [1, 3, 7])
def test_num_labels_from_dense_scores_module(static_embedding: StaticEmbedding, out_features: int):
    """A Dense module with ``module_output_name="scores"`` should dictate ``num_labels``."""
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        out_features,
        activation_function=None,
        module_output_name="scores",
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    assert model.num_labels == out_features


def test_num_labels_ignores_dense_without_scores_output(static_embedding: StaticEmbedding):
    """A trailing Dense whose ``module_output_name`` is not ``"scores"`` must not be
    treated as the classification head by ``num_labels``.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        64,
        activation_function=None,
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    # No Transformer/LogitScore in the stack, and the Dense doesn't expose "scores",
    # so num_labels falls through to the default of 1 rather than picking up Dense.out_features.
    assert model.num_labels == 1


def test_config_and_model_none_without_transformer(static_embedding: StaticEmbedding):
    """``.config`` and ``.model`` must degrade to ``None`` when the module stack has no
    underlying HuggingFace Transformer rather than raising AttributeError on ``self[0].model``.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(static_embedding.get_embedding_dimension(), 1, module_output_name="scores")
    model = CrossEncoder(modules=[static_embedding, dense])

    assert model.config is None
    assert model.model is None


@pytest.mark.parametrize(
    ["out_features", "expected_activation_cls"],
    [(1, torch.nn.Sigmoid), (3, torch.nn.Identity)],
)
def test_default_activation_fn_without_transformer(
    static_embedding: StaticEmbedding, out_features: int, expected_activation_cls: type
):
    """``get_default_activation_fn`` must pick Sigmoid/Identity based on ``num_labels`` even
    when there's no Transformer (and thus no ``config``) to inspect for a saved activation.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        out_features,
        activation_function=None,
        module_output_name="scores",
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    assert isinstance(model.activation_fn, expected_activation_cls)


def test_push_to_hub(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    model = reranker_bert_tiny_model

    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs
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
            # Required as of https://github.com/huggingface/huggingface_hub/pull/3679
            return CommitInfo(**commit_info_kwargs, _endpoint=None)

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", revision="revision_test")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert mock_upload_folder_kwargs["revision"] == "revision_test"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/678901"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", create_pr=True)
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/discussions/123"
    mock_upload_folder_kwargs.clear()


@pytest.mark.parametrize(
    ["in_args", "in_kwargs", "out_args", "out_kwargs"],
    [
        [
            tuple(),
            {"model_name": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", "classifier_dropout": 0.1234},
            tuple(),
            {
                "model_name_or_path": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "config_kwargs": {"classifier_dropout": 0.1234},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {"classifier_dropout": 0.1234},
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {"config_kwargs": {"classifier_dropout": 0.1234}},
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "automodel_args": {"foo": "bar"},
                "tokenizer_args": {"foo": "baz"},
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "model_kwargs": {"foo": "bar"},
                "processor_kwargs": {"foo": "baz"},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "config_args": {"foo": "bar"},
                "cache_dir": "local_tmp",
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "config_kwargs": {"foo": "bar"},
                "cache_folder": "local_tmp",
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "automodel_args": {"foo": "bar"},
                "model_kwargs": {"faa": "baz"},
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "model_kwargs": {"faa": "baz"},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "default_activation_function": "torch.nn.Sigmoid",
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "activation_fn": "torch.nn.Sigmoid",
            },
        ],
        [tuple(), {}, tuple(), {}],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {},
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {},
        ],
        [
            tuple(),
            {
                "model_name": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "automodel_args": {"foo": "bar"},
                "tokenizer_args": {"foo": "baz"},
                "config_args": {"foo": "bar"},
                "cache_dir": "local_tmp",
            },
            tuple(),
            {
                "model_name_or_path": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "model_kwargs": {"foo": "bar"},
                "processor_kwargs": {"foo": "baz"},
                "config_kwargs": {"foo": "bar"},
                "cache_folder": "local_tmp",
            },
        ],
    ],
)
def test_init_args_decorator(
    monkeypatch: pytest.MonkeyPatch, in_args: tuple, in_kwargs: dict, out_args: tuple, out_kwargs: dict
):
    decorated_out_args = None
    decorated_out_kwargs = None

    @cross_encoder_init_args_decorator
    def mock_init(self, *args, **kwargs):
        nonlocal decorated_out_args
        nonlocal decorated_out_kwargs
        decorated_out_args = args
        decorated_out_kwargs = kwargs
        return None

    monkeypatch.setattr(CrossEncoder, "__init__", mock_init)

    CrossEncoder(*in_args, **in_kwargs)
    assert decorated_out_args == out_args
    assert decorated_out_kwargs == out_kwargs


@pytest.mark.parametrize(
    ["in_kwargs", "out_kwargs"],
    [
        [
            {"inputs": [["Hello there!", "Hello, World!"]], "num_workers": 2},
            {"inputs": [["Hello there!", "Hello, World!"]]},
        ],
        [
            {
                "inputs": [["Hello there!", "Hello, World!"]],
                "activation_fct": torch.nn.Identity,
                "activation_fn": torch.nn.Sigmoid,
            },
            {"inputs": [["Hello there!", "Hello, World!"]], "activation_fn": torch.nn.Sigmoid},
        ],
        [
            {"sentences": [["Hello there!", "Hello, World!"]]},
            {"inputs": [["Hello there!", "Hello, World!"]]},
        ],
    ],
)
def test_predict_rank_args_decorator(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch, caplog, in_kwargs: dict, out_kwargs: dict
):
    model = reranker_bert_tiny_model
    decorated_out_kwargs = None

    @cross_encoder_predict_rank_args_decorator
    def mock_predict(self, *args, **kwargs):
        nonlocal decorated_out_kwargs
        decorated_out_kwargs = kwargs
        return None

    monkeypatch.setattr(CrossEncoder, "predict", mock_predict)

    with caplog.at_level(logging.WARNING):
        model.predict(**in_kwargs)
        assert caplog.text != ""
    assert decorated_out_kwargs == out_kwargs


def test_logger_warning(caplog):
    model_name = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"
    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, classifier_dropout=0.1234)
        assert "`classifier_dropout` argument is deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, automodel_args={"torch_dtype": torch.float32})
        assert "`automodel_args` argument was renamed and is now deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, tokenizer_args={"model_max_length": 8192})
        assert "`tokenizer_args` argument was renamed and is now deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, config_args={"classifier_dropout": 0.2})
        assert "`config_args` argument was renamed and is now deprecated" in caplog.text


def test_predict_with_falsey_activation_override(reranker_bert_tiny_model: CrossEncoder):
    model = reranker_bert_tiny_model
    inputs = [["Hello there!", "Hello, World!"], ["A cat is sleeping.", "An animal is resting."]]
    raw_scores = model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True)

    torch.testing.assert_close(
        model.predict(inputs, activation_fn=torch.nn.Sequential(), convert_to_tensor=True), raw_scores
    )
    torch.testing.assert_close(model.predict(inputs, convert_to_tensor=True), torch.sigmoid(raw_scores))


def test_explicit_activation_overrides_unconstructible_saved_activation(
    reranker_bert_tiny_model: CrossEncoder, tmp_path: Path
):
    model = reranker_bert_tiny_model
    model.activation_fn = torch.nn.Threshold(0.0, -0.5)
    inputs = [["Hello there!", "Hello, World!"], ["A cat is sleeping.", "An animal is resting."]]
    expected = model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True)
    model.save_pretrained(tmp_path)

    restored = CrossEncoder(str(tmp_path), activation_fn=torch.nn.Identity(), local_files_only=True)

    torch.testing.assert_close(restored.predict(inputs, convert_to_tensor=True), expected)


@pytest.mark.parametrize(
    ["num_labels", "activation_fn", "expected_activation_fn"],
    [
        [1, torch.nn.Tanh(), torch.nn.Tanh()],
        [1, None, torch.nn.Sigmoid()],
        [3, None, torch.nn.Identity()],
    ],
)
def test_load_activation_fn_from_kwargs(
    num_labels: int,
    activation_fn: torch.nn.Module | None,
    expected_activation_fn: torch.nn.Module,
    tmp_path: Path,
):
    model = CrossEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", num_labels=num_labels, activation_fn=activation_fn
    )
    inputs = [["Hello there!", "Hello, World!"], ["A cat is sleeping.", "An animal is resting."]]
    raw_scores = model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True)
    expected_scores = expected_activation_fn(raw_scores.reshape(len(inputs), num_labels))
    torch.testing.assert_close(
        model.predict(inputs, convert_to_tensor=True).reshape_as(expected_scores), expected_scores
    )

    model.save_pretrained(tmp_path)
    loaded_model = CrossEncoder(str(tmp_path), local_files_only=True)
    torch.testing.assert_close(
        loaded_model.predict(inputs, convert_to_tensor=True).reshape_as(expected_scores), expected_scores
    )

    # A per-call override must not change subsequent predictions.
    torch.testing.assert_close(
        loaded_model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True), raw_scores
    )
    torch.testing.assert_close(
        loaded_model.predict(inputs, convert_to_tensor=True).reshape_as(expected_scores), expected_scores
    )

    loaded_model = CrossEncoder(str(tmp_path), activation_fn=torch.nn.Identity(), local_files_only=True)
    torch.testing.assert_close(loaded_model.predict(inputs, convert_to_tensor=True), raw_scores)


@pytest.mark.parametrize(
    "tanh_model_name",
    [
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v3",
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4",
    ],
)
def test_load_activation_fn_from_config(tanh_model_name: str, tmp_path):
    model = CrossEncoder(tanh_model_name)
    inputs = [["Hello there!", "Hello, World!"], ["A cat is sleeping.", "An animal is resting."]]
    raw_scores = model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True)
    expected_scores = torch.tanh(raw_scores)
    torch.testing.assert_close(model.predict(inputs, convert_to_tensor=True), expected_scores)

    model.save_pretrained(tmp_path)
    loaded_model = CrossEncoder(str(tmp_path), local_files_only=True)
    torch.testing.assert_close(loaded_model.predict(inputs, convert_to_tensor=True), expected_scores)


def test_load_activation_fn_from_config_custom(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path):
    model = reranker_bert_tiny_model
    inputs = [["Hello there!", "Hello, World!"], ["A cat is sleeping.", "An animal is resting."]]
    raw_scores = model.predict(inputs, activation_fn=torch.nn.Identity(), convert_to_tensor=True)

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config_sentence_transformers.json") as f:
        config = json.load(f)
    config["activation_fn"] = "sentence_transformers.custom.activations.CustomActivation"
    with open(tmp_path / "config_sentence_transformers.json", "w") as f:
        json.dump(config, f)

    loaded_model = CrossEncoder(str(tmp_path), local_files_only=True)
    torch.testing.assert_close(loaded_model.predict(inputs, convert_to_tensor=True), torch.sigmoid(raw_scores))

    with pytest.raises(ModuleNotFoundError):
        CrossEncoder(str(tmp_path), trust_remote_code=True, local_files_only=True)

    loaded_model = CrossEncoder(
        str(tmp_path), activation_fn=torch.nn.Identity(), trust_remote_code=True, local_files_only=True
    )
    torch.testing.assert_close(loaded_model.predict(inputs, convert_to_tensor=True), raw_scores)


@pytest.mark.parametrize(
    "activation_fn",
    [torch.relu, lambda scores: scores, functools.partial(torch.clamp, min=0.0), torch.sigmoid],
    ids=["builtin", "lambda", "partial", "default"],
)
def test_save_warns_for_plain_function_activation_fn(
    reranker_bert_tiny_model: CrossEncoder,
    tmp_path: Path,
    activation_fn: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = reranker_bert_tiny_model
    model.activation_fn = activation_fn
    inputs = [["Hello there!", "Hello, World!"]]
    expected = model.predict(inputs, convert_to_tensor=True)
    with caplog.at_level(logging.WARNING, logger="sentence_transformers.cross_encoder.model"):
        model.save_pretrained(tmp_path)
    assert "plain functions can't be restored" in caplog.text

    loaded = CrossEncoder(str(tmp_path), activation_fn=activation_fn, local_files_only=True)
    torch.testing.assert_close(loaded.predict(inputs, convert_to_tensor=True), expected)


def test_default_activation_fn(reranker_bert_tiny_model: CrossEncoder):
    scores = torch.tensor([-2.0, 0.0, 2.0])
    with pytest.warns(DeprecationWarning):
        activation = reranker_bert_tiny_model.default_activation_function
    torch.testing.assert_close(activation(scores), torch.sigmoid(scores))


@pytest.mark.parametrize("tokenizer_max_length, expected_max_length", [(256, 256), (512, 512), (1024, 512)])
def test_max_length(
    reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, tokenizer_max_length: int, expected_max_length: int
) -> None:
    reranker_bert_tiny_model.model.save_pretrained(tmp_path)
    reranker_bert_tiny_model.tokenizer.model_max_length = tokenizer_max_length
    reranker_bert_tiny_model.tokenizer.save_pretrained(tmp_path)

    model = CrossEncoder(str(tmp_path), local_files_only=True)
    assert model.max_seq_length == expected_max_length
    assert model.tokenizer.model_max_length == expected_max_length
    with pytest.warns(DeprecationWarning, match="max_length"):
        assert model.max_length == expected_max_length

    with pytest.warns(DeprecationWarning, match="max_length"):
        model.max_length = 128
    with pytest.warns(DeprecationWarning, match="max_length"):
        assert model.max_length == 128
    assert model.max_seq_length == 128
    assert model.tokenizer.model_max_length == 128


def test_predict_with_dataset_column(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Test that predict can handle a dataset column as input."""
    model = reranker_bert_tiny_model
    from datasets import Dataset

    # Create a simple dataset with a text column
    dataset = Dataset.from_dict(
        {
            "text": [
                ["This is the start of a pair.", "And this the end."],
                ["This is a second pair.", "And this the end of the second pair."],
            ]
        }
    )

    # Encode the dataset column
    embeddings = model.predict(dataset["text"], convert_to_tensor=True)

    # Check the shape of the embeddings
    assert embeddings.shape == (2,)


def test_predict_per_call_processing_kwargs(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Per-call ``processing_kwargs`` should be accepted by ``predict`` and reach ``preprocess``.

    If the kwarg were silently dropped, truncated and full predictions on the same pair would
    produce identical scores.
    """
    model = reranker_bert_tiny_model
    pair = ["this is a moderately long query sentence", "this is a moderately long candidate sentence"]
    truncated = model.predict(
        [pair],
        processing_kwargs={"text": {"max_length": 4, "truncation": True}},
    )
    full = model.predict([pair])
    assert not np.isclose(truncated[0], full[0])


# ChatML emits the role rather than branching on it, so unknown roles carry their content through
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
)
# Structured-content variant of a rerank template, as a multimodal processor would need
STRUCTURED_RERANKER_TEMPLATE = (
    '<Query>: {{ (messages | selectattr("role", "eq", "query") | first).content[0].text }}\n'
    '<Document>: {{ (messages | selectattr("role", "eq", "document") | first).content[0].text }}'
)
BERLIN_PAIRS = [
    ["How many people live in Berlin?", "Berlin has a population of 3,520,031 registered inhabitants."],
    ["How many people live in Berlin?", "Berlin is well known for its museums."],
]
PAIR_ROLES_DROPPED = "cannot carry a 'query'/'document' pair"


def test_replacing_a_rejected_chat_template_takes_effect() -> None:
    """The error names a remedy, so applying it to a loaded model has to work."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model.predict(BERLIN_PAIRS)

    model.processor.chat_template = RERANKER_TEMPLATE
    decoded = model[0].tokenizer.decode(
        model[0].preprocess(BERLIN_PAIRS[:1])["input_ids"][0], skip_special_tokens=True
    )
    assert "< query > : how many people live in berlin?" in decoded


def test_preprocess_raises_on_non_text_pair_without_pair_roles() -> None:
    """Non-text pairs reach the roles by a different route, and are dropped just the same."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess([(image, "a document")])
    # A text pair sharing a batch with a non-text one takes the same route
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess([(image, "a document"), ("a query", "a document")])


def test_predict_raises_on_chat_template_without_pair_roles() -> None:
    """A chat template that drops the pair roles must be reported, not worked around.

    Backbones such as the Qwen3 instruct models carry a template that only branches on
    ``system``/``user``/``assistant``. Rendering the ``query``/``document`` roles through it drops both
    messages, leaving the model with a zero-length input. Tokenizing the pair directly instead would
    score an input the model was never trained on, so the misconfiguration is raised rather than hidden.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    transformer = model[0]
    # The template is picked up, it just cannot render the roles that text pairs take
    assert "message" in transformer.modality_config
    assert transformer.input_formatter.pair_roles_failure() is not None

    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model.predict(BERLIN_PAIRS)

    # Single texts never take those roles, so they are unaffected
    features = transformer.preprocess(["a solo text"])
    assert "a solo text" in transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)


def test_predict_raises_on_mixed_batch_containing_a_pair() -> None:
    """One pair in a batch is enough, its content would be dropped just the same."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess(["a solo text", ("How many people live in Berlin?", "Berlin has 3.5M.")])


def test_predict_applies_chat_template_with_pair_roles() -> None:
    """A template that does reference the ``query``/``document`` roles must still be applied."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": RERANKER_TEMPLATE},
    )
    transformer = model[0]
    assert transformer.input_formatter.pair_roles_failure() is None

    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "< query > : how many people live in berlin?" in decoded
    assert "< document > : berlin has a population" in decoded


def test_predict_applies_chat_template_that_only_echoes_the_role() -> None:
    """A template that carries unknown roles through must keep working, not raise.

    Pass-through ChatML templates such as SmolLM2's emit ``message['role']`` verbatim, so they never
    mention ``query``/``document`` yet render both messages in full. Deciding from the template source
    rather than from a render would break models that are fine today.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": CHATML_TEMPLATE},
    )
    transformer = model[0]
    assert transformer.input_formatter.pair_roles_failure() is None

    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "query" in decoded and "how many people live in berlin?" in decoded
    assert "document" in decoded and "berlin has a population" in decoded


def test_message_input_with_a_single_pair_role_is_not_checked() -> None:
    """A hand-written message may use one pair role alone, and that is not a pair.

    The check guards the pair mapping, which always produces both roles in one conversation, so a
    template rendering just the role the user actually sent must not be rejected for dropping a role
    the batch never uses.
    """
    query_only = "{% for m in messages %}{% if m.role == 'query' %}Instruct: {{ m.content }}{% endif %}{% endfor %}"
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": query_only},
    )
    features = model[0].preprocess([[{"role": "query", "content": "hello world"}]])
    decoded = model[0].tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "instruct : hello world" in decoded


def test_probe_uses_the_configured_chat_template_kwargs() -> None:
    """A model may select a named rerank template via processing kwargs, as Qwen3-VL-Reranker does.

    The probe must measure the selected template rather than the default one, since rejecting the
    model for a default that drops the pair roles would break a correctly configured reranker.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={
            "chat_template": {"default": GENERIC_INSTRUCT_TEMPLATE, "reranker": STRUCTURED_RERANKER_TEMPLATE}
        },
    )
    transformer = model[0]
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED) as excinfo:
        transformer.preprocess(BERLIN_PAIRS[:1])
    # The dict template infers structured format, so the remedy example must read typed content
    assert "content[0].text" in str(excinfo.value)

    transformer.processing_kwargs["chat_template"] = {"chat_template": "reranker"}
    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "< query > : how many people live in berlin?" in decoded


# Test suite converted from demo_3406_simple_og.py
def format_queries(query, instruction=None):
    """Helper function to format queries with the template."""
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    """Helper function to format documents with the template."""
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_formatted_pairs():
    """Test Qwen3 Reranker with manually formatted query-document pairs."""
    model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", activation_fn=torch.nn.Identity())
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]

    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[format_queries(query, task), format_document(doc)] for query, doc in zip(queries, documents)]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_with_chat_template():
    """Test Qwen3 Reranker with Chat template."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    task = "Given a web search query, retrieve relevant passages that answer the query"
    model = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        activation_fn=torch.nn.Identity(),
        processor_kwargs={"chat_template": chat_template},
        prompts={"web_search": task},
        default_prompt_name="web_search",
    )

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_original_with_identity_activation():
    """Test original Qwen3 Reranker with Identity activation function."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    task = "Given a web search query, retrieve relevant passages that answer the query"
    model = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        prompts={"web_search": task},
        default_prompt_name="web_search",
        activation_fn=torch.nn.Identity(),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"chat_template": chat_template},
    )
    assert model.dtype == torch.float32

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


def test_conversion_ignores_source_prompts_and_default_prompt_name(tmp_path: Path) -> None:
    """Converting a SentenceTransformer save appends a fresh classification head, so the source's
    embedding prompts and default_prompt_name are not inherited: they would otherwise silently be
    prepended to every predict call."""
    from sentence_transformers import SentenceTransformer

    source = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "Represent this sentence: "},
        default_prompt_name="query",
    )
    source.save_pretrained(str(tmp_path))

    model = CrossEncoder(str(tmp_path))
    assert model.prompts == {}
    assert model.default_prompt_name is None


def test_predict_routes_through_module_call(reranker_bert_tiny_model: CrossEncoder) -> None:
    """predict() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = reranker_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.predict([["Hello there!", "Hello, World!"]])
    finally:
        handle.remove()
    assert calls, "predict() should invoke the model via __call__, not call forward() directly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_original_without_prompt():
    """Test original Qwen3 Reranker with Identity activation function."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    model = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        activation_fn=torch.nn.Identity(),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"chat_template": chat_template},
    )
    assert model.dtype == torch.float32

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


class TestKeepInFP32Modules:
    def test_saved_fp32_selection_and_overrides(self, tmp_path):
        head = Dense(8, 1, activation_function=None, module_output_name="scores")
        model = CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device="cpu")
        model.save_pretrained(str(tmp_path), create_model_card=False)

        config = json.loads((tmp_path / "config_sentence_transformers.json").read_text())
        assert config["keep_in_fp32_modules"] == ["0.linear"]

        restored = CrossEncoder(str(tmp_path), device="cpu").bfloat16()
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = restored({"sentence_embedding": x})["scores"]

        assert output.dtype == torch.float32
        expected = F.linear(x.float(), restored[0].linear.weight.float(), restored[0].linear.bias.float())
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        disabled = CrossEncoder(str(tmp_path), keep_in_fp32_modules=[], device="cpu").bfloat16()
        assert type(disabled[0].linear) is nn.Linear
        assert disabled({"sentence_embedding": x})["scores"].dtype == torch.bfloat16
        assert "keep_in_fp32_modules" not in disabled._get_model_config()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
    )
    def test_precision_gradients_and_parameter_identity(self, dtype, device):
        head = Dense(8, 1, activation_function=None).to(device=device, dtype=dtype)
        linear = head.linear
        parameters = list(head.parameters())
        keys = list(head.state_dict())
        optimizer = torch.optim.SGD(parameters, lr=0.1)

        observed = []
        handle = linear.register_forward_hook(lambda module, args, result: observed.append(result.dtype))

        model = CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device=device)
        model.to(dtype)

        assert head.linear is linear
        assert list(head.state_dict()) == keys
        assert all(before is after for before, after in zip(parameters, head.parameters()))
        assert optimizer.param_groups[0]["params"][0] is linear.weight
        assert linear.weight.dtype == dtype

        x = torch.randn(4, 8, device=device, dtype=dtype, requires_grad=True)
        with torch.autocast(device, dtype=dtype):
            result = head({"sentence_embedding": x})["sentence_embedding"]
            following = F.linear(x.float(), linear.weight.float(), linear.bias.float())

        expected = F.linear(x.float(), linear.weight.float(), linear.bias.float())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert result.dtype == torch.float32
        assert following.dtype == dtype
        assert observed == [torch.float32]

        actual_grads = torch.autograd.grad(result.sum(), (x, *parameters))
        expected_grads = torch.autograd.grad(expected.sum(), (x, *parameters))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.isfinite(actual).all()

        handle.remove()

    def test_selection_replacement_validation_and_defaults(self):
        head = Dense(8, 1, activation_function=None, use_residual=True).bfloat16()
        model = CrossEncoder(modules=[head], device="cpu")
        assert "keep_in_fp32_modules" not in model._get_model_config()

        model.set_keep_in_fp32_modules(["0.linear", "0.linear"])
        assert model.keep_in_fp32_modules == ["0.linear"]

        model.set_keep_in_fp32_modules(["0.residual"])
        assert type(head.linear) is nn.Linear

        with pytest.raises(AttributeError):
            model.set_keep_in_fp32_modules(["0.linear", "missing"])
        with pytest.raises(ValueError, match="plain nn.Linear"):
            model.set_keep_in_fp32_modules(["0.linear", "0"])

        assert type(head.linear) is nn.Linear
        assert model.keep_in_fp32_modules == ["0.residual"]

        x = torch.randn(3, 8, dtype=torch.bfloat16)
        assert head.residual(x).dtype == torch.float32

        model.set_keep_in_fp32_modules([])
        assert type(head.residual) is nn.Linear
        assert head.residual(x).dtype == torch.bfloat16

    @pytest.mark.parametrize("selection", ["0.linear", [None], {}, 0])
    def test_invalid_config_selection(self, selection):
        with pytest.raises(ValueError, match="list of module paths"):
            CrossEncoder(modules=[Dense(8, 1)], keep_in_fp32_modules=selection, device="cpu")

    def test_unsupported_modules_and_backends(self):
        class AdaptedLinear(nn.Linear):
            pass

        head = Dense(8, 1)
        head.linear = AdaptedLinear(8, 1)
        with pytest.raises(ValueError, match="plain nn.Linear"):
            CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device="cpu")

        head.linear = nn.Linear(8, 1)
        head.linear.forward = lambda x: x
        with pytest.raises(ValueError, match="custom forwards"):
            CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device="cpu")

        with pytest.raises(ValueError, match="torch backend"):
            CrossEncoder(modules=[Dense(8, 1)], keep_in_fp32_modules=["0.linear"], backend="onnx", device="cpu")

    def test_copy_pickle_and_compile(self):
        model = CrossEncoder(modules=[Dense(8, 1)], keep_in_fp32_modules=["0.linear"], device="cpu").bfloat16()
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model({"sentence_embedding": x})["sentence_embedding"]

            for restored in (deepcopy(model), pickle.loads(pickle.dumps(model))):
                torch.testing.assert_close(restored({"sentence_embedding": x})["sentence_embedding"], expected)

            compiled = torch.compile(model[0], backend="eager", fullgraph=True)
            torch.testing.assert_close(compiled({"sentence_embedding": x})["sentence_embedding"], expected)

    @pytest.mark.parametrize("offload", [False, True])
    @pytest.mark.parametrize("selected_before_dispatch", [False, True])
    def test_accelerate_hooks(self, offload, selected_before_dispatch):
        pytest.importorskip("accelerate")
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

        head = Dense(8, 1, activation_function=None).bfloat16()
        model = CrossEncoder(
            modules=[head], keep_in_fp32_modules=["0.linear"] if selected_before_dispatch else [], device="cpu"
        )
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = F.linear(x.float(), head.linear.weight.float(), head.linear.bias.float())
        hook = AlignDevicesHook(execution_device="cpu", offload=offload)
        add_hook_to_module(head.linear, hook)
        wrapped_forward = head.linear.forward

        for selection in (["0.linear"], [], ["0.linear"]):
            model.set_keep_in_fp32_modules(selection)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = head.linear(x)
            dtype = torch.float32 if selection else torch.bfloat16
            assert output.dtype == dtype
            torch.testing.assert_close(output, expected.to(dtype))
            assert head.linear.forward is wrapped_forward
            assert head.linear._hf_hook is hook
            assert head.linear.weight.device.type == ("meta" if offload else "cpu")

        remove_hook_from_module(head.linear)
        for selection in ([], ["0.linear"], []):
            model.set_keep_in_fp32_modules(selection)
            assert head.linear(x).dtype == (torch.float32 if selection else torch.bfloat16)

        add_hook_to_module(head.linear, AlignDevicesHook(execution_device="cpu", offload=offload))
        model.set_keep_in_fp32_modules(["0.linear"])
        assert head.linear(x).dtype == torch.float32
        remove_hook_from_module(head.linear)

    def test_accelerate_hook_preserves_custom_forward_rejection(self):
        pytest.importorskip("accelerate")
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

        head = Dense(8, 1)
        head.linear.forward = lambda x: x
        add_hook_to_module(head.linear, AlignDevicesHook(execution_device="cpu"))
        with pytest.raises(ValueError, match="custom forwards"):
            CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device="cpu")

    @skip_bfloat16_cpu_crash
    @pytest.mark.parametrize("architecture", ["classifier", "lm_head"])
    def test_transformers_heads_and_tied_embeddings(self, architecture):
        if architecture == "classifier":
            transformer = BertForSequenceClassification(
                BertConfig(
                    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
                )
            )
        else:
            transformer = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16))

        transformer.bfloat16().eval()
        head = transformer.get_submodule(architecture)
        original = head.weight

        model = CrossEncoder(modules=[transformer], keep_in_fp32_modules=[f"0.{architecture}"], device="cpu")
        ids = torch.randint(0, 32, (2, 8))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert model[0](ids).logits.dtype == torch.float32

        assert head.weight is original
        assert head.weight.dtype == torch.bfloat16

        if architecture == "lm_head":
            assert head.weight is transformer.get_input_embeddings().weight
            assert transformer.get_input_embeddings()(ids).dtype == torch.bfloat16

        model.set_keep_in_fp32_modules([])
        assert type(head) is nn.Linear
        assert model[0](ids).logits.dtype == torch.bfloat16

    def test_peft_modules_to_save_selection(self):
        pytest.importorskip("peft")
        from peft.utils import ModulesToSaveWrapper

        head = Dense(8, 1, activation_function=None).bfloat16()
        model = CrossEncoder(modules=[head], keep_in_fp32_modules=["0.linear"], device="cpu")
        original = head.linear
        wrapper = ModulesToSaveWrapper(original, "first")
        wrapper.update("second")
        head.linear = wrapper
        parameters = list(model.parameters())
        keys = list(model.state_dict())
        x = torch.randn(3, 8, dtype=torch.bfloat16)

        for selection in ([], ["0.linear"], []):
            model.set_keep_in_fp32_modules(selection)
            expected_dtype = torch.float32 if selection else torch.bfloat16
            assert head.linear is wrapper
            assert wrapper.original_module is original
            assert all(before is after for before, after in zip(parameters, model.parameters()))
            assert list(model.state_dict()) == keys

            with torch.autocast("cpu", dtype=torch.bfloat16):
                for adapter in ("first", "second"):
                    wrapper.set_adapter(adapter)
                    assert head({"sentence_embedding": x})["sentence_embedding"].dtype == expected_dtype

                wrapper.enable_adapters(False)
                assert head({"sentence_embedding": x})["sentence_embedding"].dtype == expected_dtype
                wrapper.enable_adapters(True)

    @pytest.mark.parametrize(
        "adapter_order, offload",
        [(None, False), ("before_selection", False), ("after_selection", False), (None, True)],
    )
    def test_cross_encoder_checkpoint(self, tmp_path, adapter_order, offload):
        if offload:
            pytest.importorskip("accelerate")
        if adapter_order:
            pytest.importorskip("peft")
            from peft import LoraConfig, TaskType

        checkpoint = tmp_path / "checkpoint"
        BertForSequenceClassification(
            BertConfig(vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
        ).save_pretrained(checkpoint)

        tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
        ).save_pretrained(checkpoint)

        model = CrossEncoder(
            str(checkpoint),
            keep_in_fp32_modules=[] if adapter_order == "before_selection" else ["0.model.classifier"],
            model_kwargs={"dtype": torch.bfloat16},
            device="cpu",
        )
        if adapter_order:
            model.add_adapter(
                LoraConfig(
                    target_modules=["query", "value"], task_type=TaskType.SEQ_CLS, r=2, modules_to_save=["classifier"]
                )
            )
            if adapter_order == "before_selection":
                model.set_keep_in_fp32_modules(["0.model.classifier"])

        pairs = [("a", "b"), ("b", "a")]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model.predict(pairs, convert_to_tensor=True)
            assert model.model(**model.preprocess(pairs)).logits.dtype == torch.float32

        assert model.model.classifier.weight.dtype == torch.bfloat16

        saved = tmp_path / "saved"
        model.save_pretrained(str(saved), create_model_card=False)

        model_kwargs = {"dtype": torch.bfloat16}
        if offload:
            model_kwargs.update(
                device_map={"bert": "cpu", "classifier": "disk"}, offload_folder=str(tmp_path / "offload")
            )
        restored = CrossEncoder(str(saved), model_kwargs=model_kwargs, device="cpu")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.testing.assert_close(restored.predict(pairs, convert_to_tensor=True), expected, rtol=0, atol=0)

        restored.set_keep_in_fp32_modules([])
        if adapter_order:
            classifier = restored.model.classifier
            assert type(classifier.original_module) is nn.Linear
            assert all(type(module) is nn.Linear for module in classifier.modules_to_save.values())
        else:
            assert type(restored.model.classifier) is nn.Linear
        if offload:
            assert restored.model.classifier.weight.device.type == "meta"
            assert restored.model(**restored.preprocess(pairs)).logits.dtype == torch.bfloat16
