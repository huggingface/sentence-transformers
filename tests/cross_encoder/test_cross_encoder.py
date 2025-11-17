from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from pytest import FixtureRequest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.util import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from sentence_transformers.util import fullname


def test_classifier_dropout_is_set() -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", classifier_dropout=0.1234)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.config.classifier_dropout == 0.1234


def test_classifier_dropout_default_value() -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None


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
def test_rank(return_documents: bool, request: FixtureRequest) -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
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
    expected_ranking = [0, 1, 3, 6, 2, 5, 7, 4, 8]

    # 1. We rank all sentences in the corpus for the query
    ranks = model.rank(query=query, documents=corpus, return_documents=return_documents)
    if request.node.callspec.id == "return-docs":
        assert {*corpus} == {rank.get("text") for rank in ranks}

    pred_ranking = [rank["corpus_id"] for rank in ranks]
    assert pred_ranking == expected_ranking


def test_rank_multiple_labels():
    model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
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


def test_predict_softmax():
    model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
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


@pytest.mark.parametrize(
    "model_name", ["cross-encoder-testing/reranker-bert-tiny-gooaq-bce", "cross-encoder/nli-MiniLM2-L6-H768"]
)
def test_predict_single_input(model_name: str):
    model = CrossEncoder(model_name)
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
    convert_to_tensor: bool,
    convert_to_numpy: bool,
) -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
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
def test_safe_serialization(safe_serialization: bool) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
        if safe_serialization:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save_pretrained(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            assert 1 == len(model_files)


def test_bfloat16() -> None:
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
    model = CrossEncoder("prajjwal1/bert-tiny")
    assert model.num_labels == 1


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
        if kwargs.get("revision") is None:
            revision = "123456"
        else:
            revision = "678901"
        return CommitInfo(
            commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/{revision}",
            commit_message="commit_message",
            commit_description="commit_description",
            oid="oid",
            pr_url=f"https://huggingface.co/{kwargs.get('repo_id')}/discussions/123",
        )

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

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", tags="test-push-to-hub-tag-1")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()
    assert "test-push-to-hub-tag-1" in model.model_card_data.tags

    url = model.push_to_hub(
        "cross-encoder-testing/stsb-distilroberta-base", tags=["test-push-to-hub-tag-2", "test-push-to-hub-tag-3"]
    )
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()
    assert "test-push-to-hub-tag-2" in model.model_card_data.tags
    assert "test-push-to-hub-tag-3" in model.model_card_data.tags


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
                "tokenizer_kwargs": {"foo": "baz"},
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
                "tokenizer_kwargs": {"foo": "baz"},
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
            {
                "num_workers": 2,
            },
            {},
        ],
        [
            {  # You have to pass instances normally, but this is easier for testing
                "activation_fct": torch.nn.Sigmoid,
            },
            {
                "activation_fn": torch.nn.Sigmoid,
            },
        ],
        [
            {
                "activation_fct": torch.nn.Identity,
                "activation_fn": torch.nn.Sigmoid,
            },
            {
                "activation_fn": torch.nn.Sigmoid,
            },
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
        model.predict([["Hello there!", "Hello, World!"]], **in_kwargs)
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


@pytest.mark.parametrize(
    ["num_labels", "activation_fn", "saved_activation_fn"],
    [
        [
            1,
            torch.nn.Sigmoid(),
            "torch.nn.modules.activation.Sigmoid",
        ],
        [
            1,
            torch.nn.Identity(),
            "torch.nn.modules.linear.Identity",
        ],
        [
            1,
            torch.nn.Tanh(),
            "torch.nn.modules.activation.Tanh",
        ],
        [
            1,
            torch.nn.Softmax(),
            "torch.nn.modules.activation.Softmax",
        ],
        [
            1,
            None,
            "torch.nn.modules.activation.Sigmoid",
        ],
        [
            3,
            None,
            "torch.nn.modules.linear.Identity",
        ],
    ],
)
def test_load_activation_fn_from_kwargs(num_labels: int, activation_fn: str, saved_activation_fn: str, tmp_path: Path):
    model = CrossEncoder("prajjwal1/bert-tiny", num_labels=num_labels, activation_fn=activation_fn)
    assert fullname(model.activation_fn) == saved_activation_fn

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        config = json.load(f)
    assert config["sentence_transformers"]["activation_fn"] == saved_activation_fn
    assert "sbert_ce_default_activation_function" not in config

    loaded_model = CrossEncoder(tmp_path)
    assert fullname(loaded_model.activation_fn) == saved_activation_fn

    # Setting the activation function via a prediction updates the instance, but not the config
    loaded_model.predict([["Hello there!", "Hello, World!"]], activation_fn=torch.nn.Identity())
    assert fullname(loaded_model.activation_fn) == "torch.nn.modules.linear.Identity"
    assert loaded_model.config.sentence_transformers["activation_fn"] == saved_activation_fn


@pytest.mark.parametrize(
    "tanh_model_name",
    [
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v3",
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4",
    ],
)
def test_load_activation_fn_from_config(tanh_model_name: str, tmp_path):
    saved_activation_fn = "torch.nn.modules.activation.Tanh"

    model = CrossEncoder(tanh_model_name)
    assert fullname(model.activation_fn) == saved_activation_fn

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        config = json.load(f)
    assert config["sentence_transformers"]["activation_fn"] == saved_activation_fn
    assert "sbert_ce_default_activation_function" not in config

    loaded_model = CrossEncoder(tmp_path)
    assert fullname(loaded_model.activation_fn) == saved_activation_fn


def test_load_activation_fn_from_config_custom(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, caplog):
    model = reranker_bert_tiny_model

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        config = json.load(f)
    config["sentence_transformers"]["activation_fn"] = "sentence_transformers.custom.activations.CustomActivation"
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    with caplog.at_level(logging.WARNING):
        CrossEncoder(tmp_path)
        assert (
            "Activation function path 'sentence_transformers.custom.activations.CustomActivation' is not trusted, using default activation function instead."
            in caplog.text
        )

    # If we use trust_remote_code, it'll try to load the custom activation function, which doesn't exist
    with pytest.raises(ImportError):
        model = CrossEncoder(tmp_path, trust_remote_code=True)


def test_default_activation_fn(reranker_bert_tiny_model: CrossEncoder):
    model = reranker_bert_tiny_model
    assert fullname(model.activation_fn) == "torch.nn.modules.activation.Sigmoid"
    with pytest.warns(
        DeprecationWarning, match="The `default_activation_function` property was renamed and is now deprecated.*"
    ):
        assert fullname(model.default_activation_function) == "torch.nn.modules.activation.Sigmoid"


def test_bge_reranker_max_length():
    model = CrossEncoder("BAAI/bge-reranker-base")
    assert model.max_length == 512
    assert model.tokenizer.model_max_length == 512

    model.max_length = 256
    assert model.max_length == 256
    assert model.tokenizer.model_max_length == 256


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
