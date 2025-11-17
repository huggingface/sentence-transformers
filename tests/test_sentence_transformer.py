"""
Tests general behaviour of the SentenceTransformer class
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from tokenizers.processors import TemplateProcessing
from torch import nn
from transformers import BertModel
from transformers.utils import is_peft_available

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import (
    CNN,
    LSTM,
    CLIPModel,
    Dense,
    LayerNorm,
    Normalize,
    Pooling,
    Router,
    Transformer,
    WeightedLayerPooling,
)
from sentence_transformers.similarity_functions import SimilarityFunction
from tests.utils import is_ci


def test_load_with_safetensors() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        safetensors_model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_folder=cache_folder,
        )

        # Only the safetensors file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 0 == len(pytorch_files), "PyTorch model file must not be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 1 == len(safetensors_files), "Safetensors model file must be downloaded."

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        transformer = Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_dir=cache_folder,
            model_args={"use_safetensors": False},
        )
        pooling = Pooling(transformer.get_word_embedding_dimension())
        pytorch_model = SentenceTransformer(modules=[transformer, pooling])

        # Only the pytorch file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 1 == len(pytorch_files), "PyTorch model file must be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 0 == len(safetensors_files), "Safetensors model file must not be downloaded."

    sentences = ["This is a test sentence", "This is another test sentence"]
    assert torch.equal(
        safetensors_model.encode(sentences, convert_to_tensor=True),
        pytorch_model.encode(sentences, convert_to_tensor=True),
    ), "Ensure that Safetensors and PyTorch loaded models result in identical embeddings"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_to() -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

    test_device = torch.device("cuda")
    assert model.device.type == "cpu"
    assert test_device.type == "cuda"

    model.to(test_device)
    assert model.device.type == "cuda", "The model device should have updated"

    model.encode("Test sentence")
    assert model.device.type == "cuda", "Encoding shouldn't change the device"

    assert model._target_device == model.device, "Prevent backwards compatibility failure for _target_device"
    model._target_device = "cpu"
    assert model.device.type == "cpu", "Ensure that setting `_target_device` doesn't crash."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test fp16 and bf16 inference.")
@pytest.mark.parametrize("torch_dtype", ["auto", "float16", "bfloat16", torch.float16, torch.bfloat16])
def test_torch_dtype(torch_dtype) -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/all-nli-bert-tiny-dense",
        device="cuda",
        model_kwargs={"torch_dtype": torch_dtype},
    )
    embedding = model.encode("Test sentence")
    assert embedding.shape[-1] == model.get_sentence_embedding_dimension()


def test_push_to_hub(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs
        if kwargs.get("revision") is None:
            return CommitInfo(
                commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/123456",
                commit_message="commit_message",
                commit_description="commit_description",
                oid="oid",
            )
        else:
            return CommitInfo(
                commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/678901",
                commit_message="commit_message",
                commit_description="commit_description",
                oid="oid",
            )

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")

    url = model.push_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        mock_upload_folder_kwargs.clear()
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )

    with pytest.raises(
        ValueError, match="Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
    ):
        model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="unrelated")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="sentence-transformers-testing"
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub("stsb-bert-tiny-safetensors", organization="sentence-transformers-testing")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", local_model_path="my_fake_local_model_path"
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert mock_upload_folder_kwargs["folder_path"] == "my_fake_local_model_path"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Using deprecated "repo_name" positional argument
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(repo_name="sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
        )
        assert (
            caplog.record_tuples[1][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Use positional arguments from before "token" was introduced
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "stsb-bert-tiny-safetensors",  # repo_name
            "sentence-transformers-testing",  # organization
            True,  # private
            commit_message="Adding new awesome Model!",
            exist_ok=True,
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert mock_upload_folder_kwargs["commit_message"] == "Adding new awesome Model!"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 2
        assert (
            caplog.record_tuples[0][2]
            == "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers. Please use `push_to_hub` instead for future model uploads."
        )
        assert (
            caplog.record_tuples[1][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="revision_test")
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert mock_upload_folder_kwargs["revision"] == "revision_test"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/678901"
    mock_upload_folder_kwargs.clear()


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(stsb_bert_tiny_model: SentenceTransformer, safe_serialization: bool) -> None:
    model = stsb_bert_tiny_model
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        if safe_serialization:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            assert 1 == len(model_files)


def test_load_with_revision() -> None:
    main_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="main")
    latest_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="f3cb857cba53019a20df283396bcca179cf051a4"
    )
    older_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24"
    )

    test_sentence = ["Hello there!"]
    main_embeddings = main_model.encode(test_sentence, convert_to_tensor=True)
    assert torch.equal(main_embeddings, latest_model.encode(test_sentence, convert_to_tensor=True))
    assert not torch.equal(main_embeddings, older_model.encode(test_sentence, convert_to_tensor=True))


def test_load_local_without_normalize_directory(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    model.add_module("Normalize", Normalize())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save(str(model_path))

        assert (model_path / "2_Normalize").exists()
        os.rmdir(model_path / "2_Normalize")
        assert not (model_path / "2_Normalize").exists()

        # This fails in v2.3.0
        fresh_tiny_model = SentenceTransformer(str(model_path))
        assert isinstance(fresh_tiny_model, SentenceTransformer)


def test_prompts(stsb_bert_tiny_model: SentenceTransformer, caplog: pytest.LogCaptureFixture) -> None:
    model = stsb_bert_tiny_model
    assert model.prompts == {"query": "", "document": ""}
    assert model.default_prompt_name is None
    texts = ["How to bake a chocolate cake", "Symptoms of the flu"]
    no_prompt_embedding = model.encode(texts)
    prompt_embedding = model.encode([f"query: {text}" for text in texts])
    assert not np.array_equal(no_prompt_embedding, prompt_embedding)

    for query in ["query: ", "query:", "query:   "]:
        # Test prompt="... {}"
        model.prompts = {"query": "", "document": ""}
        assert np.array_equal(model.encode(texts, prompt=query), prompt_embedding)

        # Test prompt_name="..."
        model.prompts = {"query": query, "document": ""}
        assert np.array_equal(model.encode(texts, prompt_name="query"), prompt_embedding)

        caplog.clear()
        # Test prompt_name="..." & prompt="..."
        with caplog.at_level(logging.WARNING):
            assert np.array_equal(model.encode(texts, prompt=query, prompt_name="query"), prompt_embedding)
            assert len(caplog.record_tuples) == 1
            assert (
                caplog.record_tuples[0][2]
                == "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                "Ignoring the `prompt_name` in favor of `prompt`."
            )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query', 'document']."
            ),
        ):
            model.encode(texts, prompt_name="invalid_prompt_name")


def test_save_load_prompts() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Default prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query', 'document']."
        ),
    ):
        model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            prompts={"query": "query: "},
            default_prompt_name="invalid_prompt_name",
        )

    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "query: "},
        default_prompt_name="query",
    )
    assert model.prompts == {"query": "query: ", "document": ""}
    assert model.default_prompt_name == "query"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save(str(model_path))
        config_path = model_path / "config_sentence_transformers.json"
        assert config_path.exists()
        with open(config_path, encoding="utf8") as f:
            saved_config = json.load(f)
        assert saved_config["prompts"] == {"query": "query: ", "document": ""}
        assert saved_config["default_prompt_name"] == "query"

        fresh_model = SentenceTransformer(str(model_path))
        assert fresh_model.prompts == {"query": "query: ", "document": ""}
        assert fresh_model.default_prompt_name == "query"


def test_prompt_output_value_None(stsb_bert_tiny_model) -> None:
    model = stsb_bert_tiny_model
    outputs = model.encode(
        ["Text one", "Text two"],
        prompt="query: ",
        output_value=None,
    )
    assert len(outputs) == 2
    assert isinstance(outputs, list)
    expected_keys = {
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "sentence_embedding",
        "token_embeddings",
        "prompt_length",
    }
    assert set(outputs[0].keys()) == expected_keys
    assert set(outputs[1].keys()) == expected_keys


@pytest.mark.parametrize("has_bos_token", [True, False])
@pytest.mark.parametrize("has_eos_token", [True, False])
def test_prompt_length_calculation(
    stsb_bert_tiny_model: SentenceTransformer,
    has_bos_token: bool,
    has_eos_token: bool,
) -> None:
    # This test checks that the prompt length is calculated correctly, regardless of whether the tokenizer has a
    # beginning-of-sequence or end-of-sequence token.
    model = stsb_bert_tiny_model
    model.prompts = {"query": "Prompt: ", "document": ""}
    # We need to set this to False, otherwise the prompt length wont be needed:
    model.set_pooling_include_prompt(False)
    dummy_bos_token_id = 400
    dummy_eos_token_id = 500
    model.tokenizer.cls_token_id = dummy_bos_token_id if has_bos_token else None
    model.tokenizer.sep_token_id = dummy_eos_token_id if has_eos_token else None
    if has_bos_token:
        if has_eos_token:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0 [SEP]",
                special_tokens=[
                    ("[CLS]", dummy_bos_token_id),
                    ("[SEP]", dummy_eos_token_id),
                ],
            )
        else:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0",
                special_tokens=[("[CLS]", dummy_bos_token_id)],
            )
    else:
        if has_eos_token:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="$0 [SEP]",
                special_tokens=[("[SEP]", dummy_eos_token_id)],
            )
        else:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="$0",
                special_tokens=[],
            )

    # Check that we can update the tokenizer in this way
    if has_eos_token:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[-1] == dummy_eos_token_id
    else:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[-1] != dummy_eos_token_id

    if has_bos_token:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[0] == dummy_bos_token_id
    else:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[0] != dummy_bos_token_id

    # This should populate _prompt_length_mapping
    model.encode_query("This is a test sentence")

    only_prompt_length = len(model.tokenizer(["Prompt: "], add_special_tokens=False)["input_ids"][0])
    if has_bos_token:
        only_prompt_length += 1

    assert model._prompt_length_mapping == {("Prompt: ", "query"): only_prompt_length}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test float16 support.")
def test_load_with_torch_dtype(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model

    assert model.encode(["Hello there!"], convert_to_tensor=True).dtype == torch.float32

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        fp16_model_dir = Path(tmp_folder) / "fp16_model"
        model.half()
        model.save(str(fp16_model_dir))
        del model

        fp16_model = SentenceTransformer(
            str(fp16_model_dir),
            model_kwargs={"torch_dtype": "auto"},
        )
        assert fp16_model.encode(["Hello there!"], convert_to_tensor=True).dtype == torch.float16


def test_load_with_model_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer_kwargs = {}
    original_transformer_init = Transformer.__init__

    def transformers_init(*args, **kwargs):
        nonlocal transformer_kwargs
        nonlocal original_transformer_init
        transformer_kwargs = kwargs
        return original_transformer_init(*args, **kwargs)

    monkeypatch.setattr(Transformer, "__init__", transformers_init)

    SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        model_kwargs={"attn_implementation": "eager", "low_cpu_mem_usage": False},
    )

    assert "low_cpu_mem_usage" in transformer_kwargs["model_args"]
    assert transformer_kwargs["model_args"]["low_cpu_mem_usage"] is False
    assert "attn_implementation" in transformer_kwargs["model_args"]
    assert transformer_kwargs["model_args"]["attn_implementation"] == "eager"


@pytest.mark.skipif(not is_peft_available(), reason="PEFT must be available to test PEFT support.")
def test_load_checkpoint_with_peft_and_lora(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    from peft import LoraConfig, TaskType

    peft_config = LoraConfig(
        target_modules=["query", "key", "value"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model.add_adapter(peft_config)
        model.save(tmp_folder)
        expecteds = model.encode(["Hello there!", "How are you?"], convert_to_tensor=True)

        loaded_peft_model = SentenceTransformer(tmp_folder)
        actuals = loaded_peft_model.encode(["Hello there!", "How are you?"], convert_to_tensor=True)

        assert isinstance(model[0].auto_model, nn.Module)
        assert isinstance(loaded_peft_model[0].auto_model, BertModel)
        assert loaded_peft_model[0].auto_model._hf_peft_config_loaded
        assert loaded_peft_model[0].auto_model.active_adapters() == ["default"]
        assert torch.equal(expecteds, actuals)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test float16 support.")
def test_encode_fp16(stsb_bert_tiny_model: SentenceTransformer) -> None:
    tiny_model = stsb_bert_tiny_model
    tiny_model.half()
    embeddings = tiny_model.encode(["Hello there!"], convert_to_tensor=True)
    assert embeddings.dtype == torch.float16


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize(
    ("precision", "expected_torch_dtype", "expected_numpy_dtype", "raises"),
    [
        (None, torch.float32, np.float32, False),
        ("float32", torch.float32, np.float32, False),
        ("int8", torch.int8, np.int8, False),
        ("uint8", torch.uint8, np.uint8, False),
        ("binary", torch.int8, np.int8, False),
        ("ubinary", torch.uint8, np.uint8, False),
        ("error", None, None, True),
    ],
)
def test_encode_quantization(
    stsb_bert_tiny_model: SentenceTransformer,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    precision: str,
    expected_torch_dtype,
    expected_numpy_dtype,
    raises: bool,
) -> None:
    tiny_model = stsb_bert_tiny_model
    with pytest.raises(ValueError, match=f"Precision '{precision}' is not supported") if raises else nullcontext():
        embeddings = tiny_model.encode(
            ["One sentence", "Another sentence"],
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            precision=precision,
        )

    if not raises:
        if convert_to_tensor:
            assert embeddings[0].dtype == expected_torch_dtype
            assert isinstance(embeddings, torch.Tensor)
        elif convert_to_numpy:
            assert embeddings[0].dtype == expected_numpy_dtype
            assert isinstance(embeddings, np.ndarray)
        else:
            assert embeddings[0].dtype == expected_torch_dtype
            assert isinstance(embeddings, list)


@pytest.mark.parametrize("sentences", ("Single sentence", ["One sentence", "Another sentence"]))
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("normalize_embeddings", [True, False])
@pytest.mark.parametrize("output_value", ["sentence_embedding", None])
def test_encode_truncate(
    stsb_bert_tiny_model: SentenceTransformer,
    sentences: str | list[str],
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    output_value: Literal["sentence_embedding"] | None,
) -> None:
    model = stsb_bert_tiny_model
    embeddings_full_unnormalized: torch.Tensor = model.encode(
        sentences, convert_to_numpy=False, convert_to_tensor=True
    )  # These are raw embeddings which serve as the reference to test against

    def test(model: SentenceTransformer, expected_dim: int):
        outputs = model.encode(
            sentences,
            output_value=output_value,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        # Extract the sentence embeddings out of outputs
        if output_value is None:
            # We get the whole plate
            if not isinstance(outputs, list):
                embeddings = outputs["sentence_embedding"]
            else:
                outputs = cast(list[dict[str, torch.Tensor]], outputs)
                embeddings = [out_features["sentence_embedding"] for out_features in outputs]
        else:
            embeddings = outputs

        # Test shape
        if isinstance(embeddings, list):  # list of tensors
            embeddings_shape = (len(embeddings), embeddings[0].shape[-1])
        else:
            embeddings_shape = embeddings.shape
        expected_shape = (expected_dim,) if isinstance(sentences, str) else (len(sentences), expected_dim)
        assert embeddings_shape == expected_shape
        assert model.get_sentence_embedding_dimension() == expected_dim

        # Convert embeddings to a torch Tensor for ease of testing
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        elif isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(embeddings_full_unnormalized.device)
            # On a non-cpu device, the device of torch.from_numpy(embeddings) is always CPU

        # Test content
        if normalize_embeddings:
            if output_value is None:
                # Currently, normalization is not performed; it's the raw output of the forward pass
                pass
            else:
                normalize = partial(torch.nn.functional.normalize, p=2, dim=-1)
                assert torch.allclose(
                    embeddings,
                    normalize(util.truncate_embeddings(embeddings_full_unnormalized, expected_dim)),
                )
        else:
            assert torch.allclose(embeddings, util.truncate_embeddings(embeddings_full_unnormalized, expected_dim))

    # Test init w/o setting truncate_dim (it's None)
    original_output_dim: int = model.get_sentence_embedding_dimension()
    test(model, expected_dim=original_output_dim)

    # Test init w/ a set truncate_dim
    truncate_dim = int(original_output_dim / 4)
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", truncate_dim=truncate_dim)
    test(model, expected_dim=truncate_dim)

    # Test setting the attribute after init to a greater dimension
    new_truncate_dim = 2 * truncate_dim
    model.truncate_dim = new_truncate_dim
    test(model, expected_dim=new_truncate_dim)

    # Test context manager
    final_truncate_dim = int(original_output_dim / 8)
    with model.truncate_sentence_embeddings(final_truncate_dim):
        test(model, expected_dim=final_truncate_dim)
    test(model, expected_dim=new_truncate_dim)  # b/c we've exited the context

    # Test w/ an ouptut_dim that's larger than the original_output_dim. No truncation ends up happening
    model.truncate_dim = 2 * original_output_dim
    test(model, expected_dim=original_output_dim)


@pytest.mark.parametrize("similarity_fn_name", SimilarityFunction.possible_values())
def test_similarity_score(stsb_bert_tiny_model: SentenceTransformer, similarity_fn_name: str) -> None:
    model = stsb_bert_tiny_model
    model.similarity_fn_name = similarity_fn_name
    sentences = [
        "The weather is so nice!",
        "It's so sunny outside.",
        "He's driving to the movie theater.",
        "She's going to the cinema.",
    ]
    embeddings = model.encode(sentences, normalize_embeddings=True)
    scores = model.similarity(embeddings, embeddings)
    assert scores.shape == (len(sentences), len(sentences))
    if similarity_fn_name in ("cosine", "dot"):
        expected = np.ones(4, dtype=float)
    else:
        expected = np.zeros(4, dtype=float)
    np.testing.assert_almost_equal(np.diag(scores), expected, decimal=4)
    assert scores[1][0] > scores[2][0]
    assert scores[1][0] > scores[3][0]
    assert scores[2][3] > scores[2][0]
    assert scores[2][3] > scores[2][1]

    pairwise_scores = model.similarity_pairwise(embeddings[::2], embeddings[1::2])
    assert pairwise_scores.shape == (len(sentences) // 2,)
    if similarity_fn_name in ("cosine", "dot"):
        assert (pairwise_scores > 0.5).all()


def test_similarity_score_save(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    embeddings = model.encode(["Sentence 1", "Sentence 2"])
    assert model.similarity_fn_name == "cosine"
    cosine_scores = model.similarity(embeddings, embeddings)
    # Using 'similarity' methods sets the default similarity function to 'cosine'
    assert model.similarity_fn_name == "cosine"

    model.similarity_fn_name = "euclidean"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model.save(tmp_folder)
        loaded_model = SentenceTransformer(tmp_folder)
    assert loaded_model.similarity_fn_name == "euclidean"
    dot_scores = model.similarity(embeddings, embeddings)
    assert np.not_equal(cosine_scores, dot_scores).all()


def test_model_card_save_update_model_id(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    # Removing the saved model card will cause a fresh one to be generated when we save
    model._model_card_text = ""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model.save(tmp_folder)
        with open(Path(tmp_folder) / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
            assert 'model = SentenceTransformer("sentence_transformers_model_id"' in model_card_text

        # When we reload this saved model and then re-save it, we want to override the 'sentence_transformers_model_id'
        # if we have it set
        loaded_model = SentenceTransformer(tmp_folder)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        loaded_model.save(tmp_folder, model_name="test_user/test_model")

        with open(Path(tmp_folder) / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
            assert 'model = SentenceTransformer("test_user/test_model"' in model_card_text


def test_override_config_versions(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model

    assert model._model_config["__version__"]["sentence_transformers"] == "2.2.2"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model.save(tmp_folder)
        loaded_model = SentenceTransformer(tmp_folder)
    # Verify that the version has now been updated when saving the model again
    assert loaded_model._model_config["__version__"]["sentence_transformers"] != "2.2.2"


@pytest.mark.parametrize(
    "modules",
    [
        lambda bert_model, word_model: [
            bert_model[0],
            Pooling(128, "mean"),
            Dense(128, 128),
        ],
        lambda bert_model, word_model: [
            bert_model[0],
            CNN(128, 128),
            Pooling(128, "mean"),
        ],
        lambda bert_model, word_model: [
            bert_model[0],
            Pooling(128, "mean"),
            LayerNorm(128),
        ],
        lambda bert_model, word_model: [
            word_model[0],
            LSTM(300, 128),
            Pooling(128, "mean"),
        ],
        lambda bert_model, word_model: [
            bert_model[0],
            WeightedLayerPooling(128, num_hidden_layers=2, layer_start=1),
            Pooling(128, "mean"),
        ],
        lambda bert_model, word_model: word_model,
    ],
)
def test_safetensors(
    stsb_bert_tiny_model: SentenceTransformer,
    avg_word_embeddings_levy: SentenceTransformer,
    modules: Callable[[], list[nn.Module] | SentenceTransformer],
) -> None:
    modules = modules(stsb_bert_tiny_model, avg_word_embeddings_levy)  # Call the lambda to get the actual modules
    if isinstance(modules, SentenceTransformer):
        model = modules
    else:
        # output_hidden_states must be True for WeightedLayerPooling
        if isinstance(modules[1], WeightedLayerPooling):
            modules[0].auto_model.config.output_hidden_states = True
        model = SentenceTransformer(modules=modules)
    original_embedding = model.encode("Hello, World!")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model.save(tmp_folder)
        # Ensure that we only have the safetensors file and no pytorch_model.bin
        assert list(Path(tmp_folder).rglob("**/model.safetensors"))
        assert not list(Path(tmp_folder).rglob("**/pytorch_model.bin"))

        # Ensure that we can load the model again and get the same embeddings
        loaded_model = SentenceTransformer(tmp_folder)
        loaded_embedding = loaded_model.encode("Hello, World!")
        assert np.allclose(original_embedding, loaded_embedding, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("normalize_embeddings", [True, False])
@pytest.mark.parametrize("precision", [None, "float32", "int8", "uint8", "binary", "ubinary"])
@pytest.mark.parametrize("truncate_dim", [None, 64, 128])
def test_empty_encode(
    stsb_bert_tiny_model: SentenceTransformer,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    precision: str | None,
    truncate_dim: int | None,
) -> None:
    model = stsb_bert_tiny_model
    embeddings = model.encode(
        [],
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        normalize_embeddings=normalize_embeddings,
        precision=precision,
        truncate_dim=truncate_dim,
    )

    if convert_to_tensor:
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.numel() == 0
        assert embeddings.device == model.device
    elif convert_to_numpy:
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0
    else:
        assert embeddings == []


@pytest.mark.skipif(not is_peft_available(), reason="PEFT must be available to test adapter methods.")
@pytest.mark.skipif(
    is_ci(), reason="huggingface_hub & PEFT incorrectly set the user agent in the CI, leading to failures."
)
def test_multiple_adapters(
    stsb_bert_tiny_model: SentenceTransformer, avg_word_embeddings_levy: SentenceTransformer
) -> None:
    text = "Hello, World!"
    model = stsb_bert_tiny_model
    vec_initial = model.encode(text)
    from peft import LoraConfig, TaskType, get_model_status

    # Adding a fresh adapter
    peft_config = LoraConfig(
        target_modules=["query", "key", "value"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights=False,  # Random initialization to test the adapter
    )
    model.add_adapter(peft_config)

    # Load an adapter from the hub
    model.load_adapter("sentence-transformers-testing/stsb-bert-tiny-lora", "hub_adapter")

    # Adding another one with a different name
    peft_config = LoraConfig(
        target_modules=["value"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=2,
        lora_alpha=16,
        lora_dropout=0.1,
        init_lora_weights=False,  # Random initialization to test the adapter
    )
    model.add_adapter(peft_config, "my_adapter")

    # Check that peft recognizes the adapters while we compute vectors for later comparison
    status = get_model_status(model)
    assert status.available_adapters == ["default", "hub_adapter", "my_adapter"]
    assert status.enabled
    assert status.active_adapters == ["my_adapter"]
    assert status.active_adapters == model.active_adapters()
    vec_my_adapter = model.encode(text)

    model.set_adapter("default")
    status = get_model_status(model)
    assert status.active_adapters == ["default"]
    vec_default_adapter = model.encode(text)

    model.disable_adapters()
    status = get_model_status(model)
    assert not status.enabled
    vec_no_adapter = model.encode(text)

    # Check that each vector is different
    assert not np.allclose(vec_my_adapter, vec_default_adapter)
    assert not np.allclose(vec_my_adapter, vec_no_adapter)
    assert not np.allclose(vec_default_adapter, vec_no_adapter)
    # Check that the vectors from the original model match
    assert np.allclose(vec_initial, vec_no_adapter)

    # Check that for non Transformer-based models we have an error
    model = avg_word_embeddings_levy
    with pytest.raises(ValueError, match="PEFT methods are only supported"):
        model.add_adapter(peft_config)


@pytest.mark.skipif(not is_peft_available(), reason="PEFT must be available to test loading PEFT models.")
@pytest.mark.skipif(
    is_ci(), reason="huggingface_hub & PEFT incorrectly set the user agent in the CI, leading to failures."
)
def test_load_adapter_with_revision():
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-lora", revision="3b4f75bcb3dec36a7e05da8c44ee2f7f1d023b1a"
    )
    embeddings = model.encode("Hello, World!")
    assert embeddings.shape == (128,)


def test_clip():
    model = CLIPModel()
    assert model.max_seq_length == 77
    assert model.processor.tokenizer.model_max_length == 77
    tokenized = model.tokenize(["This is my text sentence"])
    assert "input_ids" in tokenized
    assert tokenized["input_ids"].shape == (1, 7)

    model.max_seq_length = 5
    assert model.max_seq_length == 5
    assert model.processor.tokenizer.model_max_length == 5
    tokenized = model.tokenize(["This is my text sentence"])
    assert "input_ids" in tokenized
    assert tokenized["input_ids"].shape == (1, 5)


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("prompt_name", [None, "query", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
def test_encode_query(
    stsb_bert_tiny_model: SentenceTransformer,
    sentences: str | list[str],
    prompt_name: str | None,
    prompt: str | None,
    convert_to_numpy: bool,
    convert_to_tensor: bool,
):
    model = stsb_bert_tiny_model
    # Create a mock model with required prompts
    model.prompts = {"query": "query: ", "custom": "custom: "}

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_query
        model.encode_query(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=32,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        # Verify that encode was called with the correct parameters
        if prompt_name:
            expected_prompt_name = prompt_name
        elif prompt is not None:
            expected_prompt_name = None
        else:
            expected_prompt_name = "query"

        mock_encode.assert_called_once()
        args, kwargs = mock_encode.call_args

        # Check that sentences were passed correctly
        assert kwargs["sentences"] == sentences

        # Check prompt handling
        assert kwargs["prompt"] == prompt
        assert kwargs["prompt_name"] == expected_prompt_name

        # Check other parameters
        assert kwargs["convert_to_numpy"] == convert_to_numpy
        assert kwargs["convert_to_tensor"] == convert_to_tensor
        assert kwargs["task"] == "query"


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("prompt_name", [None, "document", "passage", "corpus", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
def test_encode_document(
    stsb_bert_tiny_model: SentenceTransformer,
    sentences: str | list[str],
    prompt_name: str | None,
    prompt: str | None,
    convert_to_numpy: bool,
    convert_to_tensor: bool,
):
    # Create a mock model with required prompts
    model = stsb_bert_tiny_model
    model.prompts = {"document": "document: ", "passage": "passage: ", "corpus": "corpus: ", "custom": "custom: "}

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_document
        model.encode_document(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=32,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        # Verify that encode was called with the correct parameters
        mock_encode.assert_called_once()
        args, kwargs = mock_encode.call_args

        if prompt_name:
            expected_prompt_name = prompt_name
        elif prompt is not None:
            expected_prompt_name = None
        else:
            expected_prompt_name = "document"

        # Check that sentences were passed correctly
        assert kwargs["sentences"] == sentences

        # Check prompt handling
        assert kwargs["prompt"] == prompt
        assert kwargs["prompt_name"] == expected_prompt_name

        # Check other parameters
        assert kwargs["convert_to_numpy"] == convert_to_numpy
        assert kwargs["convert_to_tensor"] == convert_to_tensor
        assert kwargs["task"] == "document"


def test_encode_document_prompt_priority(stsb_bert_tiny_model: SentenceTransformer):
    """Test that proper prompt priority is respected when multiple options are available"""
    model = stsb_bert_tiny_model
    model.prompts = {
        "document": "document: ",
        "passage": "passage: ",
        "corpus": "corpus: ",
    }

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call encode_document with no explicit prompt
        model.encode_document("test")

        # It should select "document" by default since that's first in the priority list
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "document"

        # Remove document, should fall back to passage
        mock_encode.reset_mock()
        model.prompts = {
            "passage": "passage: ",
            "corpus": "corpus: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "passage"

        # Remove passage, should fall back to corpus
        mock_encode.reset_mock()
        model.prompts = {
            "corpus": "corpus: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] == "corpus"

        # No relevant prompts defined
        mock_encode.reset_mock()
        model.prompts = {
            "query": "query: ",
        }
        model.encode_document("test")
        args, kwargs = mock_encode.call_args
        assert kwargs["prompt_name"] is None


def test_encode_advanced_parameters(stsb_bert_tiny_model: SentenceTransformer):
    """Test that additional parameters are correctly passed to encode"""
    model = stsb_bert_tiny_model

    # Create a mock for the encode method
    with patch.object(model, "encode", autospec=True) as mock_encode:
        # Call with advanced parameters
        model.encode_query(
            "test",
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
            output_value="token_embeddings",
            precision="uint8",
            truncate_dim=128,
            chunk_size=10,
            custom_param="value",
        )

        # Verify all parameters were passed correctly
        args, kwargs = mock_encode.call_args
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["batch_size"] == 64
        assert kwargs["show_progress_bar"] is True
        assert kwargs["output_value"] == "token_embeddings"
        assert kwargs["precision"] == "uint8"
        assert kwargs["truncate_dim"] == 128
        assert kwargs["chunk_size"] == 10
        assert kwargs["custom_param"] == "value"


@pytest.mark.parametrize("inputs", ["test sentence", ["test sentence"]])
def test_encode_query_document_vs_encode(stsb_bert_tiny_model: SentenceTransformer, inputs: str | list[str]):
    """Test the actual integration with encode vs encode_query/encode_document"""
    # This test requires a real model, but we'll use a small one
    model = stsb_bert_tiny_model
    model.prompts = {"query": "query: ", "document": "document: "}

    # Get embeddings with encode_query and encode_document
    query_embeddings = model.encode_query(inputs)
    document_embeddings = model.encode_document(inputs)

    # And the same but with encode via prompts (task doesn't help here)
    encode_query_embeddings = model.encode(inputs, prompt_name="query")
    encode_document_embeddings = model.encode(inputs, prompt_name="document")

    # With prompts they should be the same
    np.testing.assert_allclose(query_embeddings, encode_query_embeddings)
    np.testing.assert_allclose(document_embeddings, encode_document_embeddings)

    # Without prompts they should be different
    query_embeddings_without_prompt = model.encode(inputs)
    document_embeddings_without_prompt = model.encode(inputs)

    # Embeddings should differ when different prompts are used
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(query_embeddings_without_prompt, query_embeddings)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(document_embeddings_without_prompt, document_embeddings)


def test_encode_with_dataset_column(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Test that encode can handle a dataset column as input."""
    model = stsb_bert_tiny_model
    from datasets import Dataset

    # Create a simple dataset with a text column
    dataset = Dataset.from_dict({"text": ["This is a test.", "Another sentence."]})

    # Encode the dataset column
    embeddings = model.encode(dataset["text"], convert_to_tensor=True)

    # Check the shape of the embeddings
    assert embeddings.shape == (2, model.get_sentence_embedding_dimension())


def test_get_model_kwargs(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Test that get_model_kwargs returns the correct keyword arguments."""
    model = stsb_bert_tiny_model

    # Check that the forward kwargs are as expected, i.e. no extra forward kwargs
    # for this basic model
    forward_kwargs = model.get_model_kwargs()
    assert forward_kwargs == []
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SentenceTransformer.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SentenceTransformer.get_model_kwargs(), this model does not accept "
            "any additional keyword arguments."
        ),
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")

    # If one of the modules has additional forward kwargs, they should be included
    model[0].forward_kwargs = {"foo"}
    model[1].forward_kwargs = {"bar", "baz"}
    assert set(model.get_model_kwargs()) == {"foo", "bar", "baz"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SentenceTransformer.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SentenceTransformer.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'bar'|'baz'|, ){5}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r".*?forward\(\) got an unexpected keyword argument '(foo|bar)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", foo=True, bar=False)

    # And also if we have a Router in place
    query_pooling_copy = copy.deepcopy(model[1])
    query_pooling_copy.forward_kwargs = {"query_arg"}
    document_pooling_copy = copy.deepcopy(model[1])
    document_pooling_copy.forward_kwargs = {"document_arg_1", "document_arg_2"}
    model[1] = Router.for_query_document(
        query_modules=[query_pooling_copy],
        document_modules=[document_pooling_copy],
    )
    assert set(model.get_model_kwargs()) == {"foo", "task", "query_arg", "document_arg_1", "document_arg_2"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SentenceTransformer.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SentenceTransformer.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'task'|'query_arg'|'document_arg_1'|'document_arg_2'|, ){9}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", task="query", normalize=True)
    # This should run fine
    model.encode("Test sentence", task="document")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r".*?forward\(\) got an unexpected keyword argument '(foo|document_arg_1)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", task="document", foo=True, document_arg_1=12)
