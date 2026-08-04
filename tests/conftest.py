from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.utils import validate_repo_id
from tokenizers import Tokenizer

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.sentence_transformer.modules import Pooling, StaticEmbedding, Transformer
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import DatasetDict, load_dataset


def _fake_model_info(model_id: str) -> SimpleNamespace:
    validate_repo_id(model_id)
    if os.path.exists(model_id):
        raise ValueError(f"{model_id!r} is a local path, not a Hub model ID")
    return SimpleNamespace(id=model_id, sha="0123456789abcdef0123456789abcdef01234567")


def _fake_dataset_info(dataset_id: str) -> SimpleNamespace:
    validate_repo_id(dataset_id)
    if os.path.exists(dataset_id):
        raise ValueError(f"{dataset_id!r} is a local path, not a Hub dataset ID")
    return SimpleNamespace(id=dataset_id, cardData=None)


@pytest.fixture(scope="session", autouse=True)
def mock_hub_info():
    """Mock the Hub requests checking whether base models and datasets exist, e.g. to dodge rate limits in CI."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("sentence_transformers.base.model_card.get_model_info", _fake_model_info)
        monkeypatch.setattr("sentence_transformers.base.model_card.get_dataset_info", _fake_dataset_info)
        yield


# Sentence Transformers
@pytest.fixture(scope="session")
def _stsb_bert_tiny_model() -> SentenceTransformer:
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SentenceTransformer(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def stsb_bert_tiny_model(_stsb_bert_tiny_model: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_stsb_bert_tiny_model)


@pytest.fixture(scope="session")
def _avg_word_embeddings_levy() -> SentenceTransformer:
    model_id = "sentence-transformers/average_word_embeddings_levy_dependency"
    model = SentenceTransformer(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def avg_word_embeddings_levy(_avg_word_embeddings_levy: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_avg_word_embeddings_levy)


@pytest.fixture()
def stsb_bert_tiny_model_onnx() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-onnx")


@pytest.fixture()
def stsb_bert_tiny_model_openvino() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-openvino")


@pytest.fixture()
def paraphrase_distilroberta_base_v1_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")


@pytest.fixture(scope="session")
def _static_retrieval_mrl_en_v1_model() -> SentenceTransformer:
    model_id = "sentence-transformers/static-retrieval-mrl-en-v1"
    return SentenceTransformer(model_id)


@pytest.fixture()
def static_retrieval_mrl_en_v1_model(_static_retrieval_mrl_en_v1_model: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_static_retrieval_mrl_en_v1_model)


@pytest.fixture()
def clip_vit_b_32_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/clip-ViT-B-32")


@pytest.fixture(scope="session")
def _distilbert_base_uncased_model() -> SentenceTransformer:
    model_id = "distilbert/distilbert-base-uncased"
    word_embedding_model = Transformer(model_id)
    pooling_model = Pooling(word_embedding_model.get_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if not model.model_card_data.base_model:
        model.model_card_data.base_model = model_id
    return model


@pytest.fixture()
def distilbert_base_uncased_model(_distilbert_base_uncased_model: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_distilbert_base_uncased_model)


# Cross Encoders
@pytest.fixture(scope="session")
def _reranker_bert_tiny_model() -> CrossEncoder:
    model_id = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"
    model = CrossEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def reranker_bert_tiny_model(_reranker_bert_tiny_model) -> CrossEncoder:
    return deepcopy(_reranker_bert_tiny_model)


# Sparse Encoders
@pytest.fixture(scope="session")
def _splade_bert_tiny_model() -> SparseEncoder:
    model_id = "sparse-encoder-testing/splade-bert-tiny-nq"
    model = SparseEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def splade_bert_tiny_model(_splade_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    return deepcopy(_splade_bert_tiny_model)


@pytest.fixture(scope="session")
def _inference_free_splade_bert_tiny_model() -> SparseEncoder:
    model_id = "sparse-encoder-testing/inference-free-splade-bert-tiny-nq"
    model = SparseEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def inference_free_splade_bert_tiny_model(_inference_free_splade_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    return deepcopy(_inference_free_splade_bert_tiny_model)


@pytest.fixture(scope="session")
def _csr_bert_tiny_model() -> SparseEncoder:
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SparseEncoder(model_id)
    model[-1].k = 16
    model[-1].k_aux = 32
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def csr_bert_tiny_model(_csr_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    return deepcopy(_csr_bert_tiny_model)


# Tokenization & Datasets
@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("google-bert/bert-base-uncased")


@pytest.fixture
def embedding_weights():
    return np.random.rand(30522, 768).astype(np.float32)


@pytest.fixture
def static_embedding(tokenizer: Tokenizer, embedding_weights) -> StaticEmbedding:
    return StaticEmbedding(tokenizer, embedding_weights=embedding_weights)


@pytest.fixture
def static_embedding_model(static_embedding: StaticEmbedding) -> SentenceTransformer:
    model = SentenceTransformer(modules=[static_embedding])
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.fixture(scope="session")
def stsb_dataset_dict() -> DatasetDict:
    return load_dataset("sentence-transformers/stsb")


@pytest.fixture()
def cache_dir():
    """
    In the CI environment, we use a temporary directory as `cache_dir`
    to avoid keeping the downloaded models on disk after the test.
    """
    if os.environ.get("CI", None):
        # Note: `ignore_cleanup_errors=True` is used to avoid NotADirectoryError in Windows on GitHub Actions.
        # See https://github.com/python/cpython/issues/107408, https://www.scivision.dev/python-tempfile-permission-error-windows/
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            yield tmp_dir
    else:
        yield None
