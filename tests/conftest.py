from __future__ import annotations

import os
import tempfile
from copy import deepcopy

import numpy as np
import pytest
from tokenizers import Tokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, StaticEmbedding, Transformer
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import DatasetDict, load_dataset


@pytest.fixture(scope="session")
def _stsb_bert_tiny_model() -> SentenceTransformer:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def stsb_bert_tiny_model(_stsb_bert_tiny_model: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_stsb_bert_tiny_model)


@pytest.fixture(scope="session")
def _avg_word_embeddings_levy() -> SentenceTransformer:
    model = SentenceTransformer("sentence-transformers/average_word_embeddings_levy_dependency")
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
    return SentenceTransformer("paraphrase-distilroberta-base-v1")


@pytest.fixture(scope="session")
def _static_retrieval_mrl_en_v1_model() -> SentenceTransformer:
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    return model


@pytest.fixture()
def static_retrieval_mrl_en_v1_model(_static_retrieval_mrl_en_v1_model: SentenceTransformer) -> SentenceTransformer:
    return deepcopy(_static_retrieval_mrl_en_v1_model)


@pytest.fixture()
def clip_vit_b_32_model() -> SentenceTransformer:
    return SentenceTransformer("clip-ViT-B-32")


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def embedding_weights():
    return np.random.rand(30522, 768).astype(np.float32)


@pytest.fixture
def static_embedding_model(tokenizer: Tokenizer, embedding_weights) -> StaticEmbedding:
    return StaticEmbedding(tokenizer, embedding_weights=embedding_weights)


@pytest.fixture()
def distilbert_base_uncased_model() -> SentenceTransformer:
    word_embedding_model = Transformer("distilbert-base-uncased")
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
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
