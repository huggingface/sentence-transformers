from __future__ import annotations

import math
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.models.StaticEmbedding import StaticEmbedding

try:
    import model2vec
except ImportError:
    model2vec = None

skip_if_no_model2vec = pytest.mark.skipif(model2vec is None, reason="The model2vec library is not installed.")


def test_initialization_with_embedding_weights(tokenizer: Tokenizer, embedding_weights) -> None:
    model = StaticEmbedding(tokenizer, embedding_weights=embedding_weights)
    assert model.embedding.weight.shape == (30522, 768)


def test_initialization_with_embedding_dim(tokenizer: Tokenizer) -> None:
    model = StaticEmbedding(tokenizer, embedding_dim=768)
    assert model.embedding.weight.shape == (30522, 768)


def test_tokenize(static_embedding_model: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding_model.preprocess(texts)
    assert "input_ids" in tokens
    assert "offsets" in tokens


def test_forward(static_embedding_model: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding_model.preprocess(texts)
    output = static_embedding_model(tokens)
    assert "sentence_embedding" in output


def test_save_and_load(tmp_path: Path, static_embedding_model: StaticEmbedding) -> None:
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    static_embedding_model.save(str(save_dir))

    loaded_model = StaticEmbedding.load(str(save_dir))
    assert loaded_model.embedding.weight.shape == static_embedding_model.embedding.weight.shape


@skip_if_no_model2vec()
def test_from_distillation() -> None:
    model = StaticEmbedding.from_distillation("sentence-transformers-testing/stsb-bert-tiny-safetensors", pca_dims=32)
    # The shape has been 29528 for <0.5.0, 29525 for 0.5.0, and 29524 for >=0.6.0, so let's make a safer test
    # that checks the first dimension is close to 29525 and the second dimension is 32.
    assert abs(model.embedding.weight.shape[0] - 29525) < 5
    assert model.embedding.weight.shape[1] == 32


@skip_if_no_model2vec()
def test_from_model2vec() -> None:
    model = StaticEmbedding.from_model2vec("minishlab/M2V_base_output")
    assert model.embedding.weight.shape == (29528, 256)


def test_loading_model2vec() -> None:
    model = SentenceTransformer("minishlab/potion-base-8M")
    assert model.get_sentence_embedding_dimension() == 256
    assert model.max_seq_length == math.inf

    test_sentences = ["It's so sunny outside!", "The sun is shining outside!"]
    embeddings = model.encode(test_sentences)
    assert embeddings.shape == (2, 256)
    similarity = model.similarity(embeddings[0], embeddings[1])
    assert similarity.item() > 0.7
