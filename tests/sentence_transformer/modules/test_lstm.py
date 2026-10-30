from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import LSTM, Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lstm_preserves_padded_batch_when_split_into_shards(bidirectional: bool, dtype: torch.dtype) -> None:
    words = WordEmbeddings(
        tokenizer=WhitespaceTokenizer(vocab=["PADDING_TOKEN", "alpha", "beta", "gamma"], stop_words=[]),
        embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        encoder = LSTM(2, hidden_dim=3, bidirectional=bidirectional)
    model = SentenceTransformer(modules=[words, encoder, Pooling(encoder.get_embedding_dimension())], device="cpu")
    model.to(dtype=dtype)
    features = model.preprocess(["alpha", "beta gamma", "alpha beta gamma alpha"])

    full = model({key: value.clone() for key, value in features.items()})
    shards = [
        model({key: value[start:end].clone() for key, value in features.items()}) for start, end in [(0, 2), (2, 3)]
    ]

    torch.testing.assert_close(
        torch.cat([shard["sentence_embedding"] for shard in shards]), full["sentence_embedding"]
    )
    for shard in shards:
        assert shard["token_embeddings"].shape[:2] == shard["attention_mask"].shape
    torch.testing.assert_close(torch.cat([shard["token_embeddings"] for shard in shards]), full["token_embeddings"])

    full["sentence_embedding"].sum().backward()
    expected_gradients = {name: parameter.grad.clone() for name, parameter in encoder.named_parameters()}
    model.zero_grad()
    torch.cat([shard["sentence_embedding"] for shard in shards]).sum().backward()
    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_gradients[name])
