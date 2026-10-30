from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
from sentence_transformers.sentence_transformer.modules import WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("method", ["encode", "encode_document", "encode_query"])
@pytest.mark.parametrize("round_trip", [False, True])
@pytest.mark.parametrize("allowed_ids,document_lengths", [(None, [2, 1]), ([], [0, 0]), ([2], [1, 0])])
def test_token_allowlist_applies_after_encode_and_reload(
    tmp_path: Path, method: str, round_trip: bool, allowed_ids: list[int] | None, document_lengths: list[int]
) -> None:
    words = WordEmbeddings(
        tokenizer=WhitespaceTokenizer(vocab=["PADDING_TOKEN", "alpha", "beta"], stop_words=[]),
        embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    model = MultiVectorEncoder(modules=[words, MultiVectorMask(keep_only_token_ids=allowed_ids)], device="cpu")
    if round_trip:
        model.save(str(tmp_path))
        model = MultiVectorEncoder(str(tmp_path), device="cpu")
        assert model[1].keep_only_token_ids == allowed_ids

    embeddings = getattr(model, method)(["alpha beta", "alpha"])

    lengths = [2, 1] if method == "encode_query" else document_lengths
    assert [embedding.shape for embedding in embeddings] == [(length, 2) for length in lengths]


def test_clearing_token_allowlist_removes_document_tokens() -> None:
    module = MultiVectorMask(keep_only_token_ids=[2])
    features = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    initial = module({key: value.clone() for key, value in features.items()}, task="document")
    torch.testing.assert_close(initial["attention_mask"], torch.tensor([[False, True, False]]))

    module.keep_only_token_ids = []
    cleared = module({key: value.clone() for key, value in features.items()}, task="document")

    assert not cleared["attention_mask"].any()
    torch.testing.assert_close(features["attention_mask"], torch.tensor([[1, 1, 0]]))
