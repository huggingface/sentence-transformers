from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

TEXTS = [
    "short",
    "a somewhat longer sentence with a few more tokens in it",
    "medium length text here",
    "another document",
]


@pytest.mark.parametrize("truncate_dim", [None, 64])
@pytest.mark.parametrize("default_truncate_dim", [None, 32])
def test_multi_process_matches_single_process(
    mve_bert_tiny_model: MultiVectorEncoder, truncate_dim: int | None, default_truncate_dim: int | None
) -> None:
    model = mve_bert_tiny_model
    model.truncate_dim = default_truncate_dim
    model.to("cpu")
    direct = model.encode_document(TEXTS, truncate_dim=truncate_dim)
    pooled = model.encode_document(TEXTS, device=["cpu", "cpu"], truncate_dim=truncate_dim)
    assert len(pooled) == len(direct)
    for direct_emb, pooled_emb in zip(direct, pooled):
        assert torch.allclose(direct_emb, pooled_emb, atol=1e-5)


def test_multi_process_default_token_pooling(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = MultiVectorEncoder(
        modules=list(mve_bert_tiny_model), token_pooling=HierarchicalTokenPooling(max_tokens=4), device="cpu"
    )
    pool = model.start_multi_process_pool(["cpu", "cpu"])
    try:
        for kwargs in [{}, {"token_pooling": None}, {"token_pooling": HierarchicalTokenPooling(max_tokens=8)}]:
            direct = model.encode_document(TEXTS, **kwargs)
            multi_process = model.encode_document(TEXTS, pool=pool, **kwargs)
            for expected, actual in zip(direct, multi_process):
                torch.testing.assert_close(actual, expected)
    finally:
        model.stop_multi_process_pool(pool)


def test_multi_process_output_value_none(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = mve_bert_tiny_model
    model.to("cpu")
    direct = model.encode_document(TEXTS, output_value=None)
    pooled = model.encode_document(TEXTS, output_value=None, device=["cpu", "cpu"])
    assert len(pooled) == len(direct)
    for direct_item, pooled_item in zip(direct, pooled):
        assert sorted(direct_item.keys()) == sorted(pooled_item.keys())
        # The worker moves dict values to CPU before crossing the process boundary.
        assert pooled_item["token_embeddings"].device.type == "cpu"
        # Raw outputs are padded to their own batch's longest input, and batch composition differs
        # between the direct and the chunked run: compare the mask-sliced tokens instead.
        direct_tokens = direct_item["token_embeddings"][direct_item["attention_mask"].bool()]
        pooled_tokens = pooled_item["token_embeddings"][pooled_item["attention_mask"].bool()]
        assert torch.allclose(direct_tokens, pooled_tokens, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available to experiment with 2 separate devices"
)
def test_multi_process_output_tensors_two_devices(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    # Both result shapes are lists here, and a device tensor would only stay readable for as long
    # as the worker that produced it is alive
    model = mve_bert_tiny_model
    model.to("cpu")
    embeddings = model.encode_document(TEXTS, device=["cpu", "cuda"])
    assert len(embeddings) == len(TEXTS)
    assert all(emb.device.type == "cpu" for emb in embeddings)

    features = model.encode_document(TEXTS, output_value=None, device=["cpu", "cuda"])
    assert len(features) == len(TEXTS)
    for feature in features:
        assert all(value.device.type == "cpu" for value in feature.values() if isinstance(value, torch.Tensor))
