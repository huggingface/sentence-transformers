from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder

TEXTS = [
    "short",
    "a somewhat longer sentence with a few more tokens in it",
    "medium length text here",
    "another document",
]


@pytest.fixture(scope="module")
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")


def test_multi_process_matches_single_process(model: MultiVectorEncoder) -> None:
    direct = model.encode_document(TEXTS, convert_to_tensor=True)
    pooled = model.encode_document(TEXTS, convert_to_tensor=True, device=["cpu", "cpu"])
    assert len(pooled) == len(direct)
    for direct_emb, pooled_emb in zip(direct, pooled):
        assert torch.allclose(direct_emb, pooled_emb, atol=1e-5)


def test_multi_process_convert_to_padded_tensor(model: MultiVectorEncoder) -> None:
    # Padding happens once after merging chunks: per-chunk max lengths must not leak into the shape.
    direct = model.encode_document(TEXTS, convert_to_padded_tensor=True)
    pooled = model.encode_document(TEXTS, convert_to_padded_tensor=True, device=["cpu", "cpu"])
    assert pooled.shape == direct.shape
    assert torch.allclose(direct, pooled, atol=1e-5)


def test_multi_process_output_value_none(model: MultiVectorEncoder) -> None:
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
