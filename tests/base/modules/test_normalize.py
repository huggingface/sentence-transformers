from __future__ import annotations

import pytest
import torch

from sentence_transformers.base.modules import Normalize


@pytest.mark.parametrize("input_name,output_name", [("sentence_embedding", None), ("token_embeddings", "normalized")])
@pytest.mark.parametrize("token_level", [False, True])
def test_normalize_float16_zero_small_and_large_rows(input_name, output_name, token_level) -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1e-7, 1e-7], [60000.0, 60000.0], [3.0, 4.0]], dtype=torch.float16)
    if token_level:
        embeddings = embeddings.unsqueeze(0)
    original = embeddings.clone()
    attention_mask = torch.ones(embeddings.shape[:-1], dtype=torch.long)
    features = {input_name: embeddings, "attention_mask": attention_mask}
    expected = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1).to(embeddings.dtype)

    result = Normalize(module_input_name=input_name, module_output_name=output_name)(features)
    normalized = result[output_name or input_name]

    assert result is features
    assert normalized.dtype == embeddings.dtype
    assert normalized.device == embeddings.device
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    torch.testing.assert_close(embeddings, original, rtol=0, atol=0)
    assert result["attention_mask"] is attention_mask


def test_normalize_missing_input_is_unchanged() -> None:
    features = {"attention_mask": torch.ones(2, 3)}

    assert Normalize()(features) is features
    assert "sentence_embedding" not in features


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.complex64])
def test_normalize_preserves_other_dtypes(dtype: torch.dtype) -> None:
    embeddings = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-5.0, 12.0]], dtype=dtype)
    expected = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    result = Normalize()({"sentence_embedding": embeddings})["sentence_embedding"]

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
