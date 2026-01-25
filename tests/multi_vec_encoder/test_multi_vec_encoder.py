from __future__ import annotations

import pytest
import torch

from sentence_transformers.multi_vec_encoder import LateInteractionPooling
from sentence_transformers.multi_vec_encoder.similarity import maxsim, maxsim_pairwise


class TestLateInteractionPooling:
    """Tests for LateInteractionPooling module."""

    @pytest.mark.parametrize(
        ("word_dim", "output_dim", "expected_output_dim"),
        [
            (768, None, 768),
            (768, 768, 768),
            (768, 128, 128),
            (128, 64, 64),
        ],
    )
    def test_dimensions(self, word_dim: int, output_dim: int | None, expected_output_dim: int) -> None:
        """Test that dimension projection works correctly."""
        pooling = LateInteractionPooling(
            word_embedding_dimension=word_dim,
            output_dimension=output_dim,
            normalize=False,
        )
        features = {
            "token_embeddings": torch.randn(2, 5, word_dim),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        }
        output = pooling(features)

        assert output["token_embeddings"].shape == (2, 5, expected_output_dim)
        assert pooling.get_output_dimension() == expected_output_dim

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization(self, normalize: bool) -> None:
        """Test that L2 normalization produces unit vectors when enabled."""
        pooling = LateInteractionPooling(word_embedding_dimension=64, normalize=normalize)
        features = {
            "token_embeddings": torch.randn(2, 5, 64) * 10,
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        }
        output = pooling(features)
        norms = torch.norm(output["token_embeddings"], dim=-1)

        if normalize:
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        else:
            assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize(
        ("skip_cls", "skip_sep", "attention_mask", "expected_mask"),
        [
            (True, False, [[1, 1, 1, 0]], [[0, 1, 1, 0]]),  # CLS at 0 masked
            (False, True, [[1, 1, 1, 0]], [[1, 1, 0, 0]]),  # SEP at 2 masked (last non-padding)
            (True, True, [[1, 1, 1, 0]], [[0, 1, 0, 0]]),  # CLS at 0, SEP at 2 masked
            (False, False, [[1, 1, 1, 0]], [[1, 1, 1, 0]]),  # No masking
            (True, True, [[1, 1, 1, 1]], [[0, 1, 1, 0]]),  # CLS at 0, SEP at 3 masked
        ],
    )
    def test_skip_tokens(self, skip_cls: bool, skip_sep: bool, attention_mask: list, expected_mask: list) -> None:
        """Test that CLS/SEP token skipping modifies attention mask correctly."""
        pooling = LateInteractionPooling(
            word_embedding_dimension=32, skip_cls_token=skip_cls, skip_sep_token=skip_sep, normalize=False
        )
        features = {
            "token_embeddings": torch.randn(1, len(attention_mask[0]), 32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        output = pooling(features)

        assert torch.equal(output["attention_mask"], torch.tensor(expected_mask, dtype=torch.long))

    def test_config_roundtrip(self, tmp_path) -> None:
        """Test that config save/load preserves all settings."""
        pooling = LateInteractionPooling(
            word_embedding_dimension=768,
            output_dimension=128,
            normalize=True,
            skip_cls_token=True,
            skip_sep_token=False,
        )
        pooling.save(str(tmp_path))
        loaded = LateInteractionPooling.load(str(tmp_path))

        assert (loaded.word_embedding_dimension, loaded.output_dimension) == (768, 128)
        assert (loaded.normalize, loaded.skip_cls_token, loaded.skip_sep_token) == (True, True, False)

    def test_get_sentence_embedding_dimension_returns_none(self) -> None:
        """Test that get_sentence_embedding_dimension returns None for multi-vector models."""
        assert LateInteractionPooling(word_embedding_dimension=768).get_sentence_embedding_dimension() is None


class TestMaxSimSimilarity:
    """Tests for MaxSim similarity functions."""

    @pytest.mark.parametrize(
        ("batch_q", "batch_d", "num_q_tokens", "num_d_tokens", "dim"),
        [(1, 1, 3, 5, 64), (1, 2, 4, 6, 128), (2, 3, 5, 10, 64), (3, 3, 8, 8, 32)],
    )
    def test_output_shape(self, batch_q: int, batch_d: int, num_q_tokens: int, num_d_tokens: int, dim: int) -> None:
        """Test that maxsim returns correct output shape [batch_q, batch_d]."""
        scores = maxsim(torch.randn(batch_q, num_q_tokens, dim), torch.randn(batch_d, num_d_tokens, dim))
        assert scores.shape == (batch_q, batch_d)

    @pytest.mark.parametrize(
        ("query_mask", "doc_mask", "expected_score"),
        [
            (None, None, None),  # No masks, just check no NaN
            ([[1, 1, 0]], None, None),
            (None, [[1, 1, 1, 0, 0]], None),
            ([[1, 1, 0]], [[1, 1, 1, 0, 0]], None),
            ([[0, 0, 0]], None, 0.0),  # All query tokens masked -> score = 0
        ],
    )
    def test_with_masks(self, query_mask: list | None, doc_mask: list | None, expected_score: float | None) -> None:
        """Test that maxsim handles attention masks correctly."""
        query = torch.randn(1, 3, 64)
        doc = torch.randn(1, 5, 64)
        q_mask = torch.tensor(query_mask, dtype=torch.long) if query_mask else None
        d_mask = torch.tensor(doc_mask, dtype=torch.long) if doc_mask else None

        scores = maxsim(query, doc, q_mask, d_mask)

        assert scores.shape == (1, 1) and not torch.isnan(scores).any()
        if expected_score is not None:
            assert scores.item() == expected_score

    def test_masked_tokens_ignored(self) -> None:
        """Test that masked document tokens don't contribute to similarity."""
        query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        doc = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]])

        scores_unmasked = maxsim(query, doc)
        scores_masked = maxsim(query, doc, document_mask=torch.tensor([[1, 0]]))

        assert torch.allclose(scores_unmasked, scores_masked, atol=1e-5)

    @pytest.mark.parametrize(
        ("batch_size", "num_q_tokens", "num_d_tokens", "dim"),
        [(1, 3, 5, 64), (2, 4, 6, 128), (5, 8, 10, 32)],
    )
    def test_pairwise_output_shape(self, batch_size: int, num_q_tokens: int, num_d_tokens: int, dim: int) -> None:
        """Test that maxsim_pairwise returns correct output shape [batch]."""
        scores = maxsim_pairwise(
            torch.randn(batch_size, num_q_tokens, dim), torch.randn(batch_size, num_d_tokens, dim)
        )
        assert scores.shape == (batch_size,)

    def test_pairwise_batch_mismatch_raises(self) -> None:
        """Test that mismatched batch sizes raise an AssertionError."""
        with pytest.raises(AssertionError, match="Batch sizes must match"):
            maxsim_pairwise(torch.randn(2, 4, 64), torch.randn(3, 5, 64))

    def test_pairwise_consistency_with_maxsim(self) -> None:
        """Test that pairwise scores match diagonal of full maxsim matrix."""
        query, doc = torch.randn(3, 4, 64), torch.randn(3, 5, 64)
        assert torch.allclose(maxsim_pairwise(query, doc), torch.diagonal(maxsim(query, doc)), atol=1e-5)

    @pytest.mark.parametrize(
        ("query", "doc", "expected"),
        [
            (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]]), 1.0),  # Identical
            (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]), 0.0),  # Orthogonal
            (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[-1.0, 0.0]]]), -1.0),  # Opposite
            (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), 0.0),  # All zeros
            (
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                2.0,
            ),  # Multi-token sum
        ],
    )
    def test_edge_cases(self, query: torch.Tensor, doc: torch.Tensor, expected: float) -> None:
        """Test maxsim with edge cases: identical, orthogonal, opposite vectors, zeros, multi-token."""
        assert torch.allclose(maxsim(query, doc), torch.tensor([[expected]]), atol=1e-5)
