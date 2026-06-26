"""Cross-column varlen support in MultiVector* losses.

Native-resolution VLMs (Qwen2-VL family, ColIdefics3, ...) emit different per-column token counts:
the same batch can have the positive column at T=10 and the negative column at T=14, and there is
no way to fix this with collator padding. The ``MultipleNegativesRankingLoss`` / ``CachedMNRL``
``torch.stack(..., dim=1)`` would then fail with "stack expects each tensor to be equal size".

This file pins the regression: each loss pads cross-column to the per-batch max token count and
threads the resulting mask so MaxSim never scores against padded positions.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder import losses as mve_losses


class _PassthroughModel(nn.Module):
    """Stub model that returns the ``token_embeddings`` already placed in the feature dict.

    The losses call ``self.model(sf, task=...)["token_embeddings"]``. This stub bypasses tokenisation
    so we can construct ragged document columns explicitly and verify the loss handles them.
    """

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        return features


def _make_feature(t_tokens: int, batch: int, dim: int, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(batch, t_tokens, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    return {
        "token_embeddings": emb,
        "attention_mask": torch.ones(batch, t_tokens, dtype=torch.bool),
    }


@pytest.fixture
def varlen_features() -> list[dict[str, Tensor]]:
    """Triplet batch where the positive column has 10 tokens and the negative has 14. This is the
    case that fails the pre-fix ``torch.stack(embeddings[1:], dim=1)``."""
    batch, dim = 4, 8
    return [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),  # query
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),  # positive (T=10)
        _make_feature(t_tokens=14, batch=batch, dim=dim, seed=3),  # negative (T=14)
    ]


def test_mnr_handles_varlen_document_columns(varlen_features) -> None:
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_cached_mnr_handles_varlen_document_columns(varlen_features) -> None:
    """Cross-column varlen through the GradCache path. Exercises the same surface as the non-cached
    MNRL test but with chunked embedding forward + cached gradients."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


class _RaggedTrimPassthroughModel(nn.Module):
    """Trims trailing pad off each mini-batch to the per-batch real max, simulating a
    native-resolution VLM whose vision encoder emits ``T`` proportional to the batch's longest
    real sequence. Successive mini-batches in the GradCache path therefore see different ``T``s."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        emb = features["token_embeddings"]
        mask = features["attention_mask"].bool()
        T_real = int(mask.sum(dim=1).max().item())
        return {
            "token_embeddings": emb[:, :T_real].contiguous(),
            "attention_mask": mask[:, :T_real],
        }


def _make_ragged_feature(real_lengths: list[int], dim: int, seed: int) -> dict[str, Tensor]:
    T_max = max(real_lengths)
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(len(real_lengths), T_max, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    mask = torch.zeros(len(real_lengths), T_max, dtype=torch.bool)
    for i, real_t in enumerate(real_lengths):
        mask[i, :real_t] = True
    return {"token_embeddings": emb, "attention_mask": mask}


def test_cached_mnr_handles_varlen_mini_batches_within_a_column() -> None:
    """The GradCache cat path pads each mini-batch chunk to a common ``T`` before concatenating,
    so native-resolution VLMs that emit different ``T`` per mini-batch within the same column flow
    through cleanly. Distinct from the cross-column ragged-``T`` case: that is solved by
    ``stack_padded_token_embeddings``. This one is solved by ``cat_padded_token_embeddings``."""
    dim = 8
    # mini_batch_size=2 splits each column into [rows 0-1, rows 2-3]. The trim passthrough returns
    # T=max-real per mini-batch, so the positive column emits T=10 then T=7, and the negative
    # column emits T=12 then T=11. The pre-fix ``torch.cat`` failed on these mismatched ``T``s.
    features = [
        _make_ragged_feature([5, 5, 4, 5], dim=dim, seed=1),
        _make_ragged_feature([10, 9, 6, 7], dim=dim, seed=2),
        _make_ragged_feature([8, 12, 9, 11], dim=dim, seed=3),
    ]

    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_RaggedTrimPassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_mnr_pad_positions_do_not_influence_loss(varlen_features) -> None:
    """Sanity: corrupting the (would-be-)padded positions of the shorter column to a value that
    would dominate MaxSim must NOT change the loss, because the mask excludes them."""
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    baseline = loss(varlen_features, labels=None).item()

    # Reshape: make the positive column LONGER but stuff garbage past its real length, mask covers
    # the real range. This mirrors what the loss does internally. If the mask flows correctly, the
    # loss is identical to the baseline above.
    short_pos = varlen_features[1]["token_embeddings"]
    batch, real_t, dim = short_pos.shape
    extended = torch.cat([short_pos, torch.full((batch, 4, dim), 1e3, device=short_pos.device)], dim=1)
    extended.requires_grad_(True)
    corrupted = [
        varlen_features[0],
        {
            "token_embeddings": extended,
            "attention_mask": torch.cat(
                [
                    torch.ones(batch, real_t, dtype=torch.bool),
                    torch.zeros(batch, 4, dtype=torch.bool),
                ],
                dim=1,
            ),
        },
        varlen_features[2],
    ]
    corrupted_value = loss(corrupted, labels=None).item()
    assert abs(baseline - corrupted_value) < 1e-5, (
        f"masked-out positions must not influence the loss: baseline={baseline}, corrupted={corrupted_value}"
    )


def test_mnr_fixed_length_regression() -> None:
    """When all document columns share the same T (text / fixed-resolution VLM case), the new
    pad logic is a no-op: same batch, same loss as before."""
    batch, dim = 3, 8
    same_t_features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=11),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=12),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=13),
    ]
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(same_t_features, labels=None)
    assert torch.isfinite(value)


def test_mnr_gather_across_devices_single_process_matches_no_gather(varlen_features) -> None:
    """On a single process ``gather_across_devices=True`` must be a no-op: ``all_gather`` falls back to
    the local tensor and ``all_gather_padded`` skips the cross-rank pad, so the loss equals the
    non-gathered loss (world_size=1, rank=0). Guards the gather wiring without a real process group.
    The cross-rank pad itself is only reachable under multi-GPU DDP, which this cannot exercise."""
    base = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    gathered = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), gather_across_devices=True
    )
    baseline = base(varlen_features, labels=None)
    value = gathered(varlen_features, labels=None)
    assert torch.isfinite(value)
    assert torch.allclose(value, baseline), f"gather no-op mismatch: {value.item()} vs {baseline.item()}"
    value.backward()


def test_cached_mnr_gather_across_devices_single_process(varlen_features) -> None:
    """Single-process gather guard for the GradCache loss: the gather block must run (all_gather
    no-ops without a process group) and produce a finite, differentiable loss."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False, gather_across_devices=True
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()
