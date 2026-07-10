from __future__ import annotations

import math

import pytest
import torch

import sentence_transformers.sentence_transformer.losses.cached_gist_embed as cge
from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss


def _make_loss(margin: float, mini_batch_size: int = 32) -> CachedGISTEmbedLoss:
    """Build a CachedGISTEmbedLoss without loading real models.

    ``calculate_loss`` only needs the configuration attributes and operates on
    precomputed reps, so we bypass ``__init__`` (which requires SentenceTransformers).
    """
    obj = CachedGISTEmbedLoss.__new__(CachedGISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.mini_batch_size = mini_batch_size
    obj.show_progress_bar = False
    obj.margin_strategy = "absolute"
    obj.margin = margin
    obj.contrast_anchors = True
    obj.contrast_positives = True
    obj.gather_across_devices = True
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    return obj


@pytest.fixture
def simulate_rank1_world2(monkeypatch):
    """Monkeypatch the distributed helpers so ``calculate_loss`` runs the rank=1/world=2
    gather path in a single process: ``all_gather_with_grad`` prepends a rank-0 block to
    the local (rank-1) block, and the rank is reported as 1 so ``offset = batch_size``.
    """
    per = 3

    def fake_gather(tensor: torch.Tensor) -> torch.Tensor:
        # rank 0's block is deterministic and distinct from the local (rank 1) block.
        g = torch.Generator().manual_seed(1234)
        rank0_block = torch.randn(per, tensor.size(1), generator=g)
        return torch.cat([rank0_block, tensor], dim=0)

    monkeypatch.setattr(cge, "all_gather_with_grad", fake_gather)
    monkeypatch.setattr(cge, "get_rank", lambda: 1)
    return per


def _local_reps(per: int, dim: int = 16, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(per, dim, generator=g)
    positives = torch.randn(per, dim, generator=g)
    # reps[0] = anchors, reps[1] = positives. Each a list of minibatch tensors.
    reps = [[anchors.clone()], [positives.clone()]]
    reps_guided = [[anchors.clone()], [positives.clone()]]
    return reps, reps_guided


@pytest.mark.parametrize("mini_batch_size", [32, 2])
def test_gather_rank1_positive_not_masked_with_margin(simulate_rank1_world2, mini_batch_size):
    """Regression for the rank>0 positive-mask offset bug (gather_across_devices).

    On rank 1, each anchor's true positive lives at gathered column ``offset + row``.
    With ``margin > 0`` the positive itself exceeds ``positive_sim - margin``, so it is
    only spared by ``positive_mask``. If the mask is offset-unaware (the old
    ``roll(begin)``), it protects the wrong columns, the CE-target logit becomes -inf,
    and the loss is +inf. With the offset-aware mask the loss stays finite.

    ``mini_batch_size=2`` splits the local batch so the inner loop runs with ``begin > 0``,
    covering the minibatch-boundary case (where the old flatten-roll could also wrap).
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss = _make_loss(margin=0.1, mini_batch_size=mini_batch_size).calculate_loss(reps, reps_guided)
    assert torch.isfinite(loss), f"rank>0 loss must be finite, got {loss.item()}"


def test_gather_rank1_margin_zero_baseline(simulate_rank1_world2):
    """Sanity baseline: with margin=0 the positive equals its own threshold and is not
    suppressed regardless of the mask, so the loss is finite both before and after the fix.
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss = _make_loss(margin=0.0).calculate_loss(reps, reps_guided)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mini_batch_size", [32, 2])
def test_positive_mask_protects_ce_target_column(simulate_rank1_world2, mini_batch_size):
    """Drive the real loss and assert it protects exactly the CE-target column in every minibatch.

    We capture each minibatch's (masked) score matrix and labels as passed to cross-entropy and
    check that every anchor's CE-target logit survived suppression (is finite). With the old
    offset-unaware ``roll(begin)`` mask, on rank 1 that column is set to -inf, so this fails on the
    pre-fix code. ``mini_batch_size=2`` exercises a ``begin > 0`` chunk.
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss_fn = _make_loss(margin=0.1, mini_batch_size=mini_batch_size)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        captured.append((scores, labels))
        return real_ce(scores, labels)

    # cross_entropy_loss is registered as an nn.Module child, so drop it before assigning a plain callable.
    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn.calculate_loss(reps, reps_guided)

    assert captured, "cross-entropy was never called"
    for scores, labels in captured:
        target_logits = scores[torch.arange(scores.size(0)), labels]
        assert torch.isfinite(target_logits).all(), (
            f"each anchor's CE-target logit must survive masking, got {target_logits.tolist()}"
        )
    assert torch.isfinite(loss)


def _make_relative_loss(margin: float) -> CachedGISTEmbedLoss:
    """relative-margin CachedGISTEmbedLoss fed precomputed embeddings (#3819)."""
    obj = CachedGISTEmbedLoss.__new__(CachedGISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.mini_batch_size = 32
    obj.show_progress_bar = False
    obj.margin_strategy = "relative"
    obj.margin = margin
    obj.contrast_anchors = False
    obj.contrast_positives = False
    obj.gather_across_devices = False
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    return obj


def _negative_score_reps():
    """anchor0's positive has a negative cosine (-0.50). A non-paired candidate (column 1) is
    *more* similar (-0.49). anchor1 is paired with that candidate (diagonal cosine 1.0)."""
    a0 = [1.0, 0.0, 0.0]
    p0 = [-0.50, math.sqrt(1 - 0.50**2), 0.0]  # cos(a0, p0) = -0.50
    p1 = [-0.49, 0.0, math.sqrt(1 - 0.49**2)]  # cos(a0, p1) = -0.49 (more similar than the positive)
    a1 = p1  # cos(a1, p1) = 1.0
    anchors = torch.tensor([a0, a1])
    positives = torch.tensor([p0, p1])
    reps = [[anchors], [positives]]
    reps_guided = [[anchors], [positives]]
    return reps, reps_guided


def test_relative_margin_negative_positive_score_suppresses_closer_negative():
    """Regression for #3819 in CachedGISTEmbedLoss: with ``margin_strategy="relative"`` and a
    negative positive-pair score, a candidate more similar to the anchor than the positive must
    still be suppressed."""
    loss_fn = _make_relative_loss(margin=0.05)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        captured.append(scores)
        return real_ce(scores, labels)

    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn.calculate_loss(*_negative_score_reps())

    assert captured, "cross-entropy was never called"
    scores = captured[0]
    assert scores[0, 1].item() == float("-inf"), f"closer negative must be masked, got {scores[0, 1].item()}"
    assert torch.isfinite(scores[0, 0]) and torch.isfinite(loss)
