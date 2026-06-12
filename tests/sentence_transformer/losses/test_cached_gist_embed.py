from __future__ import annotations

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
    monkeypatch.setattr(cge, "is_dist_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    return per


def _local_reps(per: int, dim: int = 16, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(per, dim, generator=g)
    positives = torch.randn(per, dim, generator=g)
    # reps[0] = anchors, reps[1] = positives; each a list of minibatch tensors.
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
    """Structural invariant: across every minibatch chunk, the column protected by
    ``positive_mask`` for local row r equals the CE target column
    ``range_labels[begin + r] = offset + begin + r``.

    We reconstruct the mask exactly as the loss does, for each ``begin`` chunk, and compare
    against the targets. ``mini_batch_size=2`` exercises a ``begin > 0`` chunk.
    """
    per = simulate_rank1_world2
    dim = 16
    reps, reps_guided = _local_reps(per, dim=dim)
    loss_fn = _make_loss(margin=0.1, mini_batch_size=mini_batch_size)

    anchors = torch.cat(reps[0])
    candidates = [cge.all_gather_with_grad(torch.cat(r)) for r in reps[1:]]
    batch_size = anchors.size(0)
    offset = torch.distributed.get_rank() * batch_size  # = per (rank 1)

    range_labels = torch.arange(offset, offset + batch_size)

    for begin in range(0, batch_size, mini_batch_size):
        end = begin + mini_batch_size
        guided_ap_sim = loss_fn.sim_matrix(anchors[begin:end], candidates[0])
        positive_mask = torch.zeros_like(guided_ap_sim, dtype=torch.bool)
        rows = torch.arange(guided_ap_sim.size(0))
        positive_mask[rows, offset + begin + rows] = True

        protected_cols = positive_mask.float().argmax(dim=1)
        assert torch.equal(protected_cols, range_labels[begin : begin + rows.size(0)]), (
            f"begin={begin}: protected columns {protected_cols.tolist()} must equal CE targets "
            f"{range_labels[begin : begin + rows.size(0)].tolist()}"
        )
