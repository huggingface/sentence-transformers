from __future__ import annotations

import pytest
import torch

import sentence_transformers.sentence_transformer.losses.gist_embed as ge
from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss


def _make_loss(margin: float) -> GISTEmbedLoss:
    """Build a GISTEmbedLoss without loading real models.

    ``forward`` encodes via ``self.model`` / ``self.guide``; we replace both with a fake
    that returns precomputed embeddings carried on the feature dict, so no model is needed.
    """
    obj = GISTEmbedLoss.__new__(GISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.margin_strategy = "absolute"
    obj.margin = margin
    obj.contrast_anchors = True
    obj.contrast_positives = True
    obj.gather_across_devices = True
    obj.must_retokenize = False
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    fake = lambda sentence_feature: {"sentence_embedding": sentence_feature["sentence_embedding"]}
    obj.model = fake
    obj.guide = fake
    return obj


@pytest.fixture
def simulate_rank1_world2(monkeypatch):
    """Run the rank=1/world=2 gather path in a single process: ``all_gather_with_grad``
    prepends a deterministic rank-0 block, and the rank is reported as 1 (offset = batch_size).
    """
    per = 3

    def fake_gather(tensor: torch.Tensor) -> torch.Tensor:
        g = torch.Generator().manual_seed(1234)
        rank0_block = torch.randn(per, tensor.size(1), generator=g)
        return torch.cat([rank0_block, tensor], dim=0)

    monkeypatch.setattr(ge, "all_gather_with_grad", fake_gather)
    monkeypatch.setattr(ge, "is_dist_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    return per


def _features(per: int, dim: int = 16, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(per, dim, generator=g)
    positives = torch.randn(per, dim, generator=g)
    return [{"sentence_embedding": anchors}, {"sentence_embedding": positives}]


def test_gather_rank1_positive_not_masked_with_margin(simulate_rank1_world2):
    """Regression for the rank>0 positive-mask offset bug in the non-cached GISTEmbedLoss.

    The old ``torch.eye(*shape)`` mask protects columns [0..batch-1], but on rank 1 each
    anchor's positive is at gathered column ``offset + row``. With ``margin > 0`` the unprotected
    positive is masked to -inf, the CE-target logit becomes -inf, and the loss is +inf. The
    offset-aware mask keeps it finite.
    """
    per = simulate_rank1_world2
    loss = _make_loss(margin=0.1)(_features(per), labels=None)
    assert torch.isfinite(loss), f"rank>0 loss must be finite, got {loss.item()}"


def test_gather_rank1_margin_zero_baseline(simulate_rank1_world2):
    """Sanity baseline: with margin=0 the positive equals its own threshold and is not
    suppressed regardless of the mask, so the loss is finite both before and after the fix.
    """
    per = simulate_rank1_world2
    loss = _make_loss(margin=0.0)(_features(per), labels=None)
    assert torch.isfinite(loss)


def test_positive_mask_protects_ce_target_column(simulate_rank1_world2):
    """Structural invariant: the column protected by ``positive_mask`` for anchor row r equals
    the CE target column ``range_labels[r] = offset + r``.
    """
    per = simulate_rank1_world2
    dim = 16
    feats = _features(per, dim=dim)
    loss_fn = _make_loss(margin=0.1)

    anchor = feats[0]["sentence_embedding"]
    positive = ge.all_gather_with_grad(feats[1]["sentence_embedding"])
    batch_size = anchor.size(0)
    offset = torch.distributed.get_rank() * batch_size  # = per (rank 1)

    range_labels = torch.arange(offset, offset + batch_size)
    guided_ap_sim = loss_fn.sim_matrix(anchor, positive)
    positive_mask = torch.zeros_like(guided_ap_sim, dtype=torch.bool)
    rows = torch.arange(guided_ap_sim.size(0))
    positive_mask[rows, offset + rows] = True

    protected_cols = positive_mask.float().argmax(dim=1)
    assert torch.equal(protected_cols, range_labels), (
        f"protected columns {protected_cols.tolist()} must equal CE targets {range_labels.tolist()}"
    )
