from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.triplet import TripletDistanceMetric, TripletLoss


@pytest.fixture
def dummy_model():
    class DummyModel:
        pass

    return DummyModel()


@pytest.mark.parametrize(
    "distance_metric",
    [TripletDistanceMetric.COSINE, TripletDistanceMetric.EUCLIDEAN, TripletDistanceMetric.MANHATTAN],
    ids=["cosine", "euclidean", "manhattan"],
)
def test_triplet_loss_correct_direction(dummy_model, distance_metric):
    """Loss should be lower when the positive is closer to the anchor than the negative."""
    loss_fn = TripletLoss(model=dummy_model, distance_metric=distance_metric, triplet_margin=1.0)

    anchor = torch.tensor([[1.0, 0.0, 0.0]])
    positive = torch.tensor([[0.9, 0.1, 0.0]])  # close to anchor
    negative = torch.tensor([[0.0, 1.0, 0.0]])  # far from anchor

    # Good triplet: positive is closer than negative → should yield low/zero loss
    good_loss = loss_fn.compute_loss_from_embeddings([anchor, positive, negative], labels=None)

    # Bad triplet: swap positive and negative → should yield higher loss
    bad_loss = loss_fn.compute_loss_from_embeddings([anchor, negative, positive], labels=None)

    assert good_loss < bad_loss, (
        f"Good triplet loss ({good_loss:.4f}) should be less than bad triplet loss ({bad_loss:.4f})"
    )
