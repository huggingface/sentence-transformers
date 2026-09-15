from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.batch_hard_soft_margin_triplet import (
    BatchHardSoftMarginTripletLoss,
)
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


@pytest.mark.parametrize("separation", [10.0, 1000.0])
def test_batch_hard_soft_margin_finite_loss_and_gradients(dummy_model, separation):
    embeddings = torch.tensor(
        [[0.0], [separation], [2.0], [separation + 4.0]], dtype=torch.float32, requires_grad=True
    )
    reference_embeddings = embeddings.detach().double().requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = BatchHardSoftMarginTripletLoss(dummy_model)

    loss = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    loss.backward()

    # Enumerate valid pairs independently, using double precision as the oracle.
    terms = []
    for anchor in range(len(labels)):
        positive = [
            torch.linalg.vector_norm(reference_embeddings[anchor] - reference_embeddings[other])
            for other in range(len(labels))
            if anchor != other and labels[anchor] == labels[other]
        ]
        negative = [
            torch.linalg.vector_norm(reference_embeddings[anchor] - reference_embeddings[other])
            for other in range(len(labels))
            if labels[anchor] != labels[other]
        ]
        margin = torch.stack(positive).max() - torch.stack(negative).min()
        terms.append(torch.logaddexp(torch.zeros_like(margin), margin))
    reference_loss = torch.stack(terms).mean()
    reference_loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()
    torch.testing.assert_close(loss, reference_loss.to(embeddings.dtype))
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad.to(embeddings.dtype))
