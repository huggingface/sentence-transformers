from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.batch_hard_soft_margin_triplet import (
    BatchHardSoftMarginTripletLoss,
)
from sentence_transformers.sentence_transformer.losses.batch_hard_triplet import BatchHardTripletLossDistanceFunction
from sentence_transformers.sentence_transformer.losses.batch_semi_hard_triplet import BatchSemiHardTripletLoss
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


def _semi_hard_reference(embeddings, labels, distance_metric, margin):
    """Select each negative directly, without the production mining implementation."""
    distances = distance_metric(embeddings)
    losses = []
    for anchor in range(len(labels)):
        negatives = distances[anchor, labels != labels[anchor]]
        for positive in range(len(labels)):
            if anchor == positive or labels[anchor] != labels[positive]:
                continue
            farther = negatives[negatives > distances[anchor, positive]]
            if len(farther):
                negative = farther.min(dim=0).values
            elif len(negatives):
                negative = negatives.max(dim=0).values
            else:
                # Preserve the existing masked-maximum fallback for a single-class batch.
                negative = distances[anchor].min(dim=0).values
            losses.append((distances[anchor, positive] - negative + margin).clamp(min=0))
    return torch.stack(losses).mean() if losses else distances.sum() * 0


@pytest.mark.parametrize("case", ["random", "ties", "no_positives", "no_negatives"])
@pytest.mark.parametrize(
    "distance_metric",
    [BatchHardTripletLossDistanceFunction.euclidean_distance, BatchHardTripletLossDistanceFunction.cosine_distance],
    ids=["euclidean", "cosine"],
)
def test_semi_hard_triplet_matches_reference(dummy_model, case, distance_metric):
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(8, 4, generator=generator)
    labels = torch.arange(8) // 2
    if case == "ties":
        embeddings = torch.tensor([[v, 1.0] for v in (0, 1, 2, 2, 3, 4, 5, 7)])
    elif case == "no_positives":
        labels = torch.arange(8)
    elif case == "no_negatives":
        labels = torch.zeros(8, dtype=torch.long)
    actual_embeddings = embeddings.clone().requires_grad_()
    expected_embeddings = embeddings.clone().requires_grad_()
    loss_fn = BatchSemiHardTripletLoss(dummy_model, distance_metric=distance_metric, margin=0.7)
    actual = loss_fn.compute_loss_from_embeddings([actual_embeddings], labels)
    expected = _semi_hard_reference(expected_embeddings, labels, distance_metric, margin=0.7)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_embeddings.grad, expected_embeddings.grad, atol=1e-6, rtol=1e-5)


def test_semi_hard_triplet_saved_tensors_are_quadratic(dummy_model):
    """Mining must not retain a cubic distance tile or mask for backward."""
    batch_size = 32
    embeddings = torch.randn(batch_size, 8, requires_grad=True)
    labels = torch.arange(batch_size) // 2
    saved_sizes = []

    def pack(tensor):
        saved_sizes.append(tensor.numel())
        return tensor

    loss_fn = BatchSemiHardTripletLoss(dummy_model)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert max(saved_sizes) <= batch_size**2


def test_semi_hard_triplet_tied_maximum_negative_gradients(dummy_model):
    """A negative tied with the row maximum must receive its own gradient."""
    embeddings = torch.tensor([[0.0], [2.0], [1.0], [-2.0], [-2.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1])
    loss_fn = BatchSemiHardTripletLoss(dummy_model, margin=5)

    loss = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(3.25))
    torch.testing.assert_close(embeddings.grad, torch.tensor([[-1.0], [0.25], [-0.25], [0.875], [0.125]]))


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
