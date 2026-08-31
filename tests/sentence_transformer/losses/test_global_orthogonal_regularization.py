from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import GlobalOrthogonalRegularizationLoss
from sentence_transformers.util import cos_sim


@pytest.fixture
def dummy_model():
    class DummyModel:
        pass

    return DummyModel()


@pytest.fixture
def loss(dummy_model) -> GlobalOrthogonalRegularizationLoss:
    return GlobalOrthogonalRegularizationLoss(dummy_model)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_compute_gor_is_finite_for_every_batch_size(loss: GlobalOrthogonalRegularizationLoss, batch_size: int) -> None:
    """A batch of one has no off-diagonal pairs, which used to divide by zero and return nan."""
    torch.manual_seed(0)
    embeddings = torch.randn(batch_size, 8)

    mean_term, second_moment_term = loss.compute_gor(embeddings)

    assert torch.isfinite(mean_term), f"mean term is {mean_term} for batch_size={batch_size}"
    assert torch.isfinite(second_moment_term), f"second moment term is {second_moment_term}"


@pytest.mark.parametrize("batch_size", [2, 3, 4, 8])
def test_compute_gor_matches_the_off_diagonal_definition(
    loss: GlobalOrthogonalRegularizationLoss, batch_size: int
) -> None:
    """Guarding the batch-of-one case must leave batches that do have pairs untouched."""
    torch.manual_seed(batch_size)
    embeddings = torch.randn(batch_size, 16)

    mean_term, second_moment_term = loss.compute_gor(embeddings)

    off_diagonal = cos_sim(embeddings, embeddings)[~torch.eye(batch_size, dtype=torch.bool)]
    assert mean_term.item() == pytest.approx(off_diagonal.mean().pow(2).item(), abs=1e-7)
    assert second_moment_term.item() == pytest.approx(max(off_diagonal.pow(2).mean().item() - 1 / 16, 0.0), abs=1e-7)


def test_single_input_contributes_nothing(loss: GlobalOrthogonalRegularizationLoss) -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(1, 8)

    mean_term, second_moment_term = loss.compute_gor(embeddings)

    assert mean_term == 0.0
    assert second_moment_term == 0.0


def test_single_input_loss_stays_differentiable(loss: GlobalOrthogonalRegularizationLoss) -> None:
    """The returned zero has to keep the graph, or backward on a lone GOR loss raises."""
    torch.manual_seed(0)
    embeddings = torch.randn(1, 8, requires_grad=True)

    losses = loss.compute_loss_from_embeddings([embeddings])
    total = sum(losses.values())

    assert torch.isfinite(total), f"loss is {total} for a batch of one"
    assert total.requires_grad
    total.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_single_input_does_not_poison_a_combined_loss(loss: GlobalOrthogonalRegularizationLoss) -> None:
    """GOR is meant to be added to a task loss, so one nan term takes the whole sum with it."""
    torch.manual_seed(0)
    embeddings = torch.randn(1, 8, requires_grad=True)

    task_loss = embeddings.pow(2).mean()
    total = task_loss + sum(loss.compute_loss_from_embeddings([embeddings]).values())

    assert torch.isfinite(total)
    assert total.item() == pytest.approx(task_loss.item())
