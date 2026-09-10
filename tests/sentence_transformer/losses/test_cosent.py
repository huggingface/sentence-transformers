from __future__ import annotations

import math

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import AnglELoss, CoSENTLoss


@pytest.fixture
def dummy_model():
    class DummyModel:
        pass

    return DummyModel()


def _pairs(scores: list[float]) -> list[torch.Tensor]:
    """Embeddings whose dot-product similarity is exactly `scores`, one per input pair."""
    left = torch.tensor([[s] for s in scores], dtype=torch.float32)
    return [left, torch.ones_like(left)]


@pytest.mark.parametrize(
    ("low_label_score", "high_label_score"),
    [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (2.0, -1.0)],
)
def test_cosent_penalises_the_lower_labelled_pair(dummy_model, low_label_score, high_label_score) -> None:
    """The exponent is s(k,l) - s(i,j): the score of the pair with the *lower* expected similarity
    minus the score of the pair with the higher one, so ranking them the wrong way round costs more."""
    loss = CoSENTLoss(dummy_model, scale=1.0, similarity_fct=lambda a, b: (a * b).sum(-1))
    labels = torch.tensor([0.0, 1.0])

    value = loss.compute_loss_from_embeddings(_pairs([low_label_score, high_label_score]), labels)

    assert value.item() == pytest.approx(math.log(1 + math.exp(low_label_score - high_label_score)), abs=1e-5)


@pytest.mark.parametrize(("labels", "score_difference"), [([0.0, 1.0], -1.0), ([1.0, 0.0], 1.0)])
def test_angle_loss_ranks_actual_angle_scores(dummy_model, labels, score_difference) -> None:
    loss = AnglELoss(dummy_model, scale=2.0)
    embeddings = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]])]
    torch.testing.assert_close(loss.similarity_fct(*embeddings), torch.tensor([0.0, 1.0]))

    value = loss.compute_loss_from_embeddings(embeddings, torch.tensor(labels))

    assert value.item() == pytest.approx(math.log1p(math.exp(2.0 * score_difference)), abs=1e-5)
