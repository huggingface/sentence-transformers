from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sentence_transformers.cross_encoder.losses import LambdaLoss, RankNetLoss


class _ScoreModel(nn.Module):
    num_labels = 1

    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.scores = nn.Parameter(scores)

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        return {"scores": self.scores[inputs["indices"]].unsqueeze(-1)}


@pytest.mark.parametrize("loss_cls", [RankNetLoss, LambdaLoss])
@pytest.mark.parametrize("gap", [0.5, 20.0, 1000.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0], ids=["correct", "incorrect"])
@pytest.mark.parametrize("sigma", [0.5, 2.0])
@pytest.mark.parametrize("reduction_log", ["natural", "binary"])
def test_ranking_loss_matches_pairwise_cross_entropy(loss_cls, gap, direction, sigma, reduction_log):
    model = _ScoreModel(torch.tensor([0.0, direction * gap]))
    loss = loss_cls(model, sigma=sigma, reduction_log=reduction_log)(
        (["query"], [["0", "1"]]), [torch.tensor([1.0, 0.0])]
    )

    # The preferred document is 0. Compare to an independent binary cross-entropy
    # reference in float64, including the analytical two-document NDCG2++ weight.
    scores = model.scores.detach().double().requires_grad_()
    expected = F.binary_cross_entropy_with_logits(sigma * (scores[0] - scores[1]), torch.tensor(1.0).double())
    if loss_cls is LambdaLoss:
        expected = expected * (11 * (1 - 1 / math.log2(3)))
    if reduction_log == "binary":
        expected = expected / math.log(2)

    torch.testing.assert_close(loss, expected.float(), rtol=1e-5, atol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("reduction_log", ["natural", "binary"])
def test_ranknet_extreme_scores_with_padding_match_valid_pair_cross_entropies(reduction_log):
    model = _ScoreModel(torch.tensor([0.0, -3.0, 1000.0, 0.0, 100.0]))
    loss = RankNetLoss(model, mini_batch_size=2, reduction_log=reduction_log)(
        (["query a", "query b"], [["0", "1", "2"], ["3", "4"]]),
        [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([1.0, 0.0])],
    )

    scores = model.scores.detach().double().requires_grad_()
    # Four ordered pairs across the two queries; the padded document has no pair.
    pair_logits = torch.stack([scores[a] - scores[b] for a, b in [(0, 1), (0, 2), (1, 2), (3, 4)]])
    expected = F.binary_cross_entropy_with_logits(pair_logits, torch.ones_like(pair_logits))
    if reduction_log == "binary":
        expected = expected / math.log(2)

    torch.testing.assert_close(loss, expected.float(), rtol=1e-5, atol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=1e-5, atol=1e-6)
