from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesSymmetricRankingLoss,
    MultipleNegativesSymmetricRankingLoss,
)
from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

ANCHORS = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e"]
POSITIVES = ["positive a", "positive b", "positive c", "positive d", "positive e"]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("minibatching", [{"mini_batch_size": 2}, {"mini_batch_num_tokens": 16}])
def test_cached_symmetric_matches_symmetric(stsb_bert_tiny_model: SentenceTransformer, minibatching: dict) -> None:
    """The deprecated symmetric shim only wires ``directions`` and ``partition_mode`` into the
    composed loss, and that wiring has no test of its own. The cached version must reproduce the
    non-cached symmetric loss and gradient, for both mini-batching modes."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(5, dtype=torch.long)

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss_value = loss_fn([model.preprocess(ANCHORS), model.preprocess(POSITIVES)], labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(CachedMultipleNegativesSymmetricRankingLoss(model, **minibatching))
    plain_loss, plain_grads = loss_and_grads(MultipleNegativesSymmetricRankingLoss(model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)
