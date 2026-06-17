from __future__ import annotations

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import ListMLELoss, PListMLELoss

LISTWISE_LOSSES = [PListMLELoss, ListMLELoss]  # ListMLELoss subclasses PListMLELoss


@pytest.mark.parametrize("loss_cls", LISTWISE_LOSSES)
@pytest.mark.parametrize("respect_input_order", [True, False])
def test_listwise_loss_is_padding_invariant(reranker_bert_tiny_model_v6: CrossEncoder, loss_cls, respect_input_order):
    # A query's loss must not change because another query in the batch has more documents
    # (which pads this query's row). Batching must equal the mean of the per-query losses.
    # Labels are intentionally unsorted so the respect_input_order=False branch actually
    # permutes via sort/gather (exercising the sorted_mask = gather(mask, indices) path).
    model = reranker_bert_tiny_model_v6
    loss_fn = loss_cls(model, respect_input_order=respect_input_order)

    queries_a = ["what is the capital of france"]
    docs_a = [["berlin is the capital of germany", "paris is the capital of france"]]
    labels_a = [torch.tensor([0.0, 1.0], device=model.device)]

    queries_b = ["what is the largest planet"]
    docs_b = [["mercury is the smallest", "jupiter is the largest planet", "the moon orbits earth"]]
    labels_b = [torch.tensor([1.0, 2.0, 0.0], device=model.device)]

    loss_a = loss_fn((queries_a, docs_a), labels_a)
    loss_b = loss_fn((queries_b, docs_b), labels_b)
    loss_batched = loss_fn((queries_a + queries_b, docs_a + docs_b), labels_a + labels_b)

    # On the buggy implementation, padding the 2-doc query to width 3 inflates its loss, so the
    # batched mean diverges from the mean of the separately computed per-query losses.
    assert torch.allclose(loss_batched, (loss_a + loss_b) / 2, atol=1e-5)
