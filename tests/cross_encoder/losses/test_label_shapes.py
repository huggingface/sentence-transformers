"""Pointwise CrossEncoder losses must treat labels of shape (n, 1) like labels of shape (n,).

A label column of one-element lists is collated to shape (n, 1). ``nn.MSELoss`` broadcasts that against the
(n,) logits with only a warning, training every logit towards the mean label.
"""

from __future__ import annotations

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, MSELoss

INPUTS = [
    ["what is the capital of France?", "how tall is mount everest?", "who wrote hamlet?"],
    ["Paris is the capital of France.", "Bananas are yellow.", "Hamlet was written by Shakespeare."],
]
LABELS = torch.tensor([1.0, 0.0, 1.0])


@pytest.mark.parametrize("loss_class", [MSELoss, BinaryCrossEntropyLoss])
def test_column_labels_match_flat_labels(reranker_bert_tiny_model_v54: CrossEncoder, loss_class) -> None:
    model = reranker_bert_tiny_model_v54
    model.eval()  # no dropout, so both forward passes see the same logits
    loss_fn = loss_class(model)

    expected = loss_fn(INPUTS, LABELS)
    actual = loss_fn(INPUTS, LABELS[:, None])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_class", [MSELoss, BinaryCrossEntropyLoss])
def test_two_labels_per_row_raise(reranker_bert_tiny_model_v54: CrossEncoder, loss_class) -> None:
    """Flattening (n, 2) labels gives 2n labels, which must raise rather than produce a loss value."""
    loss_fn = loss_class(reranker_bert_tiny_model_v54)

    with pytest.raises((RuntimeError, ValueError)):
        loss_fn(INPUTS, LABELS[:, None].repeat(1, 2))
