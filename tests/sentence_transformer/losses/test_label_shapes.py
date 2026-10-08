"""Losses that take one label per row must treat labels of shape (n, 1) like labels of shape (n,).

A label column of one-element lists is collated to shape (n, 1). Without flattening, that broadcasts against
the (n,) per-row scores and either silently computes a different loss or raises a shape error.
"""

from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import (
    AnglELoss,
    BatchAllTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    ContrastiveLoss,
    CoSENTLoss,
    OnlineContrastiveLoss,
)
from sentence_transformers.sparse_encoder.losses import SparseAnglELoss, SparseCoSENTLoss

BATCH_SIZE = 8
SCORE_LABELS = torch.tensor([0.1, 0.9, 0.4, 0.7, 0.0, 0.3, 1.0, 0.6])
BINARY_LABELS = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
CLASS_LABELS = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

# (loss class, number of embedding columns, 1D labels)
LOSSES = [
    (CoSENTLoss, 2, SCORE_LABELS),
    (AnglELoss, 2, SCORE_LABELS),
    (AnglELoss, 3, SCORE_LABELS),  # anchor-positive-negative path, which builds labels for the negatives
    (SparseCoSENTLoss, 2, SCORE_LABELS),
    (SparseAnglELoss, 2, SCORE_LABELS),
    (ContrastiveLoss, 2, BINARY_LABELS),
    (OnlineContrastiveLoss, 2, BINARY_LABELS),
    (BatchHardTripletLoss, 1, CLASS_LABELS),
    (BatchHardSoftMarginTripletLoss, 1, CLASS_LABELS),
    (BatchAllTripletLoss, 1, CLASS_LABELS),
    (BatchSemiHardTripletLoss, 1, CLASS_LABELS),
]
LOSS_IDS = [f"{loss_class.__name__}-{num_columns}col" for loss_class, num_columns, _ in LOSSES]


def _embeddings(num_columns: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(BATCH_SIZE, 16, generator=generator).requires_grad_() for _ in range(num_columns)]


@pytest.mark.parametrize(("loss_class", "num_columns", "labels"), LOSSES, ids=LOSS_IDS)
def test_column_labels_match_flat_labels(loss_class, num_columns, labels) -> None:
    loss_fn = loss_class(torch.nn.Identity())
    flat_embeddings = _embeddings(num_columns)
    column_embeddings = _embeddings(num_columns)

    expected = loss_fn.compute_loss_from_embeddings(flat_embeddings, labels)
    actual = loss_fn.compute_loss_from_embeddings(column_embeddings, labels[:, None])
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected)
    for column_embedding, flat_embedding in zip(column_embeddings, flat_embeddings):
        torch.testing.assert_close(column_embedding.grad, flat_embedding.grad)


@pytest.mark.parametrize(("loss_class", "num_columns", "labels"), LOSSES, ids=LOSS_IDS)
def test_two_labels_per_row_raise(loss_class, num_columns, labels) -> None:
    """Flattening (n, 2) labels gives 2n labels, which must raise rather than produce a loss value."""
    loss_fn = loss_class(torch.nn.Identity())

    with pytest.raises((RuntimeError, IndexError, ValueError)):
        loss_fn.compute_loss_from_embeddings(_embeddings(num_columns), labels[:, None].repeat(1, 2))
