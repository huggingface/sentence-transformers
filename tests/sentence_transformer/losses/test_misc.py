from __future__ import annotations

import re

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
    CosineSimilarityLoss,
    OnlineContrastiveLoss,
)
from sentence_transformers.sparse_encoder.losses import SparseAnglELoss, SparseCoSENTLoss, SparseCosineSimilarityLoss

BATCH_SIZE = 8
SCORE_LABELS = torch.tensor([0.1, 0.9, 0.4, 0.7, 0.0, 0.3, 1.0, 0.6])
BINARY_LABELS = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
CLASS_LABELS = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

# (loss class, number of embedding columns, 1D labels)
LOSSES = [
    (BatchAllTripletLoss, 1, CLASS_LABELS),
    (BatchHardTripletLoss, 1, CLASS_LABELS),
    (BatchHardSoftMarginTripletLoss, 1, CLASS_LABELS),
    (BatchSemiHardTripletLoss, 1, CLASS_LABELS),
    (CosineSimilarityLoss, 2, SCORE_LABELS),
    (SparseCosineSimilarityLoss, 2, SCORE_LABELS),
    (CoSENTLoss, 2, SCORE_LABELS),
    (AnglELoss, 2, SCORE_LABELS),
    (AnglELoss, 3, SCORE_LABELS),  # anchor-positive-negative path, which builds labels for the negatives
    (SparseCoSENTLoss, 2, SCORE_LABELS),
    (SparseAnglELoss, 2, SCORE_LABELS),
    (ContrastiveLoss, 2, BINARY_LABELS),
    (OnlineContrastiveLoss, 2, BINARY_LABELS),
]
LOSS_IDS = [f"{loss_class.__name__}-{num_columns}col" for loss_class, num_columns, _ in LOSSES]


def _embeddings(num_columns: int, batch_size: int = BATCH_SIZE) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(batch_size, 16, generator=generator).requires_grad_() for _ in range(num_columns)]


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
@pytest.mark.parametrize("batch_size", [1, BATCH_SIZE])
@pytest.mark.parametrize("labels_per_row", [2, 3])
def test_multiple_labels_per_row_raise(loss_class, num_columns, labels, batch_size, labels_per_row) -> None:
    loss_fn = loss_class(torch.nn.Identity())
    labels = labels[:batch_size, None].repeat(1, labels_per_row)
    message = (
        f"{loss_class.__name__} expects {batch_size} labels (one per example), "
        f"but got {labels.numel()} labels with shape {tuple(labels.shape)}."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        loss_fn.compute_loss_from_embeddings(_embeddings(num_columns, batch_size), labels)


@pytest.mark.parametrize(("loss_class", "num_columns", "labels"), LOSSES, ids=LOSS_IDS)
def test_single_label_does_not_broadcast_over_batch(loss_class, num_columns, labels) -> None:
    loss_fn = loss_class(torch.nn.Identity())
    with pytest.raises(ValueError, match=rf"expects {BATCH_SIZE} labels \(one per example\), but got 1 labels"):
        loss_fn.compute_loss_from_embeddings(_embeddings(num_columns), labels[:1])
