from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from sentence_transformers.sentence_transformer.losses import (
    BatchAllTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
)
from sentence_transformers.sentence_transformer.losses.batch_hard_triplet import BatchHardTripletLossDistanceFunction


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("case", ["nearby", "large_norm"])
def test_triplet_distance_matches_full_precision(dtype, autocast, squared, case):
    # Both inputs are representable in the source dtype. Only the distance calculation loses precision.
    values = [[1.0, 1.0], [1.0078125, 1.0], [1.0, 1.015625], [1.0, 1.0]]
    if case == "large_norm":
        values = [[200.0, 200.0], [201.0, 200.0], [200.0, 202.0], [200.0, 200.0]]
    embeddings = torch.tensor(values, dtype=dtype, requires_grad=True)
    reference = embeddings.detach().double().requires_grad_()
    expected = torch.cdist(reference, reference, compute_mode="donot_use_mm_for_euclid_dist")
    if squared:
        expected = expected.square()

    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        actual = BatchHardTripletLossDistanceFunction.euclidean_distance(embeddings, squared=squared)
    actual.sum().backward()
    expected.sum().backward()

    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert actual.dtype == expected_dtype
    assert torch.isfinite(actual).all()
    assert torch.isfinite(embeddings.grad).all()
    torch.testing.assert_close(actual, expected.to(expected_dtype))
    torch.testing.assert_close(embeddings.grad, reference.grad.to(dtype), atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(actual.diag()) == 0


@pytest.mark.parametrize(
    "loss_class", [BatchAllTripletLoss, BatchHardTripletLoss, BatchHardSoftMarginTripletLoss, BatchSemiHardTripletLoss]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("autocast", [False, True])
def test_batch_triplet_losses_keep_finite_loss_and_gradients(loss_class, dtype, autocast):
    embeddings = torch.tensor(
        [[200.0, 200.0], [201.0, 200.0], [200.0, 202.0], [201.0, 202.0]], dtype=dtype, requires_grad=True
    )
    reference = embeddings.detach().float().requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = loss_class(nn.Identity())
    reference_loss_fn = loss_class(
        nn.Identity(), distance_metric=lambda x: torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
    )
    expected = reference_loss_fn.compute_loss_from_embeddings([reference], labels)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    actual.backward()
    expected.backward()

    assert torch.isfinite(actual)
    assert torch.isfinite(embeddings.grad).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference.grad.to(dtype), atol=1e-3, rtol=1e-3)
