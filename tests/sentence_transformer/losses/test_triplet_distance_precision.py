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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
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

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.isfinite(embeddings.grad).all()
    torch.testing.assert_close(actual, expected.float())
    torch.testing.assert_close(embeddings.grad, reference.grad.to(dtype), atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(actual.diag()) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("autocast", [False, True])
def test_cosine_distance_precision_and_diagonal(dtype, autocast):
    embeddings = torch.tensor([[1.0] * 7] * 2 + [[1.0, -1.0] + [0.0] * 5] * 2, dtype=dtype, requires_grad=True)
    reference = embeddings.detach().double().requires_grad_()
    expected = 1 - nn.functional.cosine_similarity(reference[:, None], reference[None, :], dim=-1)
    loss_fn = BatchAllTripletLoss(
        nn.Identity(), distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance, margin=1.5
    )
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        actual = BatchHardTripletLossDistanceFunction.cosine_distance(embeddings)
        loss = loss_fn.compute_loss_from_embeddings([embeddings], torch.tensor([0, 0, 1, 1]))
    actual.sum().backward()
    expected.sum().backward()

    assert (actual >= 0).all()
    assert torch.count_nonzero(actual.diag()) == 0
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss, loss.new_tensor(0.5))
    torch.testing.assert_close(embeddings.grad, reference.grad.to(dtype), atol=1e-3, rtol=1e-2)


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
