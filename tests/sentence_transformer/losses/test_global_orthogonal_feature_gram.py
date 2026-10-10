from __future__ import annotations

import pytest
import torch
from sentence_transformers.sentence_transformer.losses import GlobalOrthogonalRegularizationLoss
from sentence_transformers.util import cos_sim


def reference_terms(embeddings, similarity=cos_sim):
    scores = similarity(embeddings, embeddings)
    scores.fill_diagonal_(0.0)
    count = max(len(embeddings) * (len(embeddings) - 1), 1)
    return (scores.sum() / count).square(), torch.relu(scores.square().sum() / count - 1.0 / embeddings.shape[1])


@pytest.mark.parametrize("batch,dim", [(0, 4), (1, 4), (4, 4), (5, 4), (7, 4), (8, 4), (9, 4), (32, 4), (128, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_loss_and_gradient_match_pairwise_definition(batch, dim, dtype):
    torch.manual_seed(batch + dim)
    # A shared component activates the second-moment penalty, so both gradients are exercised.
    embeddings = (torch.randn(batch, dim, dtype=dtype) * 0.1 + 1.0).requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(reference)
    tolerance = 1e-10 if dtype == torch.float64 else 2e-5
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(observed)
        assert observed >= 0.0
    sum(actual).backward()
    sum(expected).backward()
    gradient_tolerance = 1e-9 if dtype == torch.float64 else (0.02 if dtype == torch.bfloat16 else 0.003)
    torch.testing.assert_close(embeddings.grad, reference.grad, rtol=gradient_tolerance, atol=gradient_tolerance)
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize("kind", ["zero", "mixed_zero", "identical", "antipodal", "orthogonal"])
def test_degenerate_embedding_geometries(kind):
    if kind == "zero":
        embeddings = torch.zeros(16, 4, dtype=torch.float64)
    elif kind == "mixed_zero":
        embeddings = torch.cat([torch.zeros(8, 4), torch.ones(8, 4)]).double()
    elif kind == "identical":
        embeddings = torch.ones(16, 4, dtype=torch.float64)
    elif kind == "antipodal":
        embeddings = torch.cat([torch.ones(8, 4), -torch.ones(8, 4)]).double()
    else:
        embeddings = torch.eye(4, dtype=torch.float64).repeat(4, 1)
    embeddings.requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(reference)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target, rtol=1e-10, atol=1e-10)
    sum(actual).backward()
    sum(expected).backward()
    torch.testing.assert_close(embeddings.grad, reference.grad, rtol=1e-9, atol=1e-9)


def test_custom_similarity_uses_original_pairwise_path():
    calls = []

    def custom(a, b):
        calls.append((a.shape, b.shape))
        return a @ b.T

    embeddings = torch.randn(16, 4, dtype=torch.float64, requires_grad=True)
    actual = GlobalOrthogonalRegularizationLoss(None, similarity_fct=custom).compute_gor(embeddings)
    assert len(calls) == 1
    expected = reference_terms(embeddings, similarity=custom)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target)


def test_sparse_cosine_uses_original_pairwise_path():
    embeddings = torch.eye(4).repeat(4, 1).to_sparse()
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(embeddings)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target)


def test_cpu_autocast_preserves_original_pairwise_behavior():
    embeddings = torch.randn(16, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
        expected = reference_terms(embeddings)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target)


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_torch_autocast_query_signature(monkeypatch, enabled):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: False)
    monkeypatch.setattr(torch, "is_autocast_cpu_enabled", lambda: enabled)
    embeddings = torch.randn(16, 4, dtype=torch.float64, requires_grad=True)
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(embeddings)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target)


def test_large_batch_avoids_pairwise_matrix(monkeypatch):
    original_mm = torch.mm
    shapes = []

    def record_mm(a, b, *args, **kwargs):
        shapes.append((tuple(a.shape), tuple(b.shape)))
        return original_mm(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "mm", record_mm)
    GlobalOrthogonalRegularizationLoss(None).compute_gor(torch.randn(128, 4))
    assert ((128, 4), (4, 128)) not in shapes


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_multicolumn_weighted_loss_and_gradients(aggregation):
    torch.manual_seed(7)
    embeddings = [torch.randn(32, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    loss = GlobalOrthogonalRegularizationLoss(None, mean_weight=0.5, second_moment_weight=2.0, aggregation=aggregation)
    actual = loss.compute_loss_from_embeddings(embeddings)
    terms = [reference_terms(embedding) for embedding in embeddings]
    divisor = len(embeddings) if aggregation == "mean" else 1
    expected = {
        "gor_mean": 0.5 * sum(term[0] for term in terms) / divisor,
        "gor_second_moment": 2.0 * sum(term[1] for term in terms) / divisor,
    }
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=1e-9, atol=1e-9)
    actual_grads = torch.autograd.grad(sum(actual.values()), embeddings, retain_graph=True)
    expected_grads = torch.autograd.grad(sum(expected.values()), embeddings)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nearly_orthogonal_with_zero_rows_is_finite_and_nonnegative(dtype):
    torch.manual_seed(41)
    basis, _ = torch.linalg.qr(torch.randn(32, 32, dtype=dtype))
    embeddings = torch.cat([basis, torch.zeros(32, 32, dtype=dtype)]).requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(reference)
    tolerance = 1e-10 if dtype == torch.float64 else 1e-6
    for observed, target in zip(actual, expected):
        assert torch.isfinite(observed) and observed >= 0.0
        torch.testing.assert_close(observed, target, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        torch.autograd.grad(sum(actual), embeddings)[0],
        torch.autograd.grad(sum(expected), reference)[0],
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("scale", [0.0, 1e-15, 1e-12])
def test_near_zero_normalization_preserves_pairwise_gradient(scale):
    torch.manual_seed(15)
    embeddings = torch.randn(16, 4, dtype=torch.float64)
    embeddings[0] *= scale
    embeddings.requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    actual = GlobalOrthogonalRegularizationLoss(None).compute_gor(embeddings)
    expected = reference_terms(reference)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(sum(actual), embeddings)[0],
        torch.autograd.grad(sum(expected), reference)[0],
        rtol=1e-9,
        atol=1e-9,
    )
