from __future__ import annotations

import pytest
import torch

from sentence_transformers.multi_vector_encoder.scoring import xtr


def original_scores(*args, **kwargs):
    """Exercise the untouched legacy branch with the SAME chunk budget."""
    original = xtr._streaming_xtr_scores
    try:
        xtr._streaming_xtr_scores = lambda *a, **k: None
        return xtr.xtr_scores(*args, **kwargs)
    finally:
        xtr._streaming_xtr_scores = original


def compare(q, d, **kwargs):
    q = q.clone().requires_grad_()
    d = d.clone().requires_grad_()
    expected = original_scores(q, d, **kwargs)
    actual = xtr.xtr_scores(q, d, **kwargs)
    weights = torch.arange(1, expected.numel() + 1).reshape_as(expected)
    eg = torch.autograd.grad((expected * weights).sum(), (q, d), retain_graph=True)
    ag = torch.autograd.grad((actual * weights).sum(), (q, d))
    for a, e in zip((actual, *ag), (expected, *eg)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("normalized, seed", [(False, 16), (True, 8), (True, 15)])
def test_rounding_sensitive_cutoff_retains_original_chunked_scorer(monkeypatch, normalized, seed):
    # Fixed-seed cancellation example; a shape-dependent row GEMM may round its
    # tied candidates differently. Compare native legacy membership, not a fixed
    # topk tie order (which is backend-dependent).
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(16, 17, generator=generator)
    d = torch.randn(512, 17, generator=generator)
    if normalized:
        q = torch.nn.functional.normalize(q, dim=-1)
        d = torch.nn.functional.normalize(d, dim=-1)
        qa = torch.nn.functional.normalize(q[0].double(), dim=0)
        a = torch.randn(17, generator=generator).double()
        b = torch.randn(17, generator=generator).double()
        a = torch.nn.functional.normalize(a - qa * (qa @ a), dim=0)
        b = torch.nn.functional.normalize(b - qa * (qa @ b), dim=0)
        d[0] = (0.7 * qa + (1 - 0.7**2) ** 0.5 * a).float()
        d[129] = (0.7 * qa + (1 - 0.7**2) ** 0.5 * b).float()
    else:
        d *= 0.01
        qa = q[0].double()
        a = torch.randn(17, generator=generator).double()
        b = torch.randn(17, generator=generator).double()
        d[0] = (a + qa * (100 - qa @ a) / (qa @ qa)).float()
        d[129] = (b + qa * (100 - qa @ b) / (qa @ qa)).float()
    chunk_scores = torch.cat([q @ d[i : i + 64].T for i in range(0, 512, 64)], dim=1)
    cutoff = chunk_scores[0].topk(2).values
    if cutoff[0] != cutoff[1]:
        pytest.skip("Constructed cutoff is not tied on this BLAS/backend")
    q, d = q.reshape(1, 16, 17), d.reshape(1, 8, 64, 17)
    compare(q, d, top_k=1, chunk_elements=2112)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_exact_and_adjacent_ulp_cutoffs_dispatch(monkeypatch, offset):
    # The near-cutoff cases must keep streaming and preserve the strict winner;
    # no epsilon widening or change to the matmul dtype is used.
    q = torch.zeros(1, 16, 17)
    q[..., 0] = 1
    d = torch.zeros(1, 8, 64, 17)
    d[..., 0] = 0.1
    flat = d.reshape(512, 17)
    flat[0, 0] = 0.7
    flat[129, 0] = torch.nextafter(flat[0, 0], torch.tensor(float(offset))) if offset else flat[0, 0]
    original = xtr._streaming_xtr_scores
    returned = []

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    monkeypatch.setattr(xtr, "_streaming_xtr_scores", track)
    compare(q, d, top_k=1, chunk_elements=2112)
    assert len(returned) == 1
    assert (returned[0] is None) == (offset == 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cpu_autocast_really_streams_without_cutoff_tie(monkeypatch, dtype):
    q = torch.zeros(1, 16, 17)
    q[..., 0] = 1
    d = torch.zeros(1, 8, 64, 17)
    d[..., 0] = 0.125
    d.reshape(512, 17)[129, 0] = 0.75
    original = xtr._streaming_xtr_scores
    returned = []

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    monkeypatch.setattr(xtr, "_streaming_xtr_scores", track)
    with torch.autocast("cpu", dtype=dtype):
        compare(q, d, top_k=1, chunk_elements=2112)
    assert len(returned) == 1 and returned[0] is not None
