from __future__ import annotations

import pytest
import torch

from sentence_transformers.multi_vector_encoder.scoring import xtr_kd_scores, xtr_scores, xtr_scores_pairwise


@pytest.mark.parametrize("scorer", [xtr_scores, xtr_kd_scores, xtr_scores_pairwise])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 7, 100])
@pytest.mark.parametrize("budget", [1, 200])
def test_streaming_xtr_scores_and_gradients(scorer, dtype, top_k, budget):
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 3, 8, generator=generator, dtype=dtype, requires_grad=True)
    shape = (2, 32, 8) if scorer is xtr_scores_pairwise else (2, 2, 32, 8)
    document = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    expected = scorer(query, document, top_k=top_k)
    actual = scorer(query, document, top_k=top_k, chunk_elements=budget)
    weights = torch.randn(expected.shape, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (query, document))
    tolerance = {torch.float32: 2e-5, torch.float64: 1e-6, torch.float16: 0.003, torch.bfloat16: 0.03}[dtype]
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_streaming_xtr_cpu_autocast(autocast_dtype):
    generator = torch.Generator().manual_seed(43)
    query = torch.randn(2, 3, 8, generator=generator, requires_grad=True)
    document = torch.randn(2, 2, 5, 8, generator=generator, requires_grad=True)
    with torch.autocast("cpu", dtype=autocast_dtype):
        expected = xtr_scores(query, document, top_k=3)
        actual = xtr_scores(query, document, top_k=3, chunk_elements=100)
    weights = torch.randn(expected.shape, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (query, document))
    tolerance = 0.03 if autocast_dtype == torch.bfloat16 else 0.003
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).min])
def test_streaming_xtr_exceptional_score_fallback(value):
    query = torch.ones(1, 1, 1, requires_grad=True)
    document = torch.full((1, 1, 4, 1), value, requires_grad=True)
    expected = xtr_scores(query, document, top_k=1)
    actual = xtr_scores(query, document, top_k=1, chunk_elements=1)
    expected_grad = torch.autograd.grad(expected.sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(), (query, document))
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference, equal_nan=True)


def test_streaming_xtr_ties_masks_and_negative_scores():
    query = torch.ones(2, 3, 2, requires_grad=True)
    document = -torch.ones(2, 2, 5, 2, requires_grad=True)
    query_mask = torch.tensor([[1.0, 0.5, 0.0], [1.0, 1.0, 0.0]])
    document_mask = torch.ones(2, 2, 5, dtype=torch.bool)
    document_mask[0, 0] = False
    expected = xtr_scores(query, document, query_mask, document_mask, top_k=7)
    actual = xtr_scores(query, document, query_mask, document_mask, top_k=7, chunk_elements=10)
    weights = torch.tensor([[0.3, 0.7, 1.0, -0.8], [0.2, 0.9, -0.1, 0.1]])
    expected_grad = torch.autograd.grad((expected * weights).sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (query, document))
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference)


def test_streaming_xtr_uses_optimized_path_and_skips_adverse_shapes(monkeypatch):
    import importlib

    module = importlib.import_module("sentence_transformers.multi_vector_encoder.scoring.xtr")
    helper = module._streaming_xtr_scores
    used = []

    def record(*args, **kwargs):
        result = helper(*args, **kwargs)
        used.append(result is not None)
        return result

    monkeypatch.setattr(module, "_streaming_xtr_scores", record)
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(2, 3, 8, generator=generator, requires_grad=True)
    documents = torch.randn(2, 2, 32, 8, generator=generator)
    module.xtr_scores(query, documents, top_k=7, chunk_elements=200)
    short_documents = torch.randn(2, 2, 2, 8, generator=generator)
    module.xtr_scores(query, short_documents, top_k=7, chunk_elements=200)
    assert used == [True, False]


def test_streaming_xtr_finite_input_score_overflow():
    query = torch.full((1, 1, 1), 2.0, requires_grad=True)
    document = torch.full((1, 1, 4, 1), -3e38, requires_grad=True)
    expected = xtr_scores(query, document, top_k=1)
    actual = xtr_scores(query, document, top_k=1, chunk_elements=1)
    expected_grad = torch.autograd.grad(expected.sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(), (query, document))
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference, equal_nan=True)


def test_streaming_xtr_implicit_padding():
    generator = torch.Generator().manual_seed(6)
    query = torch.randn(2, 3, 8, generator=generator)
    document = torch.randn(2, 2, 32, 8, generator=generator)
    query[:, -1] = 0
    document[0, 0] = 0
    document[:, 1, 20:] = 0
    query.requires_grad_()
    document.requires_grad_()
    expected = xtr_scores(query, document, top_k=100)
    actual = xtr_scores(query, document, top_k=100, chunk_elements=200)
    expected_grad = torch.autograd.grad(expected.sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(), (query, document))
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference)


def test_streaming_xtr_zero_embedding_dimension():
    query = torch.empty(1, 2, 0, requires_grad=True)
    document = torch.empty(1, 1, 4, 0, requires_grad=True)
    expected = xtr_scores(query, document, top_k=1)
    actual = xtr_scores(query, document, top_k=1, chunk_elements=20)
    expected_grad = torch.autograd.grad(expected.sum(), (query, document), retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(), (query, document))
    for reference, result in zip((expected, *expected_grad), (actual, *actual_grad)):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("cached", [False, True])
def test_streaming_xtr_ranking_loss_backward(cached):
    from sentence_transformers.multi_vector_encoder.losses import (
        CachedMultiVectorMultipleNegativesRankingLoss,
        MultiVectorMultipleNegativesRankingLoss,
    )
    from sentence_transformers.multi_vector_encoder.scoring import XTRScores

    class PassthroughModel(torch.nn.Module):
        def forward(self, features, task=None):
            return features

    generator = torch.Generator().manual_seed(17)
    embeddings = [torch.randn(2, length, 8, generator=generator) for length in (3, 32, 40)]
    results = []
    for budget in (None, 200):
        features = [
            {
                "token_embeddings": embedding.clone().requires_grad_(),
                "attention_mask": torch.ones(embedding.shape[:2], dtype=torch.bool),
            }
            for embedding in embeddings
        ]
        kwargs = {"mini_batch_size": 1, "show_progress_bar": False} if cached else {}
        loss_class = (
            CachedMultiVectorMultipleNegativesRankingLoss if cached else MultiVectorMultipleNegativesRankingLoss
        )
        loss = loss_class(model=PassthroughModel(), similarity_fct=XTRScores(top_k=7, chunk_elements=budget), **kwargs)
        value = loss(features, labels=None)
        value.backward()
        for feature in features:
            gradient = feature["token_embeddings"].grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
        results.append((value.detach(), *(feature["token_embeddings"].grad for feature in features)))
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_streaming_xtr_non_cpu_dispatch_retains_legacy(monkeypatch):
    import importlib

    module = importlib.import_module("sentence_transformers.multi_vector_encoder.scoring.xtr")

    def unexpected_streaming(*args, **kwargs):
        raise AssertionError("Non-CPU inputs must not enter the CPU streaming implementation")

    monkeypatch.setattr(module, "_streaming_xtr_scores", unexpected_streaming)
    # Meta tensors test dispatch without allocating or claiming validation on a GPU.
    queries = torch.empty(2, 3, 8, device="meta", requires_grad=True)
    documents = torch.empty(2, 2, 32, 8, device="meta", requires_grad=True)
    scores = module.xtr_scores(queries, documents, top_k=7, chunk_elements=200)
    assert scores.device.type == "meta"
    assert scores.shape == (2, 4)


@pytest.mark.parametrize("query_grad,document_grad", [(True, False), (False, True), (True, True)])
def test_streaming_xtr_one_sided_gradients(monkeypatch, query_grad, document_grad):
    import importlib

    module = importlib.import_module("sentence_transformers.multi_vector_encoder.scoring.xtr")
    helper = module._streaming_xtr_scores
    used = []

    def record(*args, **kwargs):
        result = helper(*args, **kwargs)
        used.append(result is not None)
        return result

    monkeypatch.setattr(module, "_streaming_xtr_scores", record)
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(2, 3, 8, generator=generator, requires_grad=query_grad)
    document = torch.randn(2, 2, 32, 8, generator=generator, requires_grad=document_grad)
    expected = module.xtr_scores(query, document, top_k=7)
    actual = module.xtr_scores(query, document, top_k=7, chunk_elements=200)
    inputs = tuple(t for t in (query, document) if t.requires_grad)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), inputs), torch.autograd.grad(expected.sum(), inputs)
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert used == [True]


@pytest.mark.parametrize("context", [torch.enable_grad, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_streaming_xtr_inference_retains_legacy(monkeypatch, context, requires_grad):
    import importlib

    if context is torch.enable_grad and requires_grad:
        pytest.skip("Training dispatch is covered separately")
    module = importlib.import_module("sentence_transformers.multi_vector_encoder.scoring.xtr")

    def unexpected_streaming(*args, **kwargs):
        raise AssertionError("Inference must retain legacy scoring")

    monkeypatch.setattr(module, "_streaming_xtr_scores", unexpected_streaming)
    generator = torch.Generator().manual_seed(20)
    query = torch.randn(2, 3, 8, generator=generator, requires_grad=requires_grad)
    document = torch.randn(2, 2, 32, 8, generator=generator, requires_grad=requires_grad)
    with context():
        expected = module.xtr_scores(query, document, top_k=7)
        actual = module.xtr_scores(query, document, top_k=7, chunk_elements=200)
    torch.testing.assert_close(actual, expected)


def test_streaming_xtr_nested_inference_retains_legacy(monkeypatch):
    import importlib

    module = importlib.import_module("sentence_transformers.multi_vector_encoder.scoring.xtr")

    def unexpected_streaming(*args, **kwargs):
        raise AssertionError("Inference mode must retain legacy scoring even with grad enabled")

    monkeypatch.setattr(module, "_streaming_xtr_scores", unexpected_streaming)
    generator = torch.Generator().manual_seed(21)
    query = torch.randn(2, 3, 8, generator=generator, requires_grad=True)
    document = torch.randn(2, 2, 32, 8, generator=generator, requires_grad=True)
    with torch.inference_mode(), torch.enable_grad():
        assert torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        expected = module.xtr_scores(query, document, top_k=7)
        actual = module.xtr_scores(query, document, top_k=7, chunk_elements=200)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
