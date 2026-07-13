from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)

ANCHORS = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f"]
POSITIVES = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f"]


def _disable_dropout(model: SparseEncoder) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for attribute in ("dropout_prob", "attention_dropout", "attention_probs_dropout_prob"):
            if isinstance(getattr(module, attribute, None), float):
                setattr(module, attribute, 0.0)


def _make_loss(model: SparseEncoder, cached: bool, mini_batch_size: int = 2) -> torch.nn.Module:
    kwargs = {
        "model": model,
        "loss": SparseMultipleNegativesRankingLoss(model),
        "document_regularizer_weight": 3e-5,
        "query_regularizer_weight": 5e-5,
    }
    if cached:
        return CachedSpladeLoss(**kwargs, mini_batch_size=mini_batch_size)
    return SpladeLoss(**kwargs)


def _loss_and_grads(
    model: SparseEncoder, loss_fn: torch.nn.Module, batch_size: int = 6
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.zero_grad()
    output = loss_fn([model.preprocess(ANCHORS[:batch_size]), model.preprocess(POSITIVES[:batch_size])], None)
    assert isinstance(output, dict)
    torch.stack(list(output.values())).sum().backward()
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    return output, grads


@pytest.mark.parametrize("mini_batch_size", [2, 4, 50])
def test_cached_splade_matches_splade(splade_bert_tiny_model: SparseEncoder, mini_batch_size: int) -> None:
    """``mini_batch_size`` only bounds memory: the cached loss must reproduce SpladeLoss's total loss,
    per-component values, and gradients, whether or not it divides the batch size."""
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    cached_output, cached_grads = _loss_and_grads(
        model, _make_loss(model, cached=True, mini_batch_size=mini_batch_size)
    )
    plain_output, plain_grads = _loss_and_grads(model, _make_loss(model, cached=False))

    assert cached_output.keys() == plain_output.keys()
    for key in plain_output:
        assert cached_output[key].item() == pytest.approx(plain_output[key].item(), rel=1e-4, abs=1e-5), key

    assert cached_grads and sum(grad.abs().sum() for grad in cached_grads.values()) > 0
    # SPLADE gradients accumulate over the whole vocabulary, so the float noise between differently
    # sized GEMMs reaches ~1e-4 absolute on gradients of magnitude ~5 (still ~2e-5 relative).
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=2e-4, msg=name)


def test_cached_splade_components_sum_to_the_total(splade_bert_tiny_model: SparseEncoder) -> None:
    """The trainer back-propagates the sum of the returned dict, so the components must sum exactly to
    the gradient-carrying total even though only one entry carries the gradient."""
    model = splade_bert_tiny_model.to("cpu")
    model.train()
    loss_fn = _make_loss(model, cached=True)

    output = loss_fn([model.preprocess(ANCHORS), model.preprocess(POSITIVES)], None)
    total = torch.stack(list(output.values())).sum()
    assert total.requires_grad
    assert sum(value.requires_grad for value in output.values()) == 1, "exactly one component must carry the gradient"
    total.backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())


def test_cached_splade_two_forwards_before_one_backward(splade_bert_tiny_model: SparseEncoder) -> None:
    """Each forward pass must hand its own cached gradients to its own backward hook."""
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    first = [model.preprocess(ANCHORS[:3]), model.preprocess(POSITIVES[:3])]
    second = [model.preprocess(ANCHORS[3:]), model.preprocess(POSITIVES[3:])]

    model.zero_grad()
    shared = _make_loss(model, cached=True)
    first_output, second_output = shared(first, None), shared(second, None)
    (torch.stack(list(first_output.values())).sum() + torch.stack(list(second_output.values())).sum()).backward()
    shared_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    model.zero_grad()
    first_output = _make_loss(model, cached=True)(first, None)
    second_output = _make_loss(model, cached=True)(second, None)
    (torch.stack(list(first_output.values())).sum() + torch.stack(list(second_output.values())).sum()).backward()
    reference_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    assert shared_grads and sum(grad.abs().sum() for grad in shared_grads.values()) > 0
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


def test_cached_splade_under_no_grad(splade_bert_tiny_model: SparseEncoder) -> None:
    """The eval loss must be computable under ``torch.no_grad`` and match the training-path total."""
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.eval()
    features = [model.preprocess(ANCHORS), model.preprocess(POSITIVES)]

    with torch.no_grad():
        cached_output = _make_loss(model, cached=True)(features, None)
        plain_output = _make_loss(model, cached=False)(features, None)

    assert cached_output.keys() == plain_output.keys()
    for key in plain_output:
        assert cached_output[key].item() == pytest.approx(plain_output[key].item(), rel=1e-4, abs=1e-5), key


def test_cached_splade_inference_free_frozen_query_route(inference_free_splade_bert_tiny_model: SparseEncoder) -> None:
    """An inference-free SPLADE model routes queries through a (frozen) SparseStaticEmbedding, whose
    embeddings then don't require grad. The backward hook must skip that route's remaining mini-batches
    (nothing to back-propagate into) instead of crashing, while the document route still trains."""
    model = inference_free_splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    # Freeze the query route, as inference-free training setups commonly do.
    query_route_parameters = [
        parameter for module in model[0].sub_modules["query"] for parameter in module.parameters()
    ]
    for parameter in query_route_parameters:
        parameter.requires_grad_(False)
    query_route_parameter_ids = {id(parameter) for parameter in query_route_parameters}

    queries = ["query a", "query b", "query c", "query d", "query e"]
    documents = ["document a", "document b", "document c", "document d", "document e"]
    features = [model.preprocess(queries, task="query"), model.preprocess(documents, task="document")]

    loss_fn = CachedSpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=3e-5,
        mini_batch_size=2,
        use_document_regularizer_only=False,
        query_regularizer_weight=None,
    )
    model.zero_grad()
    output = loss_fn(features, None)
    torch.stack(list(output.values())).sum().backward()

    document_grads = [
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) not in query_route_parameter_ids and parameter.grad is not None
    ]
    assert document_grads and sum(grad.abs().sum() for grad in document_grads) > 0, (
        "the document route must still receive gradients"
    )
    assert all(parameter.grad is None for parameter in query_route_parameters), (
        "the frozen query route must receive no gradients"
    )
