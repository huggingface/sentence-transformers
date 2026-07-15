from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from tests.sentence_transformer.losses.utils import disable_dropout as _disable_dropout

ANCHORS = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f"]
POSITIVES = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f"]


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


def test_cached_splade_token_budget_matches_splade(splade_bert_tiny_model: SparseEncoder) -> None:
    """The token budget must reproduce SpladeLoss's per-component losses and gradients."""
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    anchors = ["a", "anchor b with quite a few more words in it", "c and d", "final anchor sentence", "e", "f g"]
    positives = ["short", "positive b also has a longer surface form", "p", "another positive text", "q r", "s"]

    def loss_and_grads(cached: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        model.zero_grad()
        kwargs = {
            "model": model,
            "loss": SparseMultipleNegativesRankingLoss(model),
            "document_regularizer_weight": 3e-5,
            "query_regularizer_weight": 5e-5,
        }
        loss_fn = CachedSpladeLoss(**kwargs, mini_batch_num_tokens=16) if cached else SpladeLoss(**kwargs)
        output = loss_fn([model.preprocess(anchors), model.preprocess(positives)], None)
        torch.stack(list(output.values())).sum().backward()
        grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        return output, grads

    cached_output, cached_grads = loss_and_grads(cached=True)
    plain_output, plain_grads = loss_and_grads(cached=False)

    assert cached_output.keys() == plain_output.keys()
    for key in plain_output:
        assert cached_output[key].item() == pytest.approx(plain_output[key].item(), rel=1e-4, abs=1e-5), key
    assert cached_grads and sum(grad.abs().sum() for grad in cached_grads.values()) > 0
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=2e-4, msg=name)


def test_cached_splade_keeps_dict_base_loss_components(splade_bert_tiny_model: SparseEncoder) -> None:
    """A dict-valued base loss must keep its own component keys in the returned dict, exactly as
    SpladeLoss spreads them. The separate keys are the point of returning a dict: the trainer logs
    each component on its own. The cached loss used to collapse them into a single ``base_loss``."""

    class TwoPartLoss(torch.nn.Module):
        def __init__(self, model: SparseEncoder) -> None:
            super().__init__()
            self.model = model
            self.inner = SparseMultipleNegativesRankingLoss(model)

        def compute_loss_from_embeddings(self, embeddings, labels=None):
            loss = self.inner.compute_loss_from_embeddings(embeddings, labels)
            return {"part_a": loss * 0.25, "part_b": loss * 0.75}

    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    def outputs(cached: bool) -> dict[str, torch.Tensor]:
        kwargs = {
            "model": model,
            "loss": TwoPartLoss(model),
            "document_regularizer_weight": 3e-5,
            "query_regularizer_weight": 5e-5,
        }
        loss_fn = CachedSpladeLoss(**kwargs, mini_batch_size=2) if cached else SpladeLoss(**kwargs)
        return loss_fn([model.preprocess(ANCHORS[:4]), model.preprocess(POSITIVES[:4])], None)

    cached_output = outputs(cached=True)
    plain_output = outputs(cached=False)

    assert set(cached_output) == set(plain_output)
    assert set(cached_output) == {"part_a", "part_b", "document_regularizer_loss", "query_regularizer_loss"}
    for key, value in plain_output.items():
        assert cached_output[key].item() == pytest.approx(value.item(), rel=1e-4, abs=2e-4), key

    # The trainer back-propagates the summed dict, so the invariants of the scalar case must
    # hold for dict-valued base losses too: one gradient carrier, summing exactly to the total.
    assert sum(value.requires_grad for value in cached_output.values()) == 1
    total = torch.stack(list(cached_output.values())).sum()
    assert total.requires_grad
    total.backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())


def test_cached_splade_replays_dropout_in_the_backward_pass(splade_bert_tiny_model: SparseEncoder) -> None:
    """The backward pass must re-embed exactly what the forward pass embedded, dropout included,
    making this loss self-guarding instead of relying only on the shared engine's tests."""
    model = splade_bert_tiny_model.to("cpu")
    model.train()
    assert any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss_fn = _make_loss(model, cached=True, mini_batch_size=2)
    forward_reps: list[torch.Tensor] = []
    backward_reps: list[torch.Tensor] = []
    sink = forward_reps
    embed_minibatch = loss_fn.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss_fn.embed_minibatch = spy

    output = loss_fn([model.preprocess(ANCHORS[:4]), model.preprocess(POSITIVES[:4])], None)
    sink = backward_reps  # the hook's re-embedding lands here
    torch.stack(list(output.values())).sum().backward()

    # 2 columns x ceil(4 / 2) mini-batches, embedded once per pass
    assert len(forward_reps) == len(backward_reps) == 4
    for index, (forward_rep, backward_rep) in enumerate(zip(forward_reps, backward_reps)):
        assert torch.equal(forward_rep, backward_rep), f"mini-batch {index} was re-embedded with different dropout"
