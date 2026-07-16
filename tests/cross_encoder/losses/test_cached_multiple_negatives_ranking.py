from __future__ import annotations

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
from tests.sentence_transformer.losses.utils import disable_dropout

QUERIES = ["query a", "query b", "query c", "query d", "query e", "query f"]
ANSWERS = ["answer a", "answer b", "answer c", "answer d", "answer e", "answer f"]


def _gradients(model: CrossEncoder) -> dict[str, torch.Tensor]:
    return {name: param.grad.clone() for name, param in model.model.named_parameters() if param.grad is not None}


def _loss_and_grads(model: CrossEncoder, loss_fn: torch.nn.Module, seed: int = 12) -> tuple[float, dict]:
    # The in-batch negatives are sampled randomly inside forward, so seed for comparability.
    model.model.zero_grad()
    torch.manual_seed(seed)
    loss = loss_fn([QUERIES, ANSWERS], None)
    loss.backward()
    return loss.item(), _gradients(model)


def _disable_dropout(model: CrossEncoder) -> None:
    disable_dropout(model, canary_inputs=[["a canary query", "a canary answer"], ["another query", "another answer"]])


@pytest.mark.parametrize("mini_batch_size", [2, 5, 64])
def test_cached_ce_mnrl_matches_mnrl(reranker_bert_tiny_model_v54: CrossEncoder, mini_batch_size: int) -> None:
    """``mini_batch_size`` only bounds memory: the cached loss must reproduce MultipleNegativesRankingLoss's
    loss and gradient, whether or not it divides the pair count."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.train()

    cached_loss, cached_grads = _loss_and_grads(
        model, CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    )
    plain_loss, plain_grads = _loss_and_grads(model, MultipleNegativesRankingLoss(model))

    assert cached_grads, "no parameter received a gradient"
    assert sum(grad.abs().sum() for grad in cached_grads.values()) > 0
    assert cached_loss == pytest.approx(plain_loss, rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device RNG replay can only regress on an accelerator")
def test_cached_ce_mnrl_replays_dropout_on_cuda(reranker_bert_tiny_model_v54: CrossEncoder) -> None:
    """The backward pass must re-predict exactly the logits of the forward pass, dropout included.

    ``RandContext`` can only capture the RNG state of devices it sees tensors for. It used to be handed
    the raw string pairs, before tokenization, so it captured no device state at all, and on CUDA the
    backward hook's second forward pass drew different dropout masks than the logits the cached gradients
    belong to: silently wrong gradients for every reranker trained with dropout on a GPU.

    Note that the project CI has no GPU job, so this regression test only guards the fix on machines
    with CUDA. The CPU equivalence tests cannot catch it: the bug only manifests on an accelerator.
    """
    model = reranker_bert_tiny_model_v54
    model.model.to("cuda")
    model.model.train()
    assert any(module.p > 0 for module in model.model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3)
    forward_logits: list[torch.Tensor] = []
    backward_logits: list[torch.Tensor] = []
    sink = forward_logits
    predict_minibatch = loss_fn.predict_minibatch

    def spy(**kwargs):
        logits, random_state = predict_minibatch(**kwargs)
        sink.append(logits.detach().clone())
        return logits, random_state

    loss_fn.predict_minibatch = spy

    loss = loss_fn([QUERIES, ANSWERS], None)
    sink = backward_logits  # the hook's re-prediction lands here
    loss.backward()

    assert len(forward_logits) == len(backward_logits) > 0
    for index, (forward, backward) in enumerate(zip(forward_logits, backward_logits)):
        assert torch.equal(forward, backward), (
            f"mini-batch {index} was re-predicted with different dropout: "
            f"max|diff|={(forward - backward).abs().max().item():.3e}"
        )


def test_cached_ce_mnrl_two_forwards_before_one_backward(reranker_bert_tiny_model_v54: CrossEncoder) -> None:
    """Each forward pass must hand its own cached gradients to its own backward hook. The cache used to
    live on the loss module, where a second forward pass overwrote it."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.train()

    def losses(loss_fn_first: torch.nn.Module, loss_fn_second: torch.nn.Module) -> dict[str, torch.Tensor]:
        model.model.zero_grad()
        torch.manual_seed(12)
        first = loss_fn_first([QUERIES[:3], ANSWERS[:3]], None)
        torch.manual_seed(34)
        second = loss_fn_second([QUERIES[3:], ANSWERS[3:]], None)
        (first + second).backward()
        return _gradients(model)

    shared = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    shared_grads = losses(shared, shared)
    reference_grads = losses(
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=2),
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=2),
    )

    assert shared_grads and sum(grad.abs().sum() for grad in shared_grads.values()) > 0
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("grad_accum_steps", [2, 4])
def test_cached_ce_mnrl_scales_with_the_outer_backward(
    reranker_bert_tiny_model_v54: CrossEncoder, grad_accum_steps: int
) -> None:
    """Scaling the returned loss (gradient accumulation, fp16 loss scaling) must scale the whole gradient."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.train()

    def grads(scale: float) -> dict[str, torch.Tensor]:
        model.model.zero_grad()
        torch.manual_seed(12)
        (CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)([QUERIES, ANSWERS], None) * scale).backward()
        return _gradients(model)

    unscaled = grads(1.0)
    scaled = grads(1 / grad_accum_steps)

    assert unscaled and sum(grad.abs().sum() for grad in unscaled.values()) > 0
    for name, grad in scaled.items():
        torch.testing.assert_close(grad, unscaled[name] / grad_accum_steps, rtol=1e-4, atol=1e-6, msg=name)


def test_cached_ce_mnrl_under_no_grad(reranker_bert_tiny_model_v54: CrossEncoder) -> None:
    """The loss must be computable under ``torch.no_grad``, as the trainer does for the eval loss."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.eval()

    with torch.no_grad():
        torch.manual_seed(12)
        cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)([QUERIES, ANSWERS], None)
        torch.manual_seed(12)
        plain = MultipleNegativesRankingLoss(model)([QUERIES, ANSWERS], None)

    assert cached.item() == pytest.approx(plain.item(), rel=1e-4, abs=1e-5)


NEGATIVES = ["wrong a", "wrong b", "wrong c", "wrong d", "wrong e", "wrong f"]


@pytest.mark.parametrize("mini_batch_size", [2, 5])
def test_cached_ce_mnrl_with_hard_negatives(reranker_bert_tiny_model_v54: CrossEncoder, mini_batch_size: int) -> None:
    """Explicit hard negatives (a third input column) triple the pair count. The cached loss must still
    reproduce the non-cached loss and gradient."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.train()

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[float, dict]:
        model.model.zero_grad()
        torch.manual_seed(12)
        loss = loss_fn([QUERIES, ANSWERS, NEGATIVES], None)
        loss.backward()
        return loss.item(), _gradients(model)

    cached_loss, cached_grads = loss_and_grads(
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    )
    plain_loss, plain_grads = loss_and_grads(MultipleNegativesRankingLoss(model))

    assert cached_grads and sum(grad.abs().sum() for grad in cached_grads.values()) > 0
    assert cached_loss == pytest.approx(plain_loss, rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_cached_ce_mnrl_prompt_reaches_the_backward_pass(reranker_bert_tiny_model_v54: CrossEncoder) -> None:
    """The prompt (and task) must be carried into the backward hook's re-prediction: they change the
    tokenization, so if the hook's bundle dropped them, the re-predicted logits would belong to
    different inputs than the cached gradients."""
    model = reranker_bert_tiny_model_v54
    _disable_dropout(model)
    model.model.train()

    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    forward_logits: list[torch.Tensor] = []
    backward_logits: list[torch.Tensor] = []
    sink = forward_logits
    predict_minibatch = loss_fn.predict_minibatch

    def spy(**kwargs):
        logits, random_state = predict_minibatch(**kwargs)
        sink.append(logits.detach().clone())
        return logits, random_state

    loss_fn.predict_minibatch = spy

    torch.manual_seed(12)
    loss = loss_fn([QUERIES, ANSWERS], None, prompt="Rerank this passage: ")
    sink = backward_logits
    loss.backward()

    assert len(forward_logits) == len(backward_logits) > 0
    for index, (forward, backward) in enumerate(zip(forward_logits, backward_logits)):
        assert torch.equal(forward, backward), f"mini-batch {index} was re-predicted with different inputs"
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.model.parameters())
