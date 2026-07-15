from __future__ import annotations

import math

import pytest
import torch

import sentence_transformers.sentence_transformer.losses.cached_gist_embed as cge
from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss


def _make_loss(margin: float, mini_batch_size: int = 32) -> CachedGISTEmbedLoss:
    """Build a CachedGISTEmbedLoss without loading real models.

    ``calculate_loss`` only needs the configuration attributes and operates on
    precomputed reps, so we bypass ``__init__`` (which requires SentenceTransformers).
    """
    obj = CachedGISTEmbedLoss.__new__(CachedGISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.mini_batch_size = mini_batch_size
    obj.show_progress_bar = False
    obj.margin_strategy = "absolute"
    obj.margin = margin
    obj.contrast_anchors = True
    obj.contrast_positives = True
    obj.gather_across_devices = True
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    return obj


@pytest.fixture
def simulate_rank1_world2(monkeypatch):
    """Monkeypatch the distributed helpers so ``calculate_loss`` runs the rank=1/world=2
    gather path in a single process: ``all_gather_with_grad`` prepends a rank-0 block to
    the local (rank-1) block, and the rank is reported as 1 so ``offset = batch_size``.
    """
    per = 3

    def fake_gather(tensor: torch.Tensor) -> torch.Tensor:
        # rank 0's block is deterministic and distinct from the local (rank 1) block.
        g = torch.Generator().manual_seed(1234)
        rank0_block = torch.randn(per, tensor.size(1), generator=g)
        return torch.cat([rank0_block, tensor], dim=0)

    monkeypatch.setattr(cge, "all_gather_with_grad", fake_gather)
    monkeypatch.setattr(cge, "is_dist_initialized", lambda: True)
    # get_rank may be absent on CPU-only/ROCm builds (torch.distributed.is_available() is False).
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    return per


def _local_reps(per: int, dim: int = 16, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(per, dim, generator=g)
    positives = torch.randn(per, dim, generator=g)
    # reps[0] = anchors, reps[1] = positives; each a list of minibatch tensors.
    reps = [[anchors.clone()], [positives.clone()]]
    reps_guided = [[anchors.clone()], [positives.clone()]]
    return reps, reps_guided


@pytest.mark.parametrize("mini_batch_size", [32, 2])
def test_gather_rank1_positive_not_masked_with_margin(simulate_rank1_world2, mini_batch_size):
    """Regression for the rank>0 positive-mask offset bug (gather_across_devices).

    On rank 1, each anchor's true positive lives at gathered column ``offset + row``.
    With ``margin > 0`` the positive itself exceeds ``positive_sim - margin``, so it is
    only spared by ``positive_mask``. If the mask is offset-unaware (the old
    ``roll(begin)``), it protects the wrong columns, the CE-target logit becomes -inf,
    and the loss is +inf. With the offset-aware mask the loss stays finite.

    ``mini_batch_size=2`` splits the local batch so the inner loop runs with ``begin > 0``,
    covering the minibatch-boundary case (where the old flatten-roll could also wrap).
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss = _make_loss(margin=0.1, mini_batch_size=mini_batch_size).calculate_loss(reps, reps_guided)
    assert torch.isfinite(loss), f"rank>0 loss must be finite, got {loss.item()}"


def test_gather_rank1_margin_zero_baseline(simulate_rank1_world2):
    """Sanity baseline: with margin=0 the positive equals its own threshold and is not
    suppressed regardless of the mask, so the loss is finite both before and after the fix.
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss = _make_loss(margin=0.0).calculate_loss(reps, reps_guided)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mini_batch_size", [32, 2])
def test_positive_mask_protects_ce_target_column(simulate_rank1_world2, mini_batch_size):
    """Drive the real loss and assert it protects exactly the CE-target column in every minibatch.

    We capture each minibatch's (masked) score matrix and labels as passed to cross-entropy and
    check that every anchor's CE-target logit survived suppression (is finite). With the old
    offset-unaware ``roll(begin)`` mask, on rank 1 that column is set to -inf, so this fails on the
    pre-fix code. ``mini_batch_size=2`` exercises a ``begin > 0`` chunk.
    """
    per = simulate_rank1_world2
    reps, reps_guided = _local_reps(per)
    loss_fn = _make_loss(margin=0.1, mini_batch_size=mini_batch_size)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        captured.append((scores, labels))
        return real_ce(scores, labels)

    # cross_entropy_loss is registered as an nn.Module child, so drop it before assigning a plain callable.
    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn.calculate_loss(reps, reps_guided)

    assert captured, "cross-entropy was never called"
    for scores, labels in captured:
        target_logits = scores[torch.arange(scores.size(0)), labels]
        assert torch.isfinite(target_logits).all(), (
            f"each anchor's CE-target logit must survive masking, got {target_logits.tolist()}"
        )
    assert torch.isfinite(loss)


def _make_relative_loss(margin: float) -> CachedGISTEmbedLoss:
    """relative-margin CachedGISTEmbedLoss fed precomputed embeddings (#3819)."""
    obj = CachedGISTEmbedLoss.__new__(CachedGISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.mini_batch_size = 32
    obj.show_progress_bar = False
    obj.margin_strategy = "relative"
    obj.margin = margin
    obj.contrast_anchors = False
    obj.contrast_positives = False
    obj.gather_across_devices = False
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    return obj


def _negative_score_reps():
    """anchor0's positive has a negative cosine (-0.50); a non-paired candidate (column 1) is
    *more* similar (-0.49). anchor1 is paired with that candidate (diagonal cosine 1.0)."""
    a0 = [1.0, 0.0, 0.0]
    p0 = [-0.50, math.sqrt(1 - 0.50**2), 0.0]  # cos(a0, p0) = -0.50
    p1 = [-0.49, 0.0, math.sqrt(1 - 0.49**2)]  # cos(a0, p1) = -0.49 (more similar than the positive)
    a1 = p1  # cos(a1, p1) = 1.0
    anchors = torch.tensor([a0, a1])
    positives = torch.tensor([p0, p1])
    reps = [[anchors], [positives]]
    reps_guided = [[anchors], [positives]]
    return reps, reps_guided


def test_relative_margin_negative_positive_score_suppresses_closer_negative():
    """Regression for #3819 in CachedGISTEmbedLoss: with ``margin_strategy="relative"`` and a
    negative positive-pair score, a candidate more similar to the anchor than the positive must
    still be suppressed."""
    loss_fn = _make_relative_loss(margin=0.05)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        captured.append(scores)
        return real_ce(scores, labels)

    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn.calculate_loss(*_negative_score_reps())

    assert captured, "cross-entropy was never called"
    scores = captured[0]
    assert scores[0, 1].item() == float("-inf"), f"closer negative must be masked, got {scores[0, 1].item()}"
    assert torch.isfinite(scores[0, 0]) and torch.isfinite(loss)


def test_cached_gist_two_forwards_before_one_backward(stsb_bert_tiny_model) -> None:
    """Each forward pass must hand its own cached gradients to its own backward hook.

    The cache used to live on the loss module, so a second forward pass overwrote it and both hooks
    then back-propagated the second batch's gradients against the first batch's inputs.
    """
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    anchors = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e"]
    positives = ["positive a", "positive b", "positive c", "positive d", "positive e"]
    first = [model.preprocess(anchors[:3]), model.preprocess(positives[:3])]
    second = [model.preprocess(anchors[2:]), model.preprocess(positives[2:])]
    labels = torch.zeros(3, dtype=torch.long)

    def make_loss() -> CachedGISTEmbedLoss:
        return CachedGISTEmbedLoss(model, guide=model, mini_batch_size=2)

    model.zero_grad()
    shared = make_loss()
    (shared(first, labels) + shared(second, labels)).backward()
    shared_grads = gradients(model)

    model.zero_grad()
    (make_loss()(first, labels) + make_loss()(second, labels)).backward()
    reference_grads = gradients(model)

    assert_trained(shared_grads)
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("num_columns", [2, 3])
@pytest.mark.parametrize("mini_batch_size", [2, 3])
def test_cached_gist_matches_gist(stsb_bert_tiny_model, num_columns: int, mini_batch_size: int) -> None:
    """``mini_batch_size`` only bounds memory: for the 2- and 3-column shapes the non-cached loss
    supports, the cached loss must reproduce GISTEmbedLoss's loss and gradient."""
    from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    columns = [
        ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e"],
        ["positive a", "positive b", "positive c", "positive d", "positive e"],
        ["negative a", "negative b", "negative c", "negative d", "negative e"],
    ][:num_columns]
    labels = torch.zeros(5, dtype=torch.long)

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss_value = loss_fn([model.preprocess(column) for column in columns], labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(
        CachedGISTEmbedLoss(model, guide=model, mini_batch_size=mini_batch_size)
    )
    plain_loss, plain_grads = loss_and_grads(GISTEmbedLoss(model, guide=model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_cached_gist_replays_dropout_in_the_backward_pass(stsb_bert_tiny_model) -> None:
    """The backward pass must re-embed exactly what the forward pass embedded, dropout included,
    also with this loss's bespoke ``embed_minibatch``, which runs the guide model in the
    forward pass but skips it in the backward re-embedding."""
    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    assert any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss_fn = CachedGISTEmbedLoss(model, guide=model, mini_batch_size=2)
    forward_reps: list[torch.Tensor] = []
    backward_reps: list[torch.Tensor] = []
    sink = forward_reps
    embed_minibatch = loss_fn.embed_minibatch

    def spy(**kwargs):
        reps, guide_reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, guide_reps, random_state

    loss_fn.embed_minibatch = spy

    anchors = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e"]
    positives = ["positive a", "positive b", "positive c", "positive d", "positive e"]
    loss = loss_fn([model.preprocess(anchors), model.preprocess(positives)], torch.zeros(5, dtype=torch.long))
    sink = backward_reps  # the hook's re-embedding lands here
    loss.backward()

    # 2 columns x ceil(5 / 2) mini-batches, embedded once per pass
    assert len(forward_reps) == len(backward_reps) == 6
    for index, (forward, backward) in enumerate(zip(forward_reps, backward_reps)):
        assert torch.equal(forward, backward), f"mini-batch {index} was re-embedded with different dropout"


def test_cached_gist_rejects_static_embedding_behind_a_router(static_embedding, stsb_bert_tiny_model) -> None:
    """A Router keeps its input modules one level down, so a guard that only inspects ``model[0]``
    waves a StaticEmbedding straight through, and mini-batching then slices the EmbeddingBag
    features by token index."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Router

    model = SentenceTransformer(
        modules=[Router.for_query_document(query_modules=[static_embedding], document_modules=[static_embedding])]
    )
    with pytest.raises(ValueError, match="not compatible with a SentenceTransformer model based on a StaticEmbedding"):
        CachedGISTEmbedLoss(model, guide=stsb_bert_tiny_model)


@pytest.mark.parametrize("mini_batch_size", [2, 3])
def test_gist_matryoshka_matches_cached_gist_matryoshka(stsb_bert_tiny_model, mini_batch_size: int) -> None:
    """``MatryoshkaLoss(GISTEmbedLoss(...))`` must match ``MatryoshkaLoss(CachedGISTEmbedLoss(...))``.

    The plain loss used to run the guide on the same features dicts whose model outputs
    MatryoshkaLoss's ``ForwardDecorator`` caches, so every dim beyond the first pooled the
    guide's no-grad embeddings and only the largest dim trained. The cached loss embeds fresh
    mini-batch dicts and never had the bug, so it doubles as the reference here.

    The guide is a copy of the model rather than the model itself: MatryoshkaLoss decorates
    ``model.forward``, and a guide sharing that object would have its embeddings shrunk in the
    plain path but not in the cached path."""
    import copy

    from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss, MatryoshkaLoss
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    guide = copy.deepcopy(model)

    anchors = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e"]
    positives = ["positive a", "positive b", "positive c", "positive d", "positive e"]
    labels = torch.zeros(5, dtype=torch.long)

    def loss_and_grads(inner: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        features = [model.preprocess(anchors), model.preprocess(positives)]
        loss_value = MatryoshkaLoss(model, inner, matryoshka_dims=[128, 64, 32])(features, labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    plain_loss, plain_grads = loss_and_grads(GISTEmbedLoss(model, guide=guide))
    cached_loss, cached_grads = loss_and_grads(
        CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=mini_batch_size)
    )

    assert_trained(plain_grads)
    assert plain_loss.item() == pytest.approx(cached_loss.item(), rel=1e-4, abs=1e-5)
    # The Matryoshka sum over 3 dims triples the loss magnitude and with it the float noise.
    for name, grad in plain_grads.items():
        torch.testing.assert_close(grad, cached_grads[name], rtol=1e-4, atol=1e-4, msg=name)


def test_cached_gist_token_budget_matches_gist(stsb_bert_tiny_model) -> None:
    """The token budget must flow through this loss's bespoke ``embed_minibatch_iter`` (which also runs
    the guide model) without changing the loss or gradient."""
    from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    anchors = ["a", "anchor b with quite a few more words in it", "c and d", "final anchor sentence", "e"]
    positives = ["short", "positive b also has a longer surface form", "p", "another positive text", "q r"]
    labels = torch.zeros(5, dtype=torch.long)

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss_value = loss_fn([model.preprocess(anchors), model.preprocess(positives)], labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(CachedGISTEmbedLoss(model, guide=model, mini_batch_num_tokens=16))
    plain_loss, plain_grads = loss_and_grads(GISTEmbedLoss(model, guide=model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)
