from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    AdaptiveLayerLoss,
    MatryoshkaLoss,
    MegaBatchMarginLoss,
)
from sentence_transformers.sentence_transformer.modules import Router, StaticEmbedding
from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

ANCHORS = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f", "anchor g"]
POSITIVES = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f", "positive g"]


def _loss_and_grads(
    model: SentenceTransformer, use_mini_batched_version: bool, mini_batch_size: int, batch_size: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad()
    loss = MegaBatchMarginLoss(
        model, use_mini_batched_version=use_mini_batched_version, mini_batch_size=mini_batch_size
    )
    features = [model.preprocess(ANCHORS[:batch_size]), model.preprocess(POSITIVES[:batch_size])]
    loss_value = loss(features, torch.zeros(batch_size, dtype=torch.long))
    loss_value.backward()
    return loss_value.detach(), gradients(model)


@pytest.mark.parametrize(
    ["batch_size", "mini_batch_size"],
    [
        (6, 2),  # even split
        (6, 3),  # even split
        (7, 3),  # uneven split, the trailing mini-batch holds 1 sample
        (7, 5),  # uneven split, the trailing mini-batch holds 2 samples
        (6, 6),  # a single mini-batch
        (6, 50),  # mini_batch_size > batch_size, i.e. the default mini_batch_size on a small batch
    ],
)
def test_mega_batch_margin_mini_batched_matches_non_mini_batched(
    stsb_bert_tiny_model: SentenceTransformer, batch_size: int, mini_batch_size: int
) -> None:
    """``mini_batch_size`` only bounds memory, so it must change neither the loss nor the gradient.

    The mini-batched version embeds each mini-batch without gradients, computes the loss over the whole
    batch, caches the gradients wrt. the embeddings, and re-embeds each mini-batch in a backward hook to
    connect them to the model. The result must be exactly the gradient of the non mini-batched version,
    for any ``mini_batch_size`` and whether or not it divides the batch size.

    Regression test for two bugs. ``forward_mini_batched`` used to slice the non-tensor ``modality``
    marker (``"text"`` -> ``"te"``) and raise ``KeyError``, so on ``main`` the default code path never ran
    at all. Underneath that, its in-loop guard ``end_idx < len(cos_scores)`` was always False
    (``cos_scores`` only had ``mini_batch_size`` rows), so every mini-batch but the last silently
    contributed nothing to the gradient.
    """
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()  # the loss is used during training, so exercise the training-mode code path

    loss_mini, grads_mini = _loss_and_grads(model, True, mini_batch_size, batch_size)
    loss_full, grads_full = _loss_and_grads(model, False, batch_size, batch_size)

    assert_trained(grads_mini)
    assert_trained(grads_full)
    assert loss_mini.item() == pytest.approx(loss_full.item(), rel=1e-4, abs=1e-5)
    for name, grad in grads_mini.items():
        torch.testing.assert_close(grad, grads_full[name], rtol=1e-4, atol=1e-5, msg=name)


def test_mega_batch_margin_replays_dropout_in_the_backward_pass(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The backward pass must re-embed exactly what the forward pass embedded, dropout included.

    The cached gradients are taken wrt. the embeddings of the first (no-grad) forward pass and applied to
    the embeddings of a second one, so the two have to agree bit-for-bit or the gradient is evaluated at
    the wrong activations. ``RandContext`` is what guarantees that, by replaying the RNG state of each
    mini-batch. Every other test here disables dropout, which makes the two passes trivially equal and
    would hide a regression in the replay.
    """
    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    assert any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss = MegaBatchMarginLoss(model, mini_batch_size=3)
    forward_reps: list[torch.Tensor] = []
    backward_reps: list[torch.Tensor] = []
    sink = forward_reps
    embed_minibatch = loss.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss.embed_minibatch = spy

    features = [model.preprocess(ANCHORS), model.preprocess(POSITIVES)]
    loss_value = loss(features, torch.zeros(len(ANCHORS), dtype=torch.long))
    sink = backward_reps  # the hook's re-embedding lands here
    loss_value.backward()

    # 2 columns x ceil(7 / 3) mini-batches, embedded once in the forward pass and once in the backward pass
    assert len(forward_reps) == len(backward_reps) == 6
    for index, (forward_rep, backward_rep) in enumerate(zip(forward_reps, backward_reps)):
        assert torch.equal(forward_rep, backward_rep), f"mini-batch {index} was re-embedded with different dropout"


def test_mega_batch_margin_two_forwards_before_one_backward(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Each forward pass must hand its own cached gradients to its own backward hook.

    The cache used to live on the loss module, so a second forward pass overwrote it and both hooks then
    back-propagated the second batch's gradients against the first batch's inputs, with no error.
    """
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    first = [model.preprocess(ANCHORS[:4]), model.preprocess(POSITIVES[:4])]
    second = [model.preprocess(ANCHORS[3:]), model.preprocess(POSITIVES[3:])]
    labels = torch.zeros(4, dtype=torch.long)

    # One loss object, two forward passes, one backward pass.
    model.zero_grad()
    shared = MegaBatchMarginLoss(model, mini_batch_size=2)
    (shared(first, labels) + shared(second, labels)).backward()
    shared_grads = gradients(model)

    # Two loss objects, which cannot interfere with each other, as the reference.
    model.zero_grad()
    first_loss = MegaBatchMarginLoss(model, mini_batch_size=2)(first, labels)
    second_loss = MegaBatchMarginLoss(model, mini_batch_size=2)(second, labels)
    (first_loss + second_loss).backward()
    reference_grads = gradients(model)

    assert_trained(shared_grads)
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("training", [True, False])
def test_mega_batch_margin_under_no_grad(stsb_bert_tiny_model: SentenceTransformer, training: bool) -> None:
    """The loss must be computable under ``torch.no_grad``, as the trainer does when computing the eval loss.

    The in-loop ``backward()`` used to fire regardless of whether gradients were enabled, raising "element 0
    of tensors does not require grad" whenever the eval batch was larger than ``mini_batch_size``.
    """
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train(training)
    features = [model.preprocess(ANCHORS[:6]), model.preprocess(POSITIVES[:6])]
    labels = torch.zeros(6, dtype=torch.long)

    # mini_batch_size < batch_size, so the loss really is computed over several mini-batches
    with torch.no_grad():
        loss_value = MegaBatchMarginLoss(model, mini_batch_size=2)(features, labels)
        expected = MegaBatchMarginLoss(model, use_mini_batched_version=False)(features, labels)

    assert loss_value.item() == pytest.approx(expected.item(), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("training", [True, False])
def test_mega_batch_margin_preserves_training_mode(stsb_bert_tiny_model: SentenceTransformer, training: bool) -> None:
    """The loss must not change the mode it found the model in.

    It used to call ``self.model.train()`` unconditionally, which left the model in training mode after an
    evaluation, and so computed the eval loss with dropout active for every batch but the first.
    """
    model = stsb_bert_tiny_model.to("cpu")
    model.train(training)

    features = [model.preprocess(ANCHORS[:6]), model.preprocess(POSITIVES[:6])]
    MegaBatchMarginLoss(model, mini_batch_size=2)(features, torch.zeros(6, dtype=torch.long)).backward()

    assert model.training is training


@pytest.mark.parametrize("grad_accum_steps", [2, 4])
def test_mega_batch_margin_scales_with_the_outer_backward(
    stsb_bert_tiny_model: SentenceTransformer, grad_accum_steps: int
) -> None:
    """Every mini-batch must be scaled by whatever the outer backward pass passes in.

    The trainer divides the loss by ``gradient_accumulation_steps`` and, under fp16, hands it to a
    ``GradScaler`` before calling backward. When each mini-batch back-propagated itself inside the loss, it
    missed both, so only the last mini-batch was scaled. Back-propagating from a hook instead means every
    mini-batch is reached by ``grad_output``, so scaling the loss must scale the whole gradient.
    """
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    features = [model.preprocess(ANCHORS[:6]), model.preprocess(POSITIVES[:6])]
    labels = torch.zeros(6, dtype=torch.long)

    def grads(scale: float) -> dict[str, torch.Tensor]:
        model.zero_grad()
        (MegaBatchMarginLoss(model, mini_batch_size=2)(features, labels) * scale).backward()
        return gradients(model)

    unscaled = grads(1.0)
    scaled = grads(1 / grad_accum_steps)

    # Without these, a backward hook that never fires produces no gradient at all, both dicts come back
    # empty, and the comparison below asserts nothing.
    assert_trained(unscaled)
    assert_trained(scaled)
    for name, grad in scaled.items():
        torch.testing.assert_close(grad, unscaled[name] / grad_accum_steps, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("use_mini_batched_version", [True, False])
@pytest.mark.parametrize("num_columns", [1, 3])
def test_mega_batch_margin_requires_exactly_two_columns(
    stsb_bert_tiny_model: SentenceTransformer, use_mini_batched_version: bool, num_columns: int
) -> None:
    """The loss mines its negatives from the other positives in the batch, so it takes exactly 2 columns.

    Rewriting the two forward passes dropped the ``anchor, positive = sentence_features`` unpack that used
    to enforce this. Without it, a third column was embedded and then silently dropped from the objective
    (non mini-batched), or crashed with ``AttributeError: 'NoneType' object has no attribute 'flatten'``
    inside the backward hook (mini-batched), because it never entered the loss graph so its cached
    gradient stayed ``None``.
    """
    model = stsb_bert_tiny_model.to("cpu")
    columns = [ANCHORS[:4], POSITIVES[:4], ["negative a", "negative b", "negative c", "negative d"]]
    features = [model.preprocess(column) for column in columns[:num_columns]]
    loss = MegaBatchMarginLoss(model, use_mini_batched_version=use_mini_batched_version, mini_batch_size=2)

    with pytest.raises(
        ValueError, match=rf"expects exactly 2 input columns, \(anchor, positive\), but got {num_columns}"
    ):
        loss(features, torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("mini_batch_size", [2, 3])
def test_mega_batch_margin_matryoshka(stsb_bert_tiny_model: SentenceTransformer, mini_batch_size: int) -> None:
    """MatryoshkaLoss must route the mini-batched loss through its ``CachedLossDecorator``.

    The mini-batched loss back-propagates from a hook, which fires long after MatryoshkaLoss has restored
    the undecorated ``model.forward``, so the hook re-embedded at the full dimensionality while the cached
    gradients were for the truncated one, raising ``RuntimeError: inconsistent tensor size``. Routing it
    through the decorator instead makes it agree with the non mini-batched version, as everywhere else.
    """
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    features = [model.preprocess(ANCHORS[:6]), model.preprocess(POSITIVES[:6])]
    labels = torch.zeros(6, dtype=torch.long)

    def loss_and_grads(**kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss = MatryoshkaLoss(model, MegaBatchMarginLoss(model, **kwargs), matryoshka_dims=[128, 64, 32])
        loss_value = loss(features, labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    loss_mini, grads_mini = loss_and_grads(mini_batch_size=mini_batch_size)
    loss_full, grads_full = loss_and_grads(use_mini_batched_version=False)

    assert_trained(grads_mini)
    assert loss_mini.item() == pytest.approx(loss_full.item(), rel=1e-4, abs=1e-5)
    for name, grad in grads_mini.items():
        torch.testing.assert_close(grad, grads_full[name], rtol=1e-4, atol=1e-5, msg=name)


def test_mega_batch_margin_adaptive_layer_warns(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """AdaptiveLayerLoss calls its base loss once per layer, and the mini-batched loss defers its backward
    pass to a hook, so the combination is unsupported, exactly as it is for the Cached* losses, which it
    already warns about. It used to construct silently and then die with ``KeyError``.
    """
    model = stsb_bert_tiny_model.to("cpu")

    with pytest.warns(UserWarning, match="AdaptiveLayerLoss is not compatible with MegaBatchMarginLoss"):
        AdaptiveLayerLoss(model, MegaBatchMarginLoss(model, mini_batch_size=2))

    # Without mini-batching there is no cached backward pass, so the combination is supported.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdaptiveLayerLoss(model, MegaBatchMarginLoss(model, use_mini_batched_version=False))
    assert not [warning for warning in caught if "not compatible" in str(warning.message)]


def test_mega_batch_margin_rejects_static_embedding(static_embedding_model: SentenceTransformer) -> None:
    """StaticEmbedding features are an EmbeddingBag (``input_ids``, ``offsets``) with no batch dimension to
    slice along, so the mini-batched version must reject it rather than slice it incorrectly."""
    with pytest.raises(ValueError, match="not compatible with a SentenceTransformer model based on a StaticEmbedding"):
        MegaBatchMarginLoss(static_embedding_model)

    MegaBatchMarginLoss(static_embedding_model, use_mini_batched_version=False)  # the non mini-batched version is fine


def test_mega_batch_margin_rejects_static_embedding_behind_a_router(static_embedding: StaticEmbedding) -> None:
    """A Router keeps its input modules one level down, so a guard that only inspects ``model[0]`` waves a
    StaticEmbedding straight through, and ``_get_batch_size`` then reads the total token count as the
    batch size, mini-batching the EmbeddingBag features by token index."""
    model = SentenceTransformer(
        modules=[Router.for_query_document(query_modules=[static_embedding], document_modules=[static_embedding])]
    )

    with pytest.raises(ValueError, match="not compatible with a SentenceTransformer model based on a StaticEmbedding"):
        MegaBatchMarginLoss(model)

    MegaBatchMarginLoss(model, use_mini_batched_version=False)  # the non mini-batched version is fine


def test_mega_batch_margin_token_budget_matches_non_mini_batched(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Packing the embedding passes by token count must not change the loss or the gradient."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    anchors = ["a", "anchor b with quite a few more words in it", "c and d", "final anchor sentence", "e", "f g h"]
    positives = ["short", "positive b also has a longer surface form", "p", "another positive text", "q r", "s"]
    labels = torch.zeros(6, dtype=torch.long)

    def loss_and_grads(**kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss_fn = MegaBatchMarginLoss(model, **kwargs)
        loss_value = loss_fn([model.preprocess(anchors), model.preprocess(positives)], labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    budget_loss, budget_grads = loss_and_grads(mini_batch_num_tokens=16)
    full_loss, full_grads = loss_and_grads(use_mini_batched_version=False)

    assert_trained(budget_grads)
    assert budget_loss.item() == pytest.approx(full_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in budget_grads.items():
        torch.testing.assert_close(grad, full_grads[name], rtol=1e-4, atol=1e-5, msg=name)
