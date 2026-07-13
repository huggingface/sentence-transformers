from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    AdaptiveLayerLoss,
    BatchAllTripletLoss,
    CachedMultipleNegativesRankingLoss,
    ContrastiveLoss,
    CoSENTLoss,
    CosineSimilarityLoss,
    GISTEmbedLoss,
    GlobalOrthogonalRegularizationLoss,
    GradCacheLoss,
    MatryoshkaLoss,
    MegaBatchMarginLoss,
    MSELoss,
    MultipleNegativesRankingLoss,
    SoftmaxLoss,
    TripletLoss,
)
from sentence_transformers.sentence_transformer.modules import Router, StaticEmbedding
from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

COLUMN_A = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f", "anchor g"]
COLUMN_B = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f", "positive g"]
COLUMN_C = ["negative a", "negative b", "negative c", "negative d", "negative e", "negative f", "negative g"]

# (inner loss factory, number of input columns, labels factory) for losses the wrapper must reproduce exactly
WRAPPABLE_LOSSES = {
    "mnrl": (lambda model: MultipleNegativesRankingLoss(model), 2, lambda n: torch.zeros(n, dtype=torch.long)),
    "cosine": (lambda model: CosineSimilarityLoss(model), 2, lambda n: torch.linspace(0, 1, n)),
    "triplet": (lambda model: TripletLoss(model), 3, lambda n: torch.zeros(n, dtype=torch.long)),
    "cosent": (lambda model: CoSENTLoss(model), 2, lambda n: torch.linspace(0, 1, n)),
    "contrastive": (lambda model: ContrastiveLoss(model), 2, lambda n: torch.arange(n) % 2),
    "batch_all_triplet": (lambda model: BatchAllTripletLoss(model), 1, lambda n: torch.arange(n) // 2),
}


def _features(model: SentenceTransformer, num_columns: int, batch_size: int) -> list[dict[str, Tensor]]:
    return [model.preprocess(column[:batch_size]) for column in (COLUMN_A, COLUMN_B, COLUMN_C)[:num_columns]]


def _loss_and_grads(
    model: SentenceTransformer, loss_fn: torch.nn.Module, num_columns: int, labels: Tensor, batch_size: int
) -> tuple[Tensor, dict[str, Tensor]]:
    model.zero_grad()
    loss_value = loss_fn(_features(model, num_columns, batch_size), labels)
    if isinstance(loss_value, dict):
        loss_value = torch.stack(list(loss_value.values())).sum()
    loss_value.backward()
    return loss_value.detach(), gradients(model)


@pytest.mark.parametrize("inner", WRAPPABLE_LOSSES.keys())
@pytest.mark.parametrize("mini_batch_size", [2, 3, 5, 50])
def test_gradcache_matches_the_wrapped_loss(
    stsb_bert_tiny_model: SentenceTransformer, inner: str, mini_batch_size: int
) -> None:
    """``GradCacheLoss(model, loss)`` must produce the loss and gradient of the wrapped loss itself.

    ``mini_batch_size`` only bounds memory, so this must hold whether it divides the batch size (7)
    or not, and also when it exceeds it. The tolerance covers the float noise of computing the same
    embeddings in differently-sized GEMMs, which is of the same magnitude between
    CachedMultipleNegativesRankingLoss and MultipleNegativesRankingLoss.
    """
    make_loss, num_columns, make_labels = WRAPPABLE_LOSSES[inner]
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = make_labels(7)

    wrapped_loss, wrapped_grads = _loss_and_grads(
        model, GradCacheLoss(model, make_loss(model), mini_batch_size=mini_batch_size), num_columns, labels, 7
    )
    plain_loss, plain_grads = _loss_and_grads(model, make_loss(model), num_columns, labels, 7)

    assert_trained(wrapped_grads)
    assert_trained(plain_grads)
    assert wrapped_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in wrapped_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"scale": 50.0, "directions": ("query_to_doc", "doc_to_query"), "partition_mode": "per_direction"},
        {"hardness_mode": "in_batch_negatives", "hardness_strength": 9.0},
    ],
)
def test_gradcache_matches_cached_mnrl(stsb_bert_tiny_model: SentenceTransformer, config: dict) -> None:
    """``GradCacheLoss(model, MultipleNegativesRankingLoss(model, **config))`` is a drop-in replacement
    for ``CachedMultipleNegativesRankingLoss(model, **config)``. The two differ only in the float
    reduction order of the loss stage, which the dedicated class additionally chunks."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(7, dtype=torch.long)

    wrapper = GradCacheLoss(model, MultipleNegativesRankingLoss(model, **config), mini_batch_size=3)
    wrapped_loss, wrapped_grads = _loss_and_grads(model, wrapper, 2, labels, 7)
    cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3, **config)
    cached_loss, cached_grads = _loss_and_grads(model, cached, 2, labels, 7)

    assert_trained(wrapped_grads)
    assert wrapped_loss.item() == pytest.approx(cached_loss.item(), rel=1e-5, abs=1e-6)
    for name, grad in wrapped_grads.items():
        torch.testing.assert_close(grad, cached_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_gradcache_replays_dropout_in_the_backward_pass(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The backward pass must re-embed exactly what the forward pass embedded, dropout included; the
    cached gradients belong to the first pass's embeddings, so the two must agree bit-for-bit."""
    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    assert any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=3)
    forward_reps: list[Tensor] = []
    backward_reps: list[Tensor] = []
    sink = forward_reps
    embed_minibatch = loss.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss.embed_minibatch = spy

    loss_value = loss(_features(model, 2, 7), torch.zeros(7, dtype=torch.long))
    sink = backward_reps  # the hook's re-embedding lands here
    loss_value.backward()

    # 2 columns x ceil(7 / 3) mini-batches, embedded once per pass
    assert len(forward_reps) == len(backward_reps) == 6
    for index, (forward_rep, backward_rep) in enumerate(zip(forward_reps, backward_reps)):
        assert torch.equal(forward_rep, backward_rep), f"mini-batch {index} was re-embedded with different dropout"


def test_gradcache_two_forwards_before_one_backward(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Each forward pass must hand its own cached gradients to its own backward hook, so summing two
    forward passes of one loss object must equal using two separate loss objects."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    first = _features(model, 2, 4)
    second = [model.preprocess(COLUMN_A[3:]), model.preprocess(COLUMN_B[3:])]
    labels = torch.zeros(4, dtype=torch.long)

    model.zero_grad()
    shared = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)
    (shared(first, labels) + shared(second, labels)).backward()
    shared_grads = gradients(model)

    model.zero_grad()
    first_loss = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)(first, labels)
    second_loss = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)(second, labels)
    (first_loss + second_loss).backward()
    reference_grads = gradients(model)

    assert_trained(shared_grads)
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("grad_accum_steps", [2, 4])
def test_gradcache_scales_with_the_outer_backward(
    stsb_bert_tiny_model: SentenceTransformer, grad_accum_steps: int
) -> None:
    """Scaling the returned loss (gradient accumulation, fp16 loss scaling) must scale the whole
    gradient: the backward hook is what carries ``grad_output`` to every mini-batch."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    features = _features(model, 2, 6)
    labels = torch.zeros(6, dtype=torch.long)

    def grads(scale: float) -> dict[str, Tensor]:
        model.zero_grad()
        loss = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)
        (loss(features, labels) * scale).backward()
        return gradients(model)

    unscaled = grads(1.0)
    scaled = grads(1 / grad_accum_steps)

    # Without these, a backward hook that never fires produces no gradient at all, both dicts come
    # back empty, and the comparison below asserts nothing.
    assert_trained(unscaled)
    assert_trained(scaled)
    for name, grad in scaled.items():
        torch.testing.assert_close(grad, unscaled[name] / grad_accum_steps, rtol=1e-4, atol=1e-6, msg=name)


def test_gradcache_matryoshka(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """MatryoshkaLoss must route the wrapper through its ``CachedLossDecorator`` (the hook fires after
    Matryoshka's forward patching is undone), and the result must match Matryoshka over the plain loss."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(6, dtype=torch.long)
    dims = [128, 64, 32]

    model.zero_grad()
    wrapped = MatryoshkaLoss(
        model, GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2), matryoshka_dims=dims
    )
    assert wrapped.uses_gradient_cache
    wrapped_loss = wrapped(_features(model, 2, 6), labels)
    wrapped_loss.backward()
    wrapped_grads = gradients(model)

    model.zero_grad()
    plain = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims)
    plain_loss = plain(_features(model, 2, 6), labels)
    plain_loss.backward()
    plain_grads = gradients(model)

    assert_trained(wrapped_grads)
    assert wrapped_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    # The Matryoshka sum over 3 dims triples the loss magnitude, and with it the float noise between
    # the CachedLossDecorator and ForwardDecorator paths; grads here are O(10), so atol=1e-4 is tight.
    for name, grad in wrapped_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-4, msg=name)


def test_gradcache_adaptive_layer_warns(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """AdaptiveLayerLoss calls its base loss once per layer, which gradient caching cannot support; it
    must warn for the wrapper exactly as it does for the Cached* losses."""
    model = stsb_bert_tiny_model.to("cpu")
    with pytest.warns(UserWarning, match="AdaptiveLayerLoss is not compatible with GradCacheLoss"):
        AdaptiveLayerLoss(model, GradCacheLoss(model, MultipleNegativesRankingLoss(model)))


def test_gradcache_dict_valued_loss(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A wrapped loss that returns per-component dicts must keep its components visible for logging,
    with the dict still summing to the (gradient-carrying) total."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(6, dtype=torch.long)

    wrapper = GradCacheLoss(model, GlobalOrthogonalRegularizationLoss(model), mini_batch_size=2)
    model.zero_grad()
    output = wrapper(_features(model, 2, 6), labels)
    assert isinstance(output, dict)
    total = torch.stack(list(output.values())).sum()
    total.backward()
    wrapped_grads = gradients(model)
    assert_trained(wrapped_grads)

    plain = GlobalOrthogonalRegularizationLoss(model)
    model.zero_grad()
    plain_output = plain(_features(model, 2, 6), labels)
    assert output.keys() == plain_output.keys()
    plain_total = torch.stack(list(plain_output.values())).sum()
    plain_total.backward()
    plain_grads = gradients(model)

    assert total.item() == pytest.approx(plain_total.item(), rel=1e-4, abs=1e-5)
    for name, grad in wrapped_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize("training", [True, False])
def test_gradcache_under_no_grad(stsb_bert_tiny_model: SentenceTransformer, training: bool) -> None:
    """The loss must be computable under ``torch.no_grad``, as the trainer does for the eval loss."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train(training)
    labels = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        wrapped_loss = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)(
            _features(model, 2, 6), labels
        )
        plain_loss = MultipleNegativesRankingLoss(model)(_features(model, 2, 6), labels)

    assert wrapped_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    ["make_loss", "match"],
    [
        (lambda model: GISTEmbedLoss(model, model), "Use CachedGISTEmbedLoss instead"),
        (lambda model: SoftmaxLoss(model, model.get_embedding_dimension(), 2), "trainable parameters of its own"),
        (lambda model: CachedMultipleNegativesRankingLoss(model), "already uses gradient caching"),
        (lambda model: MegaBatchMarginLoss(model), "already uses gradient caching"),
        (
            lambda model: MegaBatchMarginLoss(model, use_mini_batched_version=False),
            "does not expose compute_loss_from_embeddings",
        ),
        (lambda model: MSELoss(model), "'teacher_embeddings', not 'labels'"),
    ],
)
def test_gradcache_rejects_unwrappable_losses(
    stsb_bert_tiny_model: SentenceTransformer,
    make_loss: Callable[[SentenceTransformer], torch.nn.Module],
    match: str,
) -> None:
    """Losses that gradient caching cannot support must be rejected at construction with a pointed error:
    a second embedding model, own trainable parameters (their gradients would miss the loss scaling the
    backward hook applies), double caching, or a ``compute_loss_from_embeddings`` that cannot receive
    the trainer's labels."""
    model = stsb_bert_tiny_model.to("cpu")
    with pytest.raises(ValueError, match=match):
        GradCacheLoss(model, make_loss(model))


def test_gradcache_rejects_splade_loss(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """SpladeLoss is scheduled by the sparse trainer via an isinstance check that a wrapper would hide,
    and CachedSpladeLoss already exists for it."""
    from sentence_transformers.sparse_encoder.losses import SpladeLoss

    model = stsb_bert_tiny_model.to("cpu")
    splade = SpladeLoss(model, MultipleNegativesRankingLoss(model), document_regularizer_weight=3e-5)
    with pytest.raises(ValueError, match="Use CachedSpladeLoss instead"):
        GradCacheLoss(model, splade)


def test_gradcache_rejects_static_embedding(static_embedding_model: SentenceTransformer) -> None:
    """StaticEmbedding features are an EmbeddingBag (``input_ids``, ``offsets``) with no batch dimension
    to slice along, so the wrapper must reject such models rather than mini-batch them incorrectly."""
    with pytest.raises(ValueError, match="not compatible with a SentenceTransformer model based on a StaticEmbedding"):
        GradCacheLoss(static_embedding_model, MultipleNegativesRankingLoss(static_embedding_model))


def test_gradcache_rejects_static_embedding_behind_a_router(static_embedding: StaticEmbedding) -> None:
    """A Router keeps its input modules one level down, so a guard that only inspects ``model[0]``
    would wave a StaticEmbedding straight through."""
    model = SentenceTransformer(
        modules=[Router.for_query_document(query_modules=[static_embedding], document_modules=[static_embedding])]
    )
    with pytest.raises(ValueError, match="not compatible with a SentenceTransformer model based on a StaticEmbedding"):
        GradCacheLoss(model, MultipleNegativesRankingLoss(model))


def test_gradcache_get_config_dict(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The config dict passes the inner loss module itself, so the model card expands it with the inner
    loss's own hyperparameters instead of losing them."""
    model = stsb_bert_tiny_model.to("cpu")
    inner = MultipleNegativesRankingLoss(model, scale=42.0)
    config = GradCacheLoss(model, inner, mini_batch_size=8).get_config_dict()
    assert config == {"loss": inner, "mini_batch_size": 8}


def test_gradcache_under_autocast(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Under autocast, the cached gradients are reduced-precision while the backward hook's
    re-embedding runs outside autocast in fp32; the surrogate must bridge the dtypes.

    A LayerNorm-final model exits autocast in fp32 and would pass vacuously, so append a Dense
    module: its Linear runs in bf16, making the cached embeddings (and their gradients) bf16.
    """
    from sentence_transformers.sentence_transformer.modules import Dense

    model = stsb_bert_tiny_model.to("cpu")
    model.append(Dense(model.get_embedding_dimension(), 64, activation_function=torch.nn.Tanh()))
    disable_dropout(model)
    model.train()
    labels = torch.zeros(6, dtype=torch.long)

    loss_fn = GradCacheLoss(model, MultipleNegativesRankingLoss(model), mini_batch_size=2)
    model.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(_features(model, 2, 6), labels)
        assert loss.dtype == torch.bfloat16, "the test premise requires reduced-precision embeddings"
    assert torch.isfinite(loss)
    loss.backward()

    grads = gradients(model)
    assert_trained(grads)
    assert all(torch.isfinite(grad).all() for grad in grads.values())


def test_gradcache_rejects_a_loss_built_with_a_different_model(
    stsb_bert_tiny_model: SentenceTransformer,
) -> None:
    """Wrapping a loss that embeds with a different model would cache gradients for embeddings the
    wrapped loss never saw; the mismatch must be named, not reported as 'own trainable parameters'."""
    import copy

    model = stsb_bert_tiny_model.to("cpu")
    other_model = copy.deepcopy(model)
    with pytest.raises(ValueError, match="initialized with a different model"):
        GradCacheLoss(model, MultipleNegativesRankingLoss(other_model))


def test_gradcache_names_the_unused_column(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A wrapped loss that ignores an input column (e.g. a single-column loss handed extra columns)
    must raise a pointed error at forward time, not crash on a None gradient inside loss.backward()."""
    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    loss = GradCacheLoss(model, BatchAllTripletLoss(model), mini_batch_size=2)

    with pytest.raises(ValueError, match=r"did not use input column\(s\) 1"):
        loss(_features(model, 2, 6), torch.arange(6) // 2)
