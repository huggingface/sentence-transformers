"""Tests for the shared GradCache engine (sentence_transformers/base/losses/gradcache.py), exercised
through CachedMultipleNegativesRankingLoss, its thinnest consumer. Loss-specific behavior lives in the
per-loss test files. This file pins the engine invariants: gradient equivalence, RNG replay,
per-forward cache isolation, grad_output scaling, token-budget boundaries, and mask non-mutation."""

from __future__ import annotations

import sys

import pytest
import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.losses.gradcache import CachedLossMixin, _create_minibatch, _minibatch_ranges
from sentence_transformers.sentence_transformer.losses import (
    AdaptiveLayerLoss,
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

COLUMN_A = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f", "anchor g"]
COLUMN_B = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f", "positive g"]

# Variable-length columns, so token-budget boundaries genuinely differ from fixed-size ones.
VARIED_A = ["a", "anchor b with quite a few more words in it", "c and d", "final anchor sentence", "e", "f g h"]
VARIED_B = ["short", "positive b also has a longer surface form here", "p", "another positive text", "q r", "s"]


def _loss_and_grads(
    model: SentenceTransformer, loss_fn: torch.nn.Module, columns: tuple[list[str], list[str]], labels: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    model.zero_grad()
    loss_value = loss_fn([model.preprocess(column) for column in columns], labels)
    loss_value.backward()
    return loss_value.detach(), gradients(model)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"scale": 50.0, "directions": ("query_to_doc", "doc_to_query"), "partition_mode": "per_direction"},
        {"directions": ("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc")},
        {"hardness_mode": "in_batch_negatives", "hardness_strength": 9.0},
        {"hardness_mode": "all_negatives", "hardness_strength": 5.0},
    ],
)
@pytest.mark.parametrize("mini_batch_size", [3, 50])
def test_cached_mnrl_matches_mnrl(
    stsb_bert_tiny_model: SentenceTransformer, config: dict, mini_batch_size: int
) -> None:
    """Gradient caching must produce the loss and gradient of the non-cached loss, for any
    configuration and whether or not ``mini_batch_size`` divides (or exceeds) the batch size."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(7, dtype=torch.long)

    cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size, **config)
    cached_loss, cached_grads = _loss_and_grads(model, cached, (COLUMN_A, COLUMN_B), labels)
    plain = MultipleNegativesRankingLoss(model, **config)
    plain_loss, plain_grads = _loss_and_grads(model, plain, (COLUMN_A, COLUMN_B), labels)

    assert_trained(cached_grads)
    assert_trained(plain_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_gradcache_replays_dropout_in_the_backward_pass(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The backward pass must re-embed exactly what the forward pass embedded, dropout included. The
    cached gradients belong to the first pass's embeddings, so the two must agree bit-for-bit."""
    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    assert any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)), (
        "the model has no active dropout, so this test would be vacuous"
    )

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3)
    forward_reps: list[Tensor] = []
    backward_reps: list[Tensor] = []
    sink = forward_reps
    embed_minibatch = loss.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss.embed_minibatch = spy

    loss_value = loss([model.preprocess(COLUMN_A), model.preprocess(COLUMN_B)], torch.zeros(7, dtype=torch.long))
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

    first = [model.preprocess(COLUMN_A[:4]), model.preprocess(COLUMN_B[:4])]
    second = [model.preprocess(COLUMN_A[3:]), model.preprocess(COLUMN_B[3:])]
    labels = torch.zeros(4, dtype=torch.long)

    model.zero_grad()
    shared = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    (shared(first, labels) + shared(second, labels)).backward()
    shared_grads = gradients(model)

    model.zero_grad()
    first_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)(first, labels)
    second_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)(second, labels)
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
    features = [model.preprocess(COLUMN_A[:6]), model.preprocess(COLUMN_B[:6])]
    labels = torch.zeros(6, dtype=torch.long)

    def grads(scale: float) -> dict[str, Tensor]:
        model.zero_grad()
        (CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)(features, labels) * scale).backward()
        return gradients(model)

    unscaled = grads(1.0)
    scaled = grads(1 / grad_accum_steps)

    # Without these, a backward hook that never fires produces no gradient at all, both dicts come
    # back empty, and the comparison below asserts nothing.
    assert_trained(unscaled)
    assert_trained(scaled)
    for name, grad in scaled.items():
        torch.testing.assert_close(grad, unscaled[name] / grad_accum_steps, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("training", [True, False])
def test_gradcache_under_no_grad(stsb_bert_tiny_model: SentenceTransformer, training: bool) -> None:
    """The loss must be computable under ``torch.no_grad``, as the trainer does for the eval loss."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train(training)
    labels = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        cached_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)(
            [model.preprocess(COLUMN_A[:6]), model.preprocess(COLUMN_B[:6])], labels
        )
        plain_loss = MultipleNegativesRankingLoss(model)(
            [model.preprocess(COLUMN_A[:6]), model.preprocess(COLUMN_B[:6])], labels
        )

    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="bfloat16 CPU matmul can hard-crash (0xc000001d) on some Windows machines. Skipping to avoid CI failures.",
)
def test_gradcache_under_autocast(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Under autocast, the cached gradients are reduced-precision while the backward hook's
    re-embedding runs outside autocast in fp32, so the surrogate must bridge the dtypes.

    A LayerNorm-final model exits autocast in fp32 and would pass vacuously, so append a Dense
    module: its Linear runs in bf16, making the cached embeddings (and their gradients) bf16.
    """
    from sentence_transformers.sentence_transformer.modules import Dense

    model = stsb_bert_tiny_model.to("cpu")
    model.append(Dense(model.get_embedding_dimension(), 64, activation_function=torch.nn.Tanh()))
    disable_dropout(model)
    model.train()
    labels = torch.zeros(6, dtype=torch.long)

    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    model.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn([model.preprocess(COLUMN_A[:6]), model.preprocess(COLUMN_B[:6])], labels)
        assert loss.dtype == torch.bfloat16, "the test premise requires reduced-precision embeddings"
    assert torch.isfinite(loss)
    loss.backward()

    grads = gradients(model)
    assert_trained(grads)
    assert all(torch.isfinite(grad).all() for grad in grads.values())


def test_adaptive_layer_warns_for_gradient_cached_losses(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """AdaptiveLayerLoss calls its base loss once per layer, which gradient caching cannot support. The
    ``uses_gradient_cache`` capability check must catch any engine consumer, not a hardcoded list."""
    model = stsb_bert_tiny_model.to("cpu")
    with pytest.warns(UserWarning, match="AdaptiveLayerLoss is not compatible with CachedMultipleNegatives"):
        AdaptiveLayerLoss(model, CachedMultipleNegativesRankingLoss(model))


def test_gradcache_names_the_unused_column(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A ``calculate_loss`` that ignores an input column leaves its embeddings without gradients. The
    mixin must raise a pointed error at forward time, not crash on a None gradient inside
    ``loss.backward()``. None of the first-party losses ignore columns, so pin the engine guard with a
    minimal consumer."""

    class FirstColumnOnlyLoss(CachedLossMixin, nn.Module):
        def __init__(self, model: SentenceTransformer) -> None:
            super().__init__()
            self.model = model
            self.mini_batch_size = 2

        def calculate_loss(self, reps, labels=None, *, with_backward=False):
            loss = torch.cat(reps[0]).pow(2).mean()  # ignores reps[1]
            if with_backward:
                loss.backward()
                loss = loss.detach()
            return loss

        def forward(self, sentence_features, labels):
            return self.forward_cached(sentence_features, labels)

    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    loss = FirstColumnOnlyLoss(model)

    with pytest.raises(ValueError, match=r"did not use input column\(s\) 1"):
        loss([model.preprocess(COLUMN_A[:4]), model.preprocess(COLUMN_B[:4])], None)


class TestMinibatchRanges:
    """Unit tests for the token-budget mini-batch boundary computation."""

    def _padded(self, lengths: list[int]) -> dict[str, torch.Tensor]:
        width = max(lengths)
        mask = torch.zeros(len(lengths), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"input_ids": torch.ones_like(mask), "attention_mask": mask}

    def _flattened(self, lengths: list[int]) -> dict[str, torch.Tensor]:
        cu_seq_lens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.long)
        return {"input_ids": torch.ones(1, int(cu_seq_lens[-1])), "cu_seq_lens_q": cu_seq_lens}

    def test_fixed_size_splitting_unchanged(self) -> None:
        ranges = _minibatch_ranges(self._padded([3] * 7), mini_batch_size=3)
        assert ranges == [(0, 3), (3, 6), (6, 7)]

    def test_padded_and_flattened_agree(self) -> None:
        lengths = [5, 3, 8, 2, 9, 1, 4]
        padded = _minibatch_ranges(self._padded(lengths), 2, mini_batch_num_tokens=10)
        flattened = _minibatch_ranges(self._flattened(lengths), 2, mini_batch_num_tokens=10)
        # 5+3=8, then 8+2=10 and 9+1=10 fill the budget exactly, and 4 remains
        assert padded == flattened == [(0, 2), (2, 4), (4, 6), (6, 7)]

    def test_padding_does_not_count(self) -> None:
        # Rows are padded to width 9, but only the real tokens fill the budget.
        ranges = _minibatch_ranges(self._padded([2, 2, 9, 2]), 1, mini_batch_num_tokens=4)
        assert ranges == [(0, 2), (2, 3), (3, 4)]

    def test_oversized_sequence_gets_its_own_minibatch(self) -> None:
        ranges = _minibatch_ranges(self._padded([2, 50, 2]), 1, mini_batch_num_tokens=4)
        assert ranges == [(0, 1), (1, 2), (2, 3)]

    def test_budget_larger_than_the_batch(self) -> None:
        ranges = _minibatch_ranges(self._padded([3, 3, 3]), 1, mini_batch_num_tokens=1000)
        assert ranges == [(0, 3)]

    def test_ranges_cover_the_batch_exactly(self) -> None:
        lengths = [7, 1, 3, 9, 2, 2, 5, 8]
        ranges = _minibatch_ranges(self._padded(lengths), 3, mini_batch_num_tokens=9)
        assert ranges[0][0] == 0 and ranges[-1][1] == len(lengths)
        assert all(previous[1] == current[0] for previous, current in zip(ranges, ranges[1:]))

    def test_missing_token_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="neither 'cu_seq_lens_q' .* nor 'attention_mask'"):
            _minibatch_ranges({"pixel_values": torch.ones(4, 3)}, 2, mini_batch_num_tokens=8)

    def test_non_positive_budget_rejected_at_construction(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model.to("cpu")
        with pytest.raises(ValueError, match="mini_batch_num_tokens must be a positive integer or None."):
            CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=0)


def test_gradcache_token_budget_replays_dropout(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The backward hook must replay the exact (uneven) token-budget boundaries of the forward pass.
    With dropout active, any boundary drift would surface as differing re-embeddings."""
    model = stsb_bert_tiny_model.to("cpu")
    model.train()

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16)
    forward_reps: list[Tensor] = []
    backward_reps: list[Tensor] = []
    sink = forward_reps
    embed_minibatch = loss.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss.embed_minibatch = spy

    loss_value = loss([model.preprocess(VARIED_A), model.preprocess(VARIED_B)], torch.zeros(6, dtype=torch.long))
    number_of_forward_minibatches = len(forward_reps)
    sink = backward_reps
    loss_value.backward()

    assert number_of_forward_minibatches == len(backward_reps) > 2, "expected several uneven mini-batches"
    for index, (forward_rep, backward_rep) in enumerate(zip(forward_reps, backward_reps)):
        assert forward_rep.shape == backward_rep.shape, f"mini-batch {index} was re-embedded with other boundaries"
        assert torch.equal(forward_rep, backward_rep), f"mini-batch {index} was re-embedded differently"


@pytest.mark.parametrize("minibatching", [{"mini_batch_size": 2}, {"mini_batch_num_tokens": 16}])
def test_gradcache_replays_through_prompt_exclusion(
    stsb_bert_tiny_model: SentenceTransformer, minibatching: dict
) -> None:
    """``Pooling(include_prompt=False)`` zeroes prompt positions in the attention mask. It used to do so
    in place on the caller's features, through the mini-batch views. The backward pass then re-embedded
    with a *different* mask than the forward pass (silently wrong gradients), and recomputed token-budget
    boundaries no longer matched the cached gradients. The mask is now pruned on a copy, so both
    mini-batching modes must replay bitwise and match the non-cached loss."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    model[1].include_prompt = False
    labels = torch.zeros(6, dtype=torch.long)

    def features() -> list[dict[str, Tensor]]:
        columns = []
        for column in (VARIED_A, VARIED_B):
            feature = model.preprocess(column)
            feature["prompt_length"] = 2  # as Transformer.preprocess sets it when a prompt is used
            columns.append(feature)
        return columns

    loss = CachedMultipleNegativesRankingLoss(model, **minibatching)
    forward_reps: list[Tensor] = []
    backward_reps: list[Tensor] = []
    sink = forward_reps
    embed_minibatch = loss.embed_minibatch

    def spy(**kwargs):
        reps, random_state = embed_minibatch(**kwargs)
        sink.append(reps.detach().clone())
        return reps, random_state

    loss.embed_minibatch = spy

    model.zero_grad()
    batch = features()
    pristine_mask = batch[0]["attention_mask"].clone()
    loss_value = loss(batch, labels)
    assert torch.equal(pristine_mask, batch[0]["attention_mask"]), "Pooling must not mutate the features' mask"
    sink = backward_reps
    loss_value.backward()
    cached_grads = gradients(model)

    for index, (forward_rep, backward_rep) in enumerate(zip(forward_reps, backward_reps)):
        assert torch.equal(forward_rep, backward_rep), f"mini-batch {index} was re-embedded differently"

    model.zero_grad()
    plain_loss = MultipleNegativesRankingLoss(model)(features(), labels)
    plain_loss.backward()
    plain_grads = gradients(model)

    assert_trained(cached_grads)
    assert loss_value.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_gradcache_backward_releases_the_hook_state(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The backward hook partial holds each forward's cache and tokenized features. Dropping
    the loss tensor after backward must release them, or a trainer that keeps a (detached)
    reference per step would still pin every step's caches until its logging flush."""
    import gc
    import weakref

    model = stsb_bert_tiny_model.to("cpu")
    model.train()
    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3)

    features = [model.preprocess(COLUMN_A[:4]), model.preprocess(COLUMN_B[:4])]
    refs = [weakref.ref(value) for feature in features for value in feature.values() if isinstance(value, Tensor)]
    assert refs, "expected tokenized tensors to track"
    loss = loss_fn(features, torch.zeros(4, dtype=torch.long))
    loss.backward()
    detached = loss.detach()

    del features
    gc.collect()
    assert any(ref() is not None for ref in refs), "the hook must hold the features until the loss is dropped"

    del loss
    gc.collect()
    assert all(ref() is None for ref in refs), "dropping the loss tensor must release the hook state"
    assert torch.isfinite(detached)


def test_gradcache_token_budget_two_forwards_before_one_backward(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The per-forward boundaries must ride each forward pass's backward hook: two batches with
    different length distributions produce different ranges, and mixing them up would misalign the
    cached gradients."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(3, dtype=torch.long)

    first = [model.preprocess(VARIED_A[:3]), model.preprocess(VARIED_B[:3])]
    second = [model.preprocess(VARIED_A[3:]), model.preprocess(VARIED_B[3:])]

    model.zero_grad()
    shared = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16)
    (shared(first, labels) + shared(second, labels)).backward()
    shared_grads = gradients(model)

    model.zero_grad()
    first_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16)(first, labels)
    second_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16)(second, labels)
    (first_loss + second_loss).backward()
    reference_grads = gradients(model)

    assert_trained(shared_grads)
    for name, grad in shared_grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=name)


def test_create_minibatch_trims_trailing_padding() -> None:
    """A padded mini-batch of short sequences must be embedded at its own width, not the whole batch's
    padded width, or mini_batch_num_tokens would not bound the activation memory."""
    width = 20
    lengths = [20, 3, 5, 2, 4]
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    feature = {"input_ids": torch.arange(len(lengths) * width).reshape(len(lengths), width), "attention_mask": mask}

    # Rows 1..4 hold sequences of length <= 5, so the mini-batch is trimmed to width 5.
    short = _create_minibatch(feature, 1, 5)
    assert short["attention_mask"].shape[1] == 5
    assert short["input_ids"].shape[1] == 5
    torch.testing.assert_close(short["input_ids"], feature["input_ids"][1:5, :5])

    # A mini-batch that includes the longest sequence keeps the full width.
    assert _create_minibatch(feature, 0, 5)["attention_mask"].shape[1] == width


def test_create_minibatch_keeps_left_padding() -> None:
    """Leading padding is left in place: trimming it would shift the positions a model derives from the
    sequence length."""
    width = 10
    mask = torch.zeros(3, width, dtype=torch.long)
    for row, length in enumerate([4, 3, 2]):
        mask[row, width - length :] = 1  # left-padded
    feature = {"input_ids": torch.ones(3, width, dtype=torch.long), "attention_mask": mask}
    assert _create_minibatch(feature, 0, 3)["attention_mask"].shape[1] == width


def test_gradcache_token_budget_trims_but_matches_mnrl(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Trimming padded mini-batches to their own width must not change the loss or gradient: the dropped
    columns are padding for every sequence in the mini-batch, so the transformer ignores them anyway."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.zeros(len(VARIED_A), dtype=torch.long)

    cached = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16)
    cached_loss, cached_grads = _loss_and_grads(model, cached, (VARIED_A, VARIED_B), labels)
    plain = MultipleNegativesRankingLoss(model)
    plain_loss, plain_grads = _loss_and_grads(model, plain, (VARIED_A, VARIED_B), labels)

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)
