from __future__ import annotations

import math
import re
import sys

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import (
    ADRMSELoss,
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    LambdaLoss,
    ListMLELoss,
    ListNetLoss,
    MSELoss,
    PListMLELambdaWeight,
    PListMLELoss,
    RankNetLoss,
)

# Learning-to-rank losses that scatter the model's logits into an internal matrix.
# Regression test for #3793: bf16/fp16 training crashed on a float32 dtype mismatch.
LISTWISE_LOSSES = [
    PListMLELoss,
    LambdaLoss,
    ListNetLoss,
    ListMLELoss,  # subclass of PListMLELoss
    RankNetLoss,  # subclass of LambdaLoss
    ADRMSELoss,
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("loss_cls", LISTWISE_LOSSES)
def test_listwise_loss_supports_low_precision(
    reranker_bert_tiny_model_v54: CrossEncoder, loss_cls, dtype: torch.dtype
) -> None:
    model = reranker_bert_tiny_model_v54
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 CrossEncoder forward is only supported when CUDA is available.")
    if dtype == torch.bfloat16 and sys.platform == "win32" and not torch.cuda.is_available():
        pytest.skip(
            "bfloat16 CPU matmul can hard-crash (0xc000001d) on some Windows machines. Skipping to avoid CI failures."
        )

    model.model.to(dtype)
    loss_fn = loss_cls(model)

    queries = ["What is Python?", "What is PyTorch?"]
    docs_list = [
        ["A programming language.", "A snake."],
        ["A deep learning framework.", "A lunch box.", "A board game."],
    ]
    labels = [torch.tensor([1.0, 0.0], device=model.device), torch.tensor([2.0, 1.0, 0.0], device=model.device)]

    # Must not raise a dtype-mismatch RuntimeError, and the loss should be computed
    # in float32 for numerical stability of exp/log/cumsum/softmax.
    loss_value = loss_fn((queries, docs_list), labels)
    assert loss_value.dtype == torch.float32
    assert torch.isfinite(loss_value)

    # Gradients must flow back into the low-precision model without dtype errors.
    loss_value.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Expected at least one parameter to receive a gradient."
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("loss_cls", [PListMLELoss, ListMLELoss])  # ListMLELoss subclasses PListMLELoss
@pytest.mark.parametrize("respect_input_order", [True, False])
def test_listwise_loss_is_padding_invariant(reranker_bert_tiny_model_v54: CrossEncoder, loss_cls, respect_input_order):
    # A query's loss must not change because another query in the batch has more documents
    # (which pads this query's row). Batching must equal the mean of the per-query losses.
    # Labels are intentionally unsorted so the respect_input_order=False branch actually
    # permutes via sort/gather (exercising the sorted_mask = gather(mask, indices) path).
    model = reranker_bert_tiny_model_v54
    loss_fn = loss_cls(model, respect_input_order=respect_input_order)

    queries_a = ["what is the capital of france"]
    docs_a = [["berlin is the capital of germany", "paris is the capital of france"]]
    labels_a = [torch.tensor([0.0, 1.0], device=model.device)]

    queries_b = ["what is the largest planet"]
    docs_b = [["mercury is the smallest", "jupiter is the largest planet", "the moon orbits earth"]]
    labels_b = [torch.tensor([1.0, 2.0, 0.0], device=model.device)]

    loss_a = loss_fn((queries_a, docs_a), labels_a)
    loss_b = loss_fn((queries_b, docs_b), labels_b)
    loss_batched = loss_fn((queries_a + queries_b, docs_a + docs_b), labels_a + labels_b)

    # On the buggy implementation, padding the 2-doc query to width 3 inflates its loss, so the
    # batched mean diverges from the mean of the separately computed per-query losses.
    assert torch.allclose(loss_batched, (loss_a + loss_b) / 2, atol=1e-5)


class _ScoreModel(CrossEncoder):
    def __init__(self, scores):
        torch.nn.Module.__init__(self)
        self.scores = torch.nn.Parameter(scores)

    @property
    def num_labels(self):
        return 1 if self.scores.ndim == 1 else self.scores.shape[1]

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        scores = self.scores[inputs["indices"]]
        return {"scores": scores.unsqueeze(-1) if scores.ndim == 1 else scores}


@pytest.mark.parametrize("loss_cls", [LambdaLoss, RankNetLoss])
@pytest.mark.parametrize("labels", [[0.0, 0.0], [1.0, 1.0], [1.0]])
def test_ranking_loss_without_ordered_pairs_has_zero_loss_and_gradients(loss_cls, labels):
    model = _ScoreModel(torch.arange(len(labels), dtype=torch.float32))
    loss = loss_cls(model)((["query"], [[str(i) for i in range(len(labels))]]), [torch.tensor(labels)])

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(model.scores.grad, torch.zeros_like(model.scores))


@pytest.mark.parametrize("loss_cls", [LambdaLoss, RankNetLoss])
def test_ranking_loss_averages_ordered_pairs_in_mixed_batch(loss_cls):
    model = _ScoreModel(torch.tensor([1.0, 0.0, -1.0, 0.0, 2.0, 3.0]))
    loss_fn = loss_cls(model)
    actual = loss_fn(
        (["ranked", "tied", "single"], [["0", "1", "2"], ["3", "4"], ["5"]]),
        [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0])],
    )

    scores = model.scores.detach().double().requires_grad_()
    pair_losses = torch.nn.functional.softplus(scores[1:3] - scores[0]) / math.log(2)
    if loss_cls is LambdaLoss:
        inverse_discount_at_2 = 1 / math.log2(3)
        weights = scores.new_tensor([11 * (1 - inverse_discount_at_2), 10 * (inverse_discount_at_2 - 0.5) + 0.5])
        pair_losses = pair_losses * weights
    expected = pair_losses.mean()

    torch.testing.assert_close(actual, expected.float())
    actual.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float())


@pytest.mark.parametrize("loss_cls", [LambdaLoss, RankNetLoss])
def test_ranking_loss_top_k_learns_from_relevant_document_below_cutoff(loss_cls):
    model = _ScoreModel(torch.tensor([2.0, 1.0, 0.0]))
    loss = loss_cls(model, k=1)((["query"], [["0", "1", "2"]]), [torch.tensor([0.0, 0.0, 1.0])])

    scores = model.scores.detach().double().requires_grad_()
    expected = torch.nn.functional.softplus(scores[0] - scores[2]) / math.log(2)
    if loss_cls is LambdaLoss:
        expected = expected * (10 * (1 / math.log2(3) - 0.5) + 0.5)

    torch.testing.assert_close(loss, expected.float())
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float())


def test_ranknet_top_k_keeps_pairs_with_a_document_in_top_k():
    model = _ScoreModel(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    loss = RankNetLoss(model, k=2)((["query"], [["0", "1", "2", "3"]]), [torch.tensor([0.0, 1.0, 0.0, 2.0])])

    scores = model.scores.detach().double().requires_grad_()
    pairs = [(1, 0), (1, 2), (3, 0), (3, 1)]
    pair_losses = torch.stack([torch.nn.functional.softplus(scores[less] - scores[more]) for more, less in pairs])
    expected = pair_losses.mean() / math.log(2)

    torch.testing.assert_close(loss, expected.float())
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float())


def test_plistmle_default_weights_for_short_lists():
    mask = torch.arange(4).unsqueeze(0) < torch.arange(1, 5).unsqueeze(1)
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0], [7.0, 3.0, 1.0, 0.0]])
    torch.testing.assert_close(PListMLELambdaWeight()(mask), expected)


@pytest.mark.parametrize("num_docs", [3, 127, 128, 129, 200, 1000])
@pytest.mark.parametrize("respect_input_order", [True, False])
def test_plistmle_long_lists_match_sequential_cross_entropy(num_docs, respect_input_order):
    model = _ScoreModel(torch.cat([torch.linspace(-1.0, 1.0, num_docs), torch.tensor([1.0, -2.0, 3.0])]))
    loss_fn = PListMLELoss(model, respect_input_order=respect_input_order)
    documents = [[str(i) for i in range(num_docs)], [str(i) for i in range(num_docs, num_docs + 3)]]
    labels = [torch.arange(num_docs), torch.tensor([1.0, 2.0, 0.0])]
    actual = loss_fn((["long query", "short query"], documents), labels)

    # Independent float64 reference: suffix cross-entropies and exact Python integer weights.
    scores = model.scores.detach().double().requires_grad_()
    orders = [list(range(num_docs)), list(range(num_docs, num_docs + 3))]
    if not respect_input_order:
        orders = [orders[0][::-1], [num_docs + 1, num_docs, num_docs + 2]]
    query_losses = []
    for order in orders:
        weights = [2 ** (len(order) - rank) - 1 for rank in range(1, len(order) + 1)]
        total_weight = sum(weights)
        normalized_weights = scores.new_tensor([weight / total_weight for weight in weights])
        terms = torch.stack(
            [
                torch.nn.functional.cross_entropy(scores[order[index:]].unsqueeze(0), torch.zeros(1, dtype=torch.long))
                for index in range(len(order))
            ]
        )
        query_losses.append((terms * normalized_weights).sum())
    expected = torch.stack(query_losses).mean()

    torch.testing.assert_close(actual, expected.float(), rtol=2e-4, atol=2e-4)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("loss_cls", [ListMLELoss, PListMLELoss])
@pytest.mark.parametrize("respect_input_order", [True, False])
@pytest.mark.parametrize("offset", [0.0, 1000.0, -1000.0])
def test_listmle_large_scores_match_sequential_cross_entropy(loss_cls, respect_input_order, offset):
    model = _ScoreModel(torch.tensor([1.0, 2.0, -1.0, 3.0, 0.0]) + offset)
    loss_fn = loss_cls(model, respect_input_order=respect_input_order)
    loss = loss_fn(
        (["query a", "query b"], [["0", "1"], ["2", "3", "4"]]),
        [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0, 0.0])],
    )

    # Compare against independent float64 cross-entropies for each suffix.
    scores = model.scores.detach().double().requires_grad_()
    orders = [[0, 1], [2, 3, 4]] if respect_input_order else [[1, 0], [3, 2, 4]]
    query_losses = []
    for order, weights in zip(orders, [[1.0, 0.0], [3 / 4, 1 / 4, 0.0]]):
        terms = torch.stack(
            [
                torch.nn.functional.cross_entropy(scores[order[index:]].unsqueeze(0), torch.zeros(1, dtype=torch.long))
                for index in range(len(order))
            ]
        )
        if loss_cls is PListMLELoss:
            terms = terms * torch.tensor(weights, dtype=terms.dtype)
        query_losses.append(terms.sum())
    expected = torch.stack(query_losses).mean()

    torch.testing.assert_close(loss, expected.float(), rtol=2e-4, atol=2e-4)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("loss_cls", [RankNetLoss, LambdaLoss])
@pytest.mark.parametrize("gap", [0.5, 20.0, 1000.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0], ids=["correct", "incorrect"])
@pytest.mark.parametrize("sigma", [0.5, 2.0])
@pytest.mark.parametrize("reduction_log", ["natural", "binary"])
def test_ranking_loss_matches_pairwise_cross_entropy(loss_cls, gap, direction, sigma, reduction_log):
    model = _ScoreModel(torch.tensor([0.0, direction * gap]))
    loss = loss_cls(model, sigma=sigma, reduction_log=reduction_log)(
        (["query"], [["0", "1"]]), [torch.tensor([1.0, 0.0])]
    )

    scores = model.scores.detach().double().requires_grad_()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        sigma * (scores[0] - scores[1]), torch.tensor(1.0).double()
    )
    if loss_cls is LambdaLoss:
        expected = expected * (11 * (1 - 1 / math.log2(3)))
    if reduction_log == "binary":
        expected = expected / math.log(2)

    torch.testing.assert_close(loss, expected.float(), rtol=1e-5, atol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("reduction_log", ["natural", "binary"])
def test_ranknet_extreme_scores_with_padding_match_valid_pair_cross_entropies(reduction_log):
    model = _ScoreModel(torch.tensor([0.0, -3.0, 1000.0, 0.0, 100.0]))
    loss = RankNetLoss(model, mini_batch_size=2, reduction_log=reduction_log)(
        (["query a", "query b"], [["0", "1", "2"], ["3", "4"]]),
        [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([1.0, 0.0])],
    )

    scores = model.scores.detach().double().requires_grad_()
    pair_logits = torch.stack([scores[a] - scores[b] for a, b in [(0, 1), (0, 2), (1, 2), (3, 4)]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, torch.ones_like(pair_logits))
    if reduction_log == "binary":
        expected = expected / math.log(2)

    torch.testing.assert_close(loss, expected.float(), rtol=1e-5, atol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, scores.grad.float(), rtol=1e-5, atol=1e-6)


POINTWISE_INPUTS = [
    ["what is the capital of France?", "how tall is mount everest?", "who wrote hamlet?"],
    ["Paris is the capital of France.", "Bananas are yellow.", "Hamlet was written by Shakespeare."],
]
POINTWISE_LABELS = torch.tensor([1.0, 0.0, 1.0])


@pytest.mark.parametrize("loss_class", [MSELoss, BinaryCrossEntropyLoss])
def test_column_labels_match_flat_labels(reranker_bert_tiny_model_v54: CrossEncoder, loss_class) -> None:
    model = reranker_bert_tiny_model_v54
    model.eval()  # no dropout, so both forward passes see the same logits
    loss_fn = loss_class(model)
    labels = POINTWISE_LABELS.to(model.device)

    expected = loss_fn(POINTWISE_INPUTS, labels)
    actual = loss_fn(POINTWISE_INPUTS, labels[:, None])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_class", [MSELoss, BinaryCrossEntropyLoss])
@pytest.mark.parametrize("batch_size", [1, len(POINTWISE_LABELS)])
@pytest.mark.parametrize("labels_per_row", [2, 3])
def test_multiple_labels_per_row_raise(
    reranker_bert_tiny_model_v54: CrossEncoder, loss_class, batch_size, labels_per_row
) -> None:
    loss_fn = loss_class(reranker_bert_tiny_model_v54)
    labels = POINTWISE_LABELS[:batch_size, None].repeat(1, labels_per_row).to(reranker_bert_tiny_model_v54.device)
    inputs = [column[:batch_size] for column in POINTWISE_INPUTS]
    message = (
        f"{loss_class.__name__} expects {batch_size} labels (one per example), "
        f"but got {labels.numel()} labels with shape {tuple(labels.shape)}."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        loss_fn(inputs, labels)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("target_type", ["class", "column", "probabilities"])
def test_cross_entropy_preserves_class_and_probability_targets(batch_size, target_type):
    model = _ScoreModel(torch.tensor([[1.0, 2.0, -1.0], [0.0, 1.0, 2.0], [2.0, -1.0, 0.0]])[:batch_size])
    inputs = (["query"] * batch_size, [str(i) for i in range(batch_size)])
    labels = torch.tensor([0, 1, 2])[:batch_size]
    if target_type == "probabilities":
        labels = torch.nn.functional.one_hot(labels, num_classes=3).float() * 0.7 + 0.1
    expected = torch.nn.functional.cross_entropy(model.scores, labels)
    actual_labels = labels[:, None] if target_type == "column" else labels
    actual = CrossEntropyLoss(model)(inputs, actual_labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, model.scores)[0], torch.autograd.grad(expected, model.scores)[0]
    )
