from __future__ import annotations

from itertools import combinations

import pytest
import torch
from torch import Tensor, nn

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    CSRLoss,
    SparseMatryoshkaLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.util import select_max_active_dims
from tests.sentence_transformer.losses.utils import disable_dropout as _disable_dropout

ANCHORS = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f"]
POSITIVES = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f"]
NEGATIVES = ["negative a", "negative b", "negative c", "negative d", "negative e", "negative f"]


class RecordingLoss(nn.Module):
    """Inner loss that records the embeddings it receives and returns their plain sum."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[Tensor]] = []

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor | None = None) -> Tensor:
        self.calls.append([embedding.detach().clone() for embedding in embeddings])
        return torch.stack([embedding.sum() for embedding in embeddings]).sum()


def _random_embeddings(n_columns: int = 3, batch_size: int = 4, dim: int = 50) -> list[Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.relu(torch.randn(batch_size, dim, generator=generator)) for _ in range(n_columns)]


def _active_dims(embeddings: Tensor) -> Tensor:
    return (embeddings != 0).sum(dim=-1)


def test_unpruned_budget_matches_inner_loss(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model
    inner = SparseMultipleNegativesRankingLoss(model)
    loss = SparseMatryoshkaLoss(model, inner, max_active_dims=[None])
    embeddings = _random_embeddings()

    expected = inner.compute_loss_from_embeddings(embeddings, None)
    assert loss.compute_loss_from_embeddings(embeddings, None).item() == pytest.approx(expected.item())


def test_prunes_queries_and_documents_to_their_own_budgets(splade_bert_tiny_model: SparseEncoder) -> None:
    inner = RecordingLoss()
    loss = SparseMatryoshkaLoss(
        splade_bert_tiny_model, inner, max_active_dims=[10, None, 20], query_max_active_dims=[2, None, 5]
    )
    embeddings = _random_embeddings(n_columns=3)

    loss.compute_loss_from_embeddings(embeddings, None)

    assert len(inner.calls) == 3
    for call, query_budget, document_budget in zip(inner.calls, [2, None, 5], [10, None, 20]):
        torch.testing.assert_close(call[0], select_max_active_dims(embeddings[0], query_budget))
        for received, original in zip(call[1:], embeddings[1:]):
            torch.testing.assert_close(received, select_max_active_dims(original, document_budget))
        if query_budget is not None:
            assert (_active_dims(call[0]) <= query_budget).all()
            assert (_active_dims(torch.cat(call[1:])) <= document_budget).all()


def test_query_budgets_default_to_document_budgets(splade_bert_tiny_model: SparseEncoder) -> None:
    inner = RecordingLoss()
    loss = SparseMatryoshkaLoss(splade_bert_tiny_model, inner, max_active_dims=[7])
    embeddings = _random_embeddings(n_columns=2)

    loss.compute_loss_from_embeddings(embeddings, None)

    (call,) = inner.calls
    for received, original in zip(call, embeddings):
        torch.testing.assert_close(received, select_max_active_dims(original, 7))


@pytest.mark.parametrize("weights", [None, [1.0, 0.5]])
def test_loss_is_weighted_sum_over_budgets(splade_bert_tiny_model: SparseEncoder, weights: list[float] | None) -> None:
    model = splade_bert_tiny_model
    inner = SparseMultipleNegativesRankingLoss(model)
    loss = SparseMatryoshkaLoss(
        model, inner, max_active_dims=[None, 10], query_max_active_dims=[None, 3], matryoshka_weights=weights
    )
    embeddings = _random_embeddings()

    pruned = [select_max_active_dims(embeddings[0], 3)] + [select_max_active_dims(e, 10) for e in embeddings[1:]]
    weights = weights if weights is not None else [0.5, 0.5]
    expected = weights[0] * inner.compute_loss_from_embeddings(embeddings, None)
    expected += weights[1] * inner.compute_loss_from_embeddings(pruned, None)
    assert loss.compute_loss_from_embeddings(embeddings, None).item() == pytest.approx(expected.item())


def test_gradient_only_reaches_kept_dimensions(splade_bert_tiny_model: SparseEncoder) -> None:
    loss = SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), max_active_dims=[5])
    embeddings = [embedding.requires_grad_() for embedding in _random_embeddings(n_columns=2)]

    loss.compute_loss_from_embeddings(embeddings, None).backward()

    for embedding in embeddings:
        kept = select_max_active_dims(embedding.detach(), 5) != 0
        assert (embedding.grad[kept] == 1).all()
        assert (embedding.grad[~kept] == 0).all()


@pytest.mark.parametrize("weights", [None, [1.0, 0.5, 0.25]])
@pytest.mark.parametrize("sample_size", [1, 2])
def test_sampling_preserves_expected_loss_and_gradients(
    splade_bert_tiny_model: SparseEncoder, monkeypatch, weights: list[float] | None, sample_size: int
) -> None:
    inner = RecordingLoss()
    loss = SparseMatryoshkaLoss(
        splade_bert_tiny_model,
        inner,
        max_active_dims=[None, 20, 10],
        query_max_active_dims=[None, 5, 2],
        matryoshka_weights=weights,
    )
    embeddings = [embedding.requires_grad_() for embedding in _random_embeddings()]
    full_loss = loss.compute_loss_from_embeddings(embeddings)
    inner.calls.clear()
    loss.n_dims_per_step = sample_size

    sampled_losses = []
    for indices in combinations(range(3), sample_size):
        monkeypatch.setattr(
            "sentence_transformers.sparse_encoder.losses.sparse_matryoshka.random.sample",
            lambda population, k: indices,
        )
        sampled_losses.append(loss.compute_loss_from_embeddings(embeddings))

    mean_loss = torch.stack(sampled_losses).mean()

    assert len(inner.calls) == len(sampled_losses) * sample_size
    torch.testing.assert_close(mean_loss, full_loss)
    torch.testing.assert_close(torch.autograd.grad(mean_loss, embeddings), torch.autograd.grad(full_loss, embeddings))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_active_dims": []}, "at least one"),
        ({"max_active_dims": [10, 0]}, "positive"),
        ({"max_active_dims": [10], "query_max_active_dims": [-1]}, "positive"),
        ({"max_active_dims": [10, 5], "query_max_active_dims": [3]}, "same length"),
        ({"max_active_dims": [10, 5], "matryoshka_weights": [1.0]}, "same length"),
    ],
)
def test_rejects_invalid_budgets(splade_bert_tiny_model: SparseEncoder, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), **kwargs)


def test_rejects_loss_without_compute_loss_from_embeddings(splade_bert_tiny_model: SparseEncoder) -> None:
    with pytest.raises(ValueError, match="compute_loss_from_embeddings"):
        SparseMatryoshkaLoss(splade_bert_tiny_model, nn.Identity(), max_active_dims=[10])


def test_dict_valued_inner_loss_is_weighted_per_key(splade_bert_tiny_model: SparseEncoder) -> None:
    class DictLoss(nn.Module):
        def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor | None = None) -> dict:
            return {"first": embeddings[0].sum(), "rest": torch.cat(embeddings[1:]).sum()}

    loss = SparseMatryoshkaLoss(
        splade_bert_tiny_model,
        DictLoss(),
        max_active_dims=[None, 10],
        query_max_active_dims=[None, 3],
        matryoshka_weights=[1, 2],
    )
    embeddings = _random_embeddings()

    output = loss.compute_loss_from_embeddings(embeddings, None)

    assert output.keys() == {"first", "rest"}
    expected_first = embeddings[0].sum() + 2 * select_max_active_dims(embeddings[0], 3).sum()
    expected_rest = sum(e.sum() + 2 * select_max_active_dims(e, 10).sum() for e in embeddings[1:])
    assert output["first"].item() == pytest.approx(expected_first.item())
    assert output["rest"].item() == pytest.approx(expected_rest.item())


def _splade_losses(model: SparseEncoder, inner: nn.Module, cached: bool) -> nn.Module:
    kwargs = {"model": model, "loss": inner, "document_regularizer_weight": 3e-5, "query_regularizer_weight": 5e-5}
    if cached:
        return CachedSpladeLoss(**kwargs, mini_batch_size=2)
    return SpladeLoss(**kwargs)


def _run(model: SparseEncoder, loss_fn: nn.Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    model.zero_grad()
    features = [model.preprocess(column) for column in (ANCHORS, POSITIVES, NEGATIVES)]
    output = loss_fn(features, None)
    torch.stack(list(output.values())).sum().backward()
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    return output, grads


def test_regularizers_see_unpruned_embeddings(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()
    inner = SparseMultipleNegativesRankingLoss(model)
    matryoshka = SparseMatryoshkaLoss(model, inner, max_active_dims=[8], query_max_active_dims=[4])

    wrapped, _ = _run(model, _splade_losses(model, matryoshka, cached=False))
    plain, _ = _run(model, _splade_losses(model, inner, cached=False))

    for key in ("document_regularizer_loss", "query_regularizer_loss"):
        assert wrapped[key].item() == pytest.approx(plain[key].item(), rel=1e-5), key
    assert wrapped["base_loss"].item() != pytest.approx(plain["base_loss"].item())


@pytest.mark.parametrize("n_dims_per_step", [-1, 1])
def test_default_weights_preserve_regularization_balance(
    splade_bert_tiny_model: SparseEncoder, n_dims_per_step: int
) -> None:
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()
    inner = SparseMultipleNegativesRankingLoss(model)
    matryoshka = SparseMatryoshkaLoss(
        model, inner, max_active_dims=[None, model.get_embedding_dimension()], n_dims_per_step=n_dims_per_step
    )

    wrapped, wrapped_grads = _run(model, _splade_losses(model, matryoshka, cached=False))
    plain, plain_grads = _run(model, _splade_losses(model, inner, cached=False))

    torch.testing.assert_close(wrapped, plain)
    torch.testing.assert_close(wrapped_grads, plain_grads, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("n_dims_per_step", [-1, 1])
def test_cached_splade_matches_splade(
    splade_bert_tiny_model: SparseEncoder, monkeypatch, n_dims_per_step: int
) -> None:
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()
    monkeypatch.setattr(
        "sentence_transformers.sparse_encoder.losses.sparse_matryoshka.random.sample",
        lambda population, k: list(population)[-k:],
    )

    def make_loss(cached: bool) -> nn.Module:
        inner = SparseMultipleNegativesRankingLoss(model)
        matryoshka = SparseMatryoshkaLoss(
            model,
            inner,
            max_active_dims=[None, 16, 8],
            query_max_active_dims=[None, 8, 4],
            matryoshka_weights=[1.0, 0.5, 0.25],
            n_dims_per_step=n_dims_per_step,
        )
        return _splade_losses(model, matryoshka, cached=cached)

    cached_output, cached_grads = _run(model, make_loss(cached=True))
    plain_output, plain_grads = _run(model, make_loss(cached=False))

    assert cached_output.keys() == plain_output.keys()
    for key in plain_output:
        assert cached_output[key].item() == pytest.approx(plain_output[key].item(), rel=1e-4, abs=1e-5), key
    assert cached_grads and sum(grad.abs().sum() for grad in cached_grads.values()) > 0
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=2e-4, msg=name)


def test_csr_budgets_match_inference(csr_bert_tiny_model: SparseEncoder) -> None:
    model = csr_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()
    inner = RecordingLoss()
    budgets = [None, model[-1].k, 4]
    query_budgets = [None, model[-1].k, 2]
    loss = CSRLoss(
        model,
        loss=SparseMatryoshkaLoss(model, inner, max_active_dims=budgets, query_max_active_dims=query_budgets),
    )

    output = loss([model.preprocess(column) for column in (ANCHORS, POSITIVES)], None)
    torch.stack(list(output.values())).sum().backward()

    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())
    assert len(inner.calls) == len(budgets)
    for call, query_budget, document_budget in zip(inner.calls, query_budgets, budgets):
        for received, texts, budget in zip(call, (ANCHORS, POSITIVES), (query_budget, document_budget)):
            expected = model.encode(
                texts, max_active_dims=budget or model[-1].k, convert_to_sparse_tensor=False, show_progress_bar=False
            )
            torch.testing.assert_close(received, expected)


@pytest.mark.parametrize("budget_name", ["max_active_dims", "query_max_active_dims"])
def test_csr_rejects_budgets_above_native_k(csr_bert_tiny_model: SparseEncoder, budget_name: str) -> None:
    model = csr_bert_tiny_model
    kwargs = {"max_active_dims": [None, model[-1].k], budget_name: [None, model[-1].k * 2]}

    with pytest.raises(ValueError, match=f"already prunes embeddings to k={model[-1].k}"):
        SparseMatryoshkaLoss(model, RecordingLoss(), **kwargs)


def test_forward_without_wrapper(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()
    loss = SparseMatryoshkaLoss(model, SparseMultipleNegativesRankingLoss(model), max_active_dims=[None, 8])

    features = [model.preprocess(column) for column in (ANCHORS, POSITIVES)]
    output = loss(features, None)
    output.backward()

    assert output.requires_grad
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())


def test_get_config_dict(splade_bert_tiny_model: SparseEncoder) -> None:
    loss = SparseMatryoshkaLoss(
        splade_bert_tiny_model,
        SparseMultipleNegativesRankingLoss(splade_bert_tiny_model),
        max_active_dims=[None, 10],
        query_max_active_dims=[None, 4],
        matryoshka_weights=[1.0, 0.5],
        n_dims_per_step=1,
    )

    assert loss.get_config_dict() == {
        "loss": "SparseMultipleNegativesRankingLoss",
        "max_active_dims": (None, 10),
        "query_max_active_dims": (None, 4),
        "matryoshka_weights": (1.0, 0.5),
        "n_dims_per_step": 1,
    }


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_active_dims": [None, 64, 40000]}, "max_active_dims"),
        ({"max_active_dims": [None, 64], "query_max_active_dims": [None, 40000]}, "query_max_active_dims"),
    ],
)
def test_budget_above_embedding_dimension_raises(
    splade_bert_tiny_model: SparseEncoder, kwargs: dict, name: str
) -> None:
    expected = f"Dimensions in {name} cannot exceed the model's embedding dimension of 30522."
    with pytest.raises(ValueError, match=expected):
        SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), **kwargs)
