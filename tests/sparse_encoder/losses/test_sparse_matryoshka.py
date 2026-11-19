from __future__ import annotations

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
        splade_bert_tiny_model, inner, max_active_dims=[None, 20, 10], query_max_active_dims=[None, 5, 2]
    )
    embeddings = _random_embeddings(n_columns=3)

    loss.compute_loss_from_embeddings(embeddings, None)

    assert len(inner.calls) == 3
    for call, query_budget, document_budget in zip(inner.calls, [None, 5, 2], [None, 20, 10]):
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


def test_loss_is_weighted_sum_over_budgets(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model
    inner = SparseMultipleNegativesRankingLoss(model)
    loss = SparseMatryoshkaLoss(
        model, inner, max_active_dims=[None, 10], query_max_active_dims=[None, 3], matryoshka_weights=[1.0, 0.5]
    )
    embeddings = _random_embeddings()

    pruned = [select_max_active_dims(embeddings[0], 3)] + [select_max_active_dims(e, 10) for e in embeddings[1:]]
    expected = inner.compute_loss_from_embeddings(embeddings, None) + 0.5 * inner.compute_loss_from_embeddings(
        pruned, None
    )
    assert loss.compute_loss_from_embeddings(embeddings, None).item() == pytest.approx(expected.item())


def test_gradient_only_reaches_kept_dimensions(splade_bert_tiny_model: SparseEncoder) -> None:
    loss = SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), max_active_dims=[5])
    embeddings = [embedding.requires_grad_() for embedding in _random_embeddings(n_columns=2)]

    loss.compute_loss_from_embeddings(embeddings, None).backward()

    for embedding in embeddings:
        kept = select_max_active_dims(embedding.detach(), 5) != 0
        assert (embedding.grad[kept] == 1).all()
        assert (embedding.grad[~kept] == 0).all()


def test_n_dims_per_step_samples_budgets(splade_bert_tiny_model: SparseEncoder) -> None:
    inner = RecordingLoss()
    loss = SparseMatryoshkaLoss(splade_bert_tiny_model, inner, max_active_dims=[None, 40, 20, 10], n_dims_per_step=2)

    loss.compute_loss_from_embeddings(_random_embeddings(), None)

    assert len(inner.calls) == 2


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
    """Wrapped in SpladeLoss, only the base loss is pruned; FLOPS still regularizes the full vectors."""
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


def test_cached_splade_matches_splade(splade_bert_tiny_model: SparseEncoder) -> None:
    """Pruning happens on the gathered embeddings, so GradCache must reproduce SpladeLoss's values and gradients."""
    model = splade_bert_tiny_model.to("cpu")
    _disable_dropout(model)
    model.train()

    def make_loss(cached: bool) -> nn.Module:
        inner = SparseMultipleNegativesRankingLoss(model)
        matryoshka = SparseMatryoshkaLoss(
            model,
            inner,
            max_active_dims=[None, 16, 8],
            query_max_active_dims=[None, 8, 4],
            matryoshka_weights=[1.0, 0.5, 0.25],
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


def test_csr_loss_prunes_only_the_main_loss(csr_bert_tiny_model: SparseEncoder) -> None:
    model = csr_bert_tiny_model.to("cpu")
    model.train()
    inner = RecordingLoss()
    loss = CSRLoss(model, loss=SparseMatryoshkaLoss(model, inner, max_active_dims=[4], query_max_active_dims=[2]))

    output = loss([model.preprocess(column) for column in (ANCHORS, POSITIVES)], None)
    torch.stack(list(output.values())).sum().backward()

    (call,) = inner.calls
    assert (_active_dims(call[0]) <= 2).all()
    assert (_active_dims(call[1]) <= 4).all()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())


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


def test_budgets_are_sorted_from_largest_to_smallest(splade_bert_tiny_model: SparseEncoder) -> None:
    loss = SparseMatryoshkaLoss(
        splade_bert_tiny_model,
        RecordingLoss(),
        max_active_dims=[10, None, 20],
        query_max_active_dims=[2, None, 5],
        matryoshka_weights=[0.25, 1.0, 0.5],
    )

    assert loss.max_active_dims == (None, 20, 10)
    assert loss.query_max_active_dims == (None, 5, 2)
    assert loss.matryoshka_weights == (1.0, 0.5, 0.25)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_active_dims": [64, 32]},
        {"max_active_dims": [None, 64], "query_max_active_dims": [16, 8]},
        {"max_active_dims": [64, None], "query_max_active_dims": [None, 8]},
    ],
)
def test_missing_unpruned_budget_warns(splade_bert_tiny_model: SparseEncoder, caplog, kwargs: dict) -> None:
    with caplog.at_level("WARNING"):
        SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), **kwargs)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "It is strongly recommended to include None" in caplog.records[0].message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_active_dims": [None, 64]},
        {"max_active_dims": [30522, 64]},
        {"max_active_dims": [64, None], "query_max_active_dims": [8, None]},
    ],
)
def test_unpruned_budget_does_not_warn(splade_bert_tiny_model: SparseEncoder, caplog, kwargs: dict) -> None:
    with caplog.at_level("WARNING"):
        SparseMatryoshkaLoss(splade_bert_tiny_model, RecordingLoss(), **kwargs)

    assert not caplog.records
