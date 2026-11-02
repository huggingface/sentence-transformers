from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.util import batch_to_device

DIM = 8


class _PassthroughModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        self.forward_calls += 1
        return features

    def _get_untruncated_embedding_dimension(self) -> int:
        return DIM


def _build_features(batch: int = 4) -> list[dict[str, Tensor]]:
    return [
        {
            "token_embeddings": F.normalize(
                torch.randn(batch, tokens, DIM, generator=torch.Generator().manual_seed(seed)), dim=-1
            ).requires_grad_(),
            "attention_mask": torch.ones(batch, tokens, dtype=torch.bool),
        }
        for seed, tokens in enumerate((6, 10, 14), start=1)
    ]


def _leaves(features: list[dict[str, Tensor]]) -> list[Tensor]:
    return [sf["token_embeddings"] for sf in features]


def _mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model, **kwargs)


def _cached_mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=model, mini_batch_size=2, show_progress_bar=False, **kwargs
    )


class _NormalizeModel(_PassthroughModel):
    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        return {
            "token_embeddings": F.normalize(features["token_embeddings"], p=2, dim=-1),
            "attention_mask": features["attention_mask"],
        }


def _shrink_reference(tensor: Tensor, dim: int) -> Tensor:
    return F.normalize(tensor[..., :dim], p=2, dim=-1)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_full_dim_only_matches_wrapped_loss(build_inner) -> None:
    plain_features = _build_features()
    plain_value = build_inner(_NormalizeModel())(plain_features, None)
    plain_value.backward()

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _NormalizeModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, build_inner(model), matryoshka_dims=[DIM])
    value = loss(wrapped_features, None)
    value.backward()

    assert torch.allclose(value, plain_value, atol=1e-6)
    for leaf, plain_sf in zip(leaves, plain_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, plain_sf["token_embeddings"].grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("use_generator", [False, True])
def test_matryoshka_equals_manual_sum_of_truncated_losses(build_inner, use_generator: bool) -> None:
    weights = [1.0, 0.5]

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(
        model, build_inner(model), matryoshka_dims=[DIM, DIM // 2], matryoshka_weights=weights
    )
    value = loss(iter(wrapped_features) if use_generator else wrapped_features, None)
    value.backward()
    if build_inner is _mnrl:
        assert model.forward_calls == len(wrapped_features)

    manual_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    manual_value = 0.0
    for weight, dim in zip(weights, [DIM, DIM // 2]):
        truncated = [
            {
                "token_embeddings": _shrink_reference(sf["token_embeddings"], dim),
                "attention_mask": sf["attention_mask"],
            }
            for sf in manual_features
        ]
        manual_value = manual_value + weight * plain(truncated, None)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_matryoshka_does_not_truncate_kd_labels() -> None:
    labels = torch.tensor([[5.0, 1.0], [4.0, 0.5], [3.0, 0.2], [6.0, 2.0]])

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, kd, matryoshka_dims=[DIM, DIM // 2])
    value = loss(wrapped_features, labels)
    value.backward()

    manual_features = _build_features()
    plain = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    manual_value = 0.0
    for dim in [DIM, DIM // 2]:
        truncated = [
            {
                "token_embeddings": _shrink_reference(sf["token_embeddings"], dim),
                "attention_mask": sf["attention_mask"],
            }
            for sf in manual_features
        ]
        manual_value = manual_value + plain(truncated, labels)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_empty_dims_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="at least one dimension"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[])


def test_non_positive_dim_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="must be > 0"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[8, 0])


def test_weights_length_mismatch_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="same length"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[8, 4], matryoshka_weights=[1.0])


def test_dim_exceeding_model_dim_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="cannot exceed the model's embedding dimension of 8"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[16, 8])


def test_missing_full_dim_warns(caplog) -> None:
    model = _PassthroughModel()
    with caplog.at_level("WARNING"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[4, 2])
    assert "is not included in matryoshka_dims" in caplog.text


def test_dims_and_weights_sorted_descending() -> None:
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(
        model, _mnrl(model), matryoshka_dims=[2, 8, 4], matryoshka_weights=[0.25, 1.0, 0.5]
    )
    assert loss.matryoshka_dims == (8, 4, 2)
    assert loss.matryoshka_weights == (1.0, 0.5, 0.25)
    assert loss.get_config_dict() == {
        "loss": "MultiVectorMultipleNegativesRankingLoss",
        "matryoshka_dims": (8, 4, 2),
        "matryoshka_weights": (1.0, 0.5, 0.25),
        "n_dims_per_step": -1,
    }


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("truncate_dim", [None, 32])
def test_matryoshka_on_a_real_model(mve_bert_tiny_model, build_inner, truncate_dim: int | None) -> None:
    model = mve_bert_tiny_model.train()
    dim = model.get_embedding_dimension()
    model.truncate_dim = truncate_dim
    features = [
        batch_to_device(model.preprocess(texts, task=task), model.device)
        for texts, task in [
            (["what is late interaction", "who wrote hamlet"], "query"),
            (
                ["late interaction scores query and document tokens directly", "hamlet was written by shakespeare"],
                "document",
            ),
        ]
    ]
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, build_inner(model), matryoshka_dims=[dim, dim // 2])
    assert model.truncate_dim == truncate_dim
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    grads = [parameter.grad for parameter in model[0].parameters() if parameter.grad is not None]
    assert grads and any(grad.abs().sum() > 0 for grad in grads)
    assert model.forward.__self__ is model


def test_validation_uses_untruncated_dimension(mve_bert_tiny_model, caplog) -> None:
    model = mve_bert_tiny_model
    dim = model.get_embedding_dimension()
    model.truncate_dim = dim // 2
    with pytest.raises(ValueError, match=f"cannot exceed the model's embedding dimension of {dim}"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[dim + 1, dim // 2])
    with caplog.at_level("WARNING"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[dim // 2])
    assert f"dimension {dim} is not included" in caplog.text
    assert model.truncate_dim == dim // 2
