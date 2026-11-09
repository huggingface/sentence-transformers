from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.multi_vector_encoder.modules import (
    BaseTokenPooling,
    HierarchicalTokenPooling,
)
from sentence_transformers.util import batch_to_device
from tests.utils import skip_fp16_cpu_layer_norm

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


def _pool_reference(features: list[dict[str, Tensor]], pooling: BaseTokenPooling | None) -> list[dict[str, Tensor]]:
    if pooling is None:
        return [dict(sf) for sf in features]
    pooled = []
    for idx, sf in enumerate(features):
        task = "query" if idx == 0 else "document"
        pooled.append(
            pooling({"token_embeddings": sf["token_embeddings"], "attention_mask": sf["attention_mask"]}, task=task)
        )
    return pooled


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("use_none", [False, True])
def test_passthrough_only_matches_wrapped_loss(build_inner, use_none: bool) -> None:
    plain_features = _build_features()
    plain_value = build_inner(_PassthroughModel())(plain_features, None)
    plain_value.backward()

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    pooling = None if use_none else HierarchicalTokenPooling(pool_factor=1, normalize_embeddings=False)
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, build_inner(model), token_poolings=[pooling])
    value = loss(wrapped_features, None)
    value.backward()

    assert torch.allclose(value, plain_value, atol=1e-6)
    for leaf, plain_sf in zip(leaves, plain_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, plain_sf["token_embeddings"].grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize(
    "token_poolings,weights,use_generator",
    [
        pytest.param([None, HierarchicalTokenPooling(pool_factor=2)], [1.0, 0.5], False, id="factor"),
        pytest.param([HierarchicalTokenPooling(pool_factor=2), None], [0.5, 1.0], False, id="pooled-first"),
        pytest.param([None, HierarchicalTokenPooling(pool_factor=2)], [1.0, 0.5], True, id="iterator"),
        pytest.param([HierarchicalTokenPooling(pool_factor=2)], [1.0], False, id="factor-only"),
        pytest.param([HierarchicalTokenPooling(max_tokens=4)], [1.0], False, id="cap-only"),
        pytest.param([HierarchicalTokenPooling(distance_threshold=1.1)], [1.0], False, id="adaptive-only"),
        pytest.param(
            [
                None,
                HierarchicalTokenPooling(pool_factor=2),
                HierarchicalTokenPooling(max_tokens=4),
                HierarchicalTokenPooling(distance_threshold=1.1),
            ],
            [0.5, 1 / 6, 1 / 6, 1 / 6],
            False,
            id="mixed",
        ),
    ],
)
def test_token_pooling_equals_manual_sum_of_pooled_losses(build_inner, token_poolings, weights, use_generator) -> None:
    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, build_inner(model), token_poolings=token_poolings, weights=weights
    )
    value = loss(iter(wrapped_features) if use_generator else wrapped_features, None)
    value.backward()
    if build_inner is _mnrl:
        assert model.forward_calls == len(wrapped_features)

    manual_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    manual_value = sum(
        weight * plain(_pool_reference(manual_features, pooling), None)
        for weight, pooling in zip(weights, token_poolings)
    )
    manual_value.backward()

    torch.testing.assert_close(value, manual_value)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert leaf.grad.abs().sum() > 0
        torch.testing.assert_close(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(loss(_build_features(), None), manual_value)


def test_token_pooling_with_kd_loss() -> None:
    labels = torch.tensor([[5.0, 1.0], [4.0, 0.5], [3.0, 0.2], [6.0, 2.0]])
    pooling = HierarchicalTokenPooling(pool_factor=2)

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, kd, token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    value = loss(wrapped_features, labels)
    value.backward()

    manual_features = _build_features()
    plain = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    manual_value = plain(manual_features, labels)
    manual_value = manual_value + plain(_pool_reference(manual_features, pooling), labels)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_stacked_modifiers_cached_matches_non_cached() -> None:

    def build_stacked(model: nn.Module, inner: nn.Module) -> nn.Module:
        truncated = mve_losses.MultiVectorMatryoshkaLoss(model, inner, matryoshka_dims=[DIM, DIM // 2])
        return mve_losses.MultiVectorTokenPoolingLoss(
            model,
            truncated,
            token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)],
        )

    plain_features = _build_features()
    plain_leaves = _leaves(plain_features)
    plain_model = _PassthroughModel()
    plain_value = build_stacked(plain_model, _mnrl(plain_model))(plain_features, None)
    plain_value.backward()

    cached_features = _build_features()
    cached_leaves = _leaves(cached_features)
    cached_model = _PassthroughModel()
    cached_value = build_stacked(cached_model, _cached_mnrl(cached_model))(cached_features, None)
    cached_value.backward()

    assert torch.isfinite(plain_value)
    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_leaf, cached_leaf in zip(plain_leaves, cached_leaves):
        assert plain_leaf.grad.abs().sum() > 0
        assert torch.allclose(plain_leaf.grad, cached_leaf.grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@skip_fp16_cpu_layer_norm
def test_stacked_modifiers_fp16_padding_has_finite_gradients(mve_bert_tiny_model, build_inner) -> None:
    from sentence_transformers.util import batch_to_device

    model = mve_bert_tiny_model.to(device="cpu", dtype=torch.float16).eval()
    dim = model.get_embedding_dimension()
    pooling = HierarchicalTokenPooling(pool_factor=2)
    truncated = mve_losses.MultiVectorMatryoshkaLoss(model, build_inner(model), matryoshka_dims=[dim, dim // 2])
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, truncated, token_poolings=[None, pooling], weights=[0.5, 0.5])
    features = [
        batch_to_device(model.preprocess(texts, task=task), model.device)
        for texts, task in [
            (["what is retrieval", "how does pooling work"], "query"),
            (
                [
                    "short text",
                    "this longer document includes many distinct words to produce a different pooled token count",
                ],
                "document",
            ),
        ]
    ]
    with torch.no_grad():
        pooled = pooling(model(dict(features[1]), task="document"), task="document")
        assert (~pooled["attention_mask"]).any()
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_matryoshka_outside_token_pooling_raises(build_inner) -> None:
    model = _PassthroughModel()
    pooled = mve_losses.MultiVectorTokenPoolingLoss(
        model, build_inner(model), token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    with pytest.raises(ValueError, match="MultiVectorTokenPoolingLoss must wrap MultiVectorMatryoshkaLoss"):
        mve_losses.MultiVectorMatryoshkaLoss(model, pooled, matryoshka_dims=[DIM, DIM // 2])


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_stacked_modifiers_match_encoding(mve_bert_tiny_model, build_inner) -> None:
    from sentence_transformers.util import batch_to_device

    model = mve_bert_tiny_model
    model.eval()
    model.prompts = {}
    model.default_prompt_name = None
    queries = ["what is late interaction", "who wrote hamlet"]
    documents = [
        "late interaction compares each query token with the most similar document token",
        "william shakespeare wrote hamlet and many other plays for the london theatre",
    ]
    dims = [model.get_embedding_dimension(), model.get_embedding_dimension() // 2]
    dim_weights = [0.4, 0.6]
    poolings = [None, HierarchicalTokenPooling(max_tokens=4)]
    pooling_weights = [0.3, 0.7]
    truncated = mve_losses.MultiVectorMatryoshkaLoss(
        model, build_inner(model, scale=1.0), matryoshka_dims=dims, matryoshka_weights=dim_weights
    )
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, truncated, token_poolings=poolings, weights=pooling_weights)
    features = [
        batch_to_device(model.preprocess(texts, task=task), model.device)
        for texts, task in [(queries, "query"), (documents, "document")]
    ]

    with torch.no_grad():
        actual = loss(features, None)
        expected = 0.0
        for pooling_weight, pooling in zip(pooling_weights, poolings):
            for dim_weight, dim in zip(dim_weights, dims):
                query_embeddings = model.encode_query(queries, token_pooling=pooling, truncate_dim=dim)
                document_embeddings = model.encode_document(documents, token_pooling=pooling, truncate_dim=dim)
                scores = model.similarity(query_embeddings, document_embeddings)
                labels = torch.arange(len(queries), device=model.device)
                expected += pooling_weight * dim_weight * F.cross_entropy(scores, labels)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_empty_variant_list_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="at least one"):
        mve_losses.MultiVectorTokenPoolingLoss(model, _mnrl(model), token_poolings=[])


def test_non_pooling_entry_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="BaseTokenPooling"):
        mve_losses.MultiVectorTokenPoolingLoss(model, _mnrl(model), token_poolings=[nn.Identity()])


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("trainable", [False, True])
def test_pooler_parameters_and_outer_gradient_scaling(build_inner, trainable: bool) -> None:
    class LearnedPooler(BaseTokenPooling):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.7, 1.3, DIM), requires_grad=trainable)

        def pool_one(self, embedding: Tensor) -> Tensor:
            return embedding[::2] * self.weight

    model = _PassthroughModel()
    inner = build_inner(model)
    pooling = LearnedPooler()
    if build_inner is _cached_mnrl and trainable:
        calculate_loss = inner.calculate_loss
        with pytest.raises(ValueError, match="token_poolings with trainable parameters"):
            mve_losses.MultiVectorTokenPoolingLoss(model, inner, token_poolings=[None, pooling])
        assert inner.calculate_loss == calculate_loss
        return

    features = _build_features()
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, inner, token_poolings=[None, pooling], weights=[0.3, 0.7])
    value = loss(features, None)
    (value / 4).backward()

    reference_pooling = LearnedPooler()
    reference_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    expected = 0.3 * plain(reference_features, None)
    expected = expected + 0.7 * plain(_pool_reference(reference_features, reference_pooling), None)
    (expected / 4).backward()

    torch.testing.assert_close(value, expected)
    for actual, reference in zip(_leaves(features), _leaves(reference_features)):
        torch.testing.assert_close(actual.grad, reference.grad)
    if trainable:
        assert pooling.weight.grad.abs().sum() > 0
        torch.testing.assert_close(pooling.weight.grad, reference_pooling.weight.grad)
    else:
        assert pooling.weight.grad is None


def test_weights_length_mismatch_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="one weight per pooling variant"):
        mve_losses.MultiVectorTokenPoolingLoss(
            model,
            _mnrl(model),
            token_poolings=[HierarchicalTokenPooling(pool_factor=1), HierarchicalTokenPooling(pool_factor=2)],
            weights=[1.0],
        )


def test_get_config_dict_describes_variants() -> None:
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model,
        _mnrl(model),
        token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)],
        weights=[1.0, 0.5],
    )
    config = loss.get_config_dict()
    assert config["loss"] == "MultiVectorMultipleNegativesRankingLoss"
    assert config["weights"] == (1.0, 0.5)
    assert config["n_poolings_per_step"] == -1
    assert config["token_poolings"] == [
        None,
        "HierarchicalTokenPooling(pool_factor=2, max_tokens=None, distance_threshold=None, num_protected_tokens=1, normalize_embeddings=True, "
        "tasks=['document'])",
    ]


def test_token_pooling_on_a_real_model(mve_bert_tiny_model) -> None:
    model = mve_bert_tiny_model.train()
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
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, _mnrl(model), token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    grads = [parameter.grad for parameter in model[0].parameters() if parameter.grad is not None]
    assert grads and any(grad.abs().sum() > 0 for grad in grads)
    assert model.forward.__self__ is model
