from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MatryoshkaLoss,
    MSELoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util import cos_sim


@pytest.fixture
def mock_loss():
    """Create a mock base loss for testing."""

    class _FakeLoss(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    return _FakeLoss()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_shrink_zero_padding_has_finite_values_and_gradients(dtype: torch.dtype) -> None:
    from sentence_transformers.sentence_transformer.losses.matryoshka import shrink

    valid = torch.tensor([[True, False, False], [True, True, True]])
    embeddings = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(42), dtype=dtype)
    embeddings[~valid] = 0
    embeddings.requires_grad_()

    truncated = shrink(embeddings, 4)
    assert truncated.dtype == dtype
    assert torch.isfinite(truncated).all()
    assert truncated[~valid].count_nonzero() == 0
    torch.testing.assert_close(truncated[valid].float().norm(dim=-1), torch.ones(valid.sum()), atol=0.005, rtol=0.005)
    truncated[valid].float().sum().backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad[~valid].count_nonzero() == 0


def test_empty_matryoshka_dims_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that empty matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="You must provide at least one dimension in matryoshka_dims."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[])


def test_zero_dimension_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that zero dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[512, 0, 256])


def test_negative_dimension_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that negative dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[512, -100, 256])


def test_weights_length_mismatch_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that length mismatch between weights and dims raises ValueError."""
    with pytest.raises(ValueError, match="matryoshka_weights must be the same length as matryoshka_dims."):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],
            matryoshka_weights=[1.0, 0.5],  # Only 2 weights for 3 dims
        )


@pytest.mark.parametrize("truncate_dim", [None, 256])
def test_dimension_exceeds_model_dimension_raises_error(
    static_retrieval_mrl_en_v1_model, mock_loss, truncate_dim: int | None
):
    """Test that dimension exceeding model's embedding dimension raises ValueError."""
    static_retrieval_mrl_en_v1_model.truncate_dim = truncate_dim
    expected_msg = "Dimensions in matryoshka_dims cannot exceed the model's embedding dimension of 1024."
    with pytest.raises(ValueError, match=expected_msg):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 1024, 2048, 256],  # 2048 > 1024
        )


@pytest.mark.parametrize("truncate_dim", [None, 256])
def test_model_dimension_not_in_dims_warns(
    static_retrieval_mrl_en_v1_model, mock_loss, caplog, truncate_dim: int | None
):
    """Test that model dimension not in matryoshka_dims produces warning."""
    static_retrieval_mrl_en_v1_model.truncate_dim = truncate_dim
    with caplog.at_level("WARNING"):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],  # 1024 not included
        )

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    expected_msg = (
        "The model's embedding dimension 1024 is not included in matryoshka_dims: [512, 256, 128]. "
        "This means that the full model dimension won't be trained, which may lead to degraded performance "
        "when using the model without specifying a lower truncation dimension. It is strongly recommended to include "
        "1024 in matryoshka_dims."
    )
    assert caplog.records[0].message == expected_msg


def test_model_dimension_none_no_validation(mock_loss):
    """Test that when model dimension is None, no validation or warning occurs."""

    class _MockModel:
        def _get_untruncated_embedding_dimension(self):
            return None

    model = _MockModel()

    # This should not raise any error or warning, even with large dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=model,
            loss=mock_loss,
            matryoshka_dims=[2048, 1024, 512],  # Large dimensions, but no validation
        )


def test_valid_initialization_no_warnings(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that valid initialization with model dimension included produces no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[1024, 768, 512, 256, 128],  # 1024 included
        )


def test_valid_initialization_with_weights(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that valid initialization with custom weights works correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        loss = MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[1024, 768, 512, 256],
            matryoshka_weights=[1.0, 0.8, 0.6, 0.4],
        )

        # Verify that dimensions and weights are sorted in descending order
        assert loss.matryoshka_dims == (1024, 768, 512, 256)
        assert loss.matryoshka_weights == (1.0, 0.8, 0.6, 0.4)


def test_dimensions_sorted_descending(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that dimensions and weights are sorted in descending order."""
    loss = MatryoshkaLoss(
        model=static_retrieval_mrl_en_v1_model,
        loss=mock_loss,
        matryoshka_dims=[256, 768, 512, 1024],  # Unsorted
        matryoshka_weights=[0.6, 1.0, 0.8, 0.4],  # Corresponding weights
    )

    # Should be sorted in descending order
    assert loss.matryoshka_dims == (1024, 768, 512, 256)
    assert loss.matryoshka_weights == (0.4, 1.0, 0.8, 0.6)


def test_default_weights_when_none(static_retrieval_mrl_en_v1_model, mock_loss):
    """Test that default weights (all 1s) are used when weights is None."""
    loss = MatryoshkaLoss(
        model=static_retrieval_mrl_en_v1_model,
        loss=mock_loss,
        matryoshka_dims=[1024, 768, 512, 256],
        matryoshka_weights=None,
    )

    assert loss.matryoshka_weights == (1, 1, 1, 1)


def test_mse_loss_matryoshka(static_retrieval_mrl_en_v1_model):
    """Test MSELoss with multiple inputs (matryoshka)."""
    model = static_retrieval_mrl_en_v1_model.cpu()
    orig_loss = MSELoss(model=model)
    loss = MatryoshkaLoss(model=model, loss=orig_loss, matryoshka_dims=[32, 64, 128, 256, 512, 768, 1024])
    sentence_features = [model.preprocess(["This is an input text"])]
    x = torch.randn(1, 1024)  # Original dimension (teacher)

    # Call the loss function and verify output type and shape
    output = loss(sentence_features, x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([])  # MSELoss returns a scalar


def test_mse_loss_matryoshka_per_column_teacher(stsb_bert_tiny_model):
    """Per-column (3D) teacher embeddings are truncated like 2D ones instead of crashing."""
    model = stsb_bert_tiny_model.cpu().eval()
    loss = MatryoshkaLoss(model=model, loss=MSELoss(model=model), matryoshka_dims=[128, 32])
    text_columns = [
        ["A man is eating food.", "A plane is taking off.", "A child is playing outside."],
        ["Someone is having lunch.", "An aircraft is leaving the runway.", "A girl is riding a bicycle."],
    ]
    with torch.no_grad():
        teacher_embeddings = [model(model.preprocess(texts))["sentence_embedding"] for texts in text_columns]
        expected = torch.stack(
            [loss([model.preprocess(texts)], teacher) for texts, teacher in zip(text_columns, teacher_embeddings)]
        ).mean()
        sentence_features = [model.preprocess(texts) for texts in text_columns]
        output = loss(sentence_features, torch.stack(teacher_embeddings, dim=1))
    assert output.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("architecture", ["concatenated_pooling", "dense_projection"])
@pytest.mark.parametrize("truncate_dim", [None, 2])
@pytest.mark.parametrize("loss_cls", [MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss])
def test_matryoshka_dims_can_exceed_token_embedding_dimension(
    architecture: str, truncate_dim: int | None, loss_cls
) -> None:
    word_embeddings = WordEmbeddings(
        WhitespaceTokenizer(vocab=["PAD", "red", "blue", "green"]),
        embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        update_embeddings=True,
    )
    if architecture == "concatenated_pooling":
        modules = [word_embeddings, Pooling(2, ("mean", "max"))]
    else:
        modules = [word_embeddings, Pooling(2, "mean"), Dense(2, 4)]
    model = SentenceTransformer(modules=modules, device="cpu", truncate_dim=truncate_dim)
    assert model.get_embedding_dimension() == (truncate_dim or 4)

    anchors, positives = ["red", "blue"], ["green", "red blue"]
    kwargs = (
        {"mini_batch_size": 1, "show_progress_bar": False} if loss_cls is CachedMultipleNegativesRankingLoss else {}
    )
    loss_fn = MatryoshkaLoss(model, loss_cls(model, **kwargs), matryoshka_dims=[4, 2])
    assert model.truncate_dim == truncate_dim
    loss = loss_fn([model.preprocess(anchors), model.preprocess(positives)], None)

    anchor_embeddings = model(model.preprocess(anchors))["sentence_embedding"]
    positive_embeddings = model(model.preprocess(positives))["sentence_embedding"]
    expected = sum(
        torch.nn.functional.cross_entropy(
            cos_sim(anchor_embeddings[:, :dim], positive_embeddings[:, :dim]) * 20, torch.arange(len(anchors))
        )
        for dim in (4, 2)
    )
    torch.testing.assert_close(loss, expected)

    loss.backward()
    grads = [param.grad.clone() for param in model.parameters()]
    model.zero_grad()
    expected.backward()
    for grad, param in zip(grads, model.parameters()):
        torch.testing.assert_close(grad, param.grad, atol=1e-4, rtol=1e-4)


def test_cached_loss_behind_wrappers_matches_non_cached(stsb_bert_tiny_model: SentenceTransformer) -> None:
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    class ScaledLoss(torch.nn.Module):
        def __init__(self, loss):
            super().__init__()
            self.loss = loss
            self.uses_gradient_cache = getattr(loss, "uses_gradient_cache", False)

        def forward(self, features, labels):
            return 0.7 * self.loss(features, labels)

    model = stsb_bert_tiny_model.cpu()
    disable_dropout(model)
    model.train()

    def loss_and_grads(inner):
        model.zero_grad()
        features = [
            model.preprocess(texts) for texts in (["anchor a", "anchor b", "anchor c"], ["doc a", "doc b", "doc c"])
        ]
        loss = MatryoshkaLoss(model, ScaledLoss(ScaledLoss(inner)), matryoshka_dims=[128, 64, 32])
        value = loss(features, None)
        value.backward()
        return value.detach(), gradients(model)

    cached_value, cached_grads = loss_and_grads(CachedMultipleNegativesRankingLoss(model, mini_batch_size=2))
    plain_value, plain_grads = loss_and_grads(MultipleNegativesRankingLoss(model))
    assert_trained(cached_grads)
    torch.testing.assert_close(cached_value, plain_value, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cached_grads, plain_grads, rtol=1e-4, atol=1e-4)
