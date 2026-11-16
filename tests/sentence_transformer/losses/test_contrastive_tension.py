from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveTensionDataLoader,
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
)
from sentence_transformers.sentence_transformer.modules import BoW


@pytest.mark.parametrize(
    "sentence_count, batch_size, ratio, expected_batches",
    [
        (0, 8, 8, 0),
        (1, 8, 8, 0),
        (14, 8, 8, 0),
        (15, 8, 8, 1),
        (16, 8, 8, 1),
        (29, 8, 8, 1),
        (30, 8, 8, 2),
        (31, 8, 8, 2),
        (30, 16, 8, 1),
        (6, 4, 2, 1),
        (4, 4, 1, 1),
        (1, 1, 1, 1),
    ],
)
def test_contrastive_tension_length_matches_batches(sentence_count, batch_size, ratio, expected_batches):
    loader = ContrastiveTensionDataLoader(
        [str(index) for index in range(sentence_count)], batch_size=batch_size, pos_neg_ratio=ratio
    )
    batches = list(iter(loader))

    assert len(loader) == len(batches) == expected_batches
    for batch in batches:
        assert len(batch) == batch_size
        assert sum(example.label for example in batch) == batch_size // ratio
        assert all((example.texts[0] == example.texts[1]) == bool(example.label) for example in batch)


def test_contrastive_tension_keeps_final_positive_pair():
    loader = ContrastiveTensionDataLoader(["one", "two"], batch_size=2, pos_neg_ratio=1)
    batches = list(iter(loader))
    assert len(batches) == 1
    assert {example.texts[0] for example in batches[0]} == {"one", "two"}


@pytest.mark.parametrize("batch_size", [1, 4])
def test_contrastive_tension_column_and_inferred_labels(batch_size):
    features = [
        {"sentence_embedding": torch.randn(batch_size, 8), "input_ids": torch.arange(batch_size)[:, None]},
        {"sentence_embedding": torch.randn(batch_size, 8), "input_ids": torch.zeros(batch_size, 1, dtype=torch.long)},
    ]
    labels = (features[0]["input_ids"] == features[1]["input_ids"]).all(dim=1).float()
    loss = ContrastiveTensionLoss(torch.nn.Identity())
    expected = loss(features, labels)
    torch.testing.assert_close(loss(features, labels[:, None]), expected)
    torch.testing.assert_close(loss(features, None), expected)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_contrastive_tension_multiple_labels_per_row_raise(batch_size):
    features = [{"sentence_embedding": torch.randn(batch_size, 8)} for _ in range(2)]
    loss = ContrastiveTensionLoss(torch.nn.Identity())
    with pytest.raises(ValueError, match="Target size"):
        loss(features, torch.zeros(batch_size, 3))


def test_contrastive_tension_in_batch_encoders_receive_original_features() -> None:
    model = SentenceTransformer(
        modules=[
            BoW(["alpha", "beta", "gamma"]),
            Dense(3, 3, bias=False, activation_function=torch.nn.Identity()),
        ],
        device="cpu",
    )
    with torch.no_grad():
        model[1].linear.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0])))
    loss = ContrastiveTensionLossInBatchNegatives(model, scale=1.0)
    features = model.preprocess(["alpha beta", "beta gamma"])
    original = features["sentence_embedding"].clone()

    # BoW leaves its features unchanged; both projections must consume the original counts.
    embeddings_a = loss.model1[1].linear(original)
    embeddings_b = loss.model2[1].linear(original)
    scores = loss.similarity_fct(embeddings_a, embeddings_b) * loss.logit_scale.exp()
    labels = torch.arange(2)
    expected = (loss.cross_entropy_loss(scores, labels) + loss.cross_entropy_loss(scores.T, labels)) / 2
    parameters = tuple(loss.parameters())
    expected_gradients = torch.autograd.grad(expected, parameters)

    actual = loss([features], None)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    for parameter, expected_gradient in zip(parameters, expected_gradients):
        torch.testing.assert_close(parameter.grad, expected_gradient)
    torch.testing.assert_close(features["sentence_embedding"], original)
