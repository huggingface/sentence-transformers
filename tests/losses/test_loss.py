from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    CachedMultipleNegativesSymmetricRankingLoss,
    GISTEmbedLoss,
    MegaBatchMarginLoss,
    MultipleNegativesRankingLoss,
    SoftmaxLoss,
    TripletLoss,
)
from sentence_transformers.util import batch_to_device


@pytest.fixture(scope="module")
def guide_model():
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("sentence-transformers-testing/all-nli-bert-tiny-dense")


def get_anchor_positive_negative_triplet():
    return {
        "losses": [
            (MultipleNegativesRankingLoss, {}),
            (CachedMultipleNegativesRankingLoss, {}),
            (CachedMultipleNegativesSymmetricRankingLoss, {}),
            (TripletLoss, {}),
            (CachedGISTEmbedLoss, {"guide": "GUIDE_MODEL_PLACEHOLDER"}),
            (GISTEmbedLoss, {"guide": "GUIDE_MODEL_PLACEHOLDER"}),
            (SoftmaxLoss, {"num_labels": 3}),
            (
                MegaBatchMarginLoss,
                {
                    "positive_margin": 0.8,
                    "negative_margin": 0.3,
                    "use_mini_batched_version": True,
                    "mini_batch_size": 2,
                },
            ),
            (
                MegaBatchMarginLoss,
                {
                    "positive_margin": 0.8,
                    "negative_margin": 0.3,
                    "use_mini_batched_version": False,
                    "mini_batch_size": 2,
                },
            ),
        ],
        "correct": Dataset.from_dict(
            {
                "anchor": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "positive": ["The sun is out today", "I like playing soccer", "I am studying at university"],
                "negative": ["Data science is fun", "Cacti are beautiful", "Speakers are loud"],
            }
        ),
        "incorrect": Dataset.from_dict(
            {
                "anchor": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "positive": ["Data science is fun", "Cacti are beautiful", "Speakers are loud"],
                "negative": ["The sun is out today", "I like playing soccer", "I am studying at university"],
            }
        ),
        "softmax_correct": Dataset.from_dict(
            {
                "sentence1": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "sentence2": ["The sun is out today", "I like playing soccer", "I am studying at university"],
                "label": [1, 1, 0],
            }
        ),
        "softmax_incorrect": Dataset.from_dict(
            {
                "sentence1": ["It's very sunny outside", "I love playing soccer", "I am a student"],
                "sentence2": ["Data science is fun", "Cacti are beautiful", "Speakers are loud"],
                "label": [2, 2, 1],
            }
        ),
    }


def get_loss_test_cases():
    anchor_positive_negative_triplet = get_anchor_positive_negative_triplet()
    test_cases = []
    for loss_class, loss_args in anchor_positive_negative_triplet["losses"]:
        if loss_class == SoftmaxLoss:
            loss_args["sentence_embedding_dimension"] = 384
            test_cases.append(
                (
                    loss_class,
                    loss_args,
                    anchor_positive_negative_triplet["softmax_correct"],
                    anchor_positive_negative_triplet["softmax_incorrect"],
                )
            )
        else:
            test_cases.append(
                (
                    loss_class,
                    loss_args,
                    anchor_positive_negative_triplet["correct"],
                    anchor_positive_negative_triplet["incorrect"],
                )
            )
    return test_cases


def prepare_features_labels_from_dataset(model: SentenceTransformer, loss_fn: nn.Module, dataset: Dataset):
    device = model.device
    if "sentence1" in dataset.column_names and "sentence2" in dataset.column_names:
        # Handle SoftmaxLoss case
        features = [batch_to_device(model.tokenize(dataset[column]), device) for column in ["sentence1", "sentence2"]]
        labels = torch.tensor(dataset["label"]).to(device)
    else:
        # Handle other losses
        columns = [col for col in dataset.column_names if col not in ["label", "score"]]
        if isinstance(loss_fn, MegaBatchMarginLoss):
            # For MegaBatchMarginLoss, only use anchor and positive
            columns = ["anchor", "positive"]
        features = [batch_to_device(model.tokenize(dataset[column]), device) for column in columns]
        labels = None
        if "label" in dataset.column_names:
            labels = torch.tensor(dataset["label"]).to(device)
        elif "score" in dataset.column_names:
            labels = torch.tensor(dataset["score"]).to(device)
    return features, labels


def get_and_assert_loss_from_dataset(model: SentenceTransformer, loss_fn: nn.Module, dataset: Dataset):
    features, labels = prepare_features_labels_from_dataset(model, loss_fn, dataset)

    loss = loss_fn.forward(features, labels)
    assert isinstance(loss, torch.Tensor), f"Loss should be a torch.Tensor, but got {type(loss)}"
    assert loss.item() > 0, "Loss should be positive"
    assert loss.shape == (), "Loss should be a scalar"
    assert loss.requires_grad, "Loss should require gradients"
    return loss


@pytest.mark.parametrize("loss_class, loss_args, correct, incorrect", get_loss_test_cases())
def test_loss_function(model, guide_model, loss_class, loss_args, correct, incorrect):
    if "guide" in loss_args and loss_args["guide"] == "GUIDE_MODEL_PLACEHOLDER":
        loss_args["guide"] = guide_model

    if loss_class == SoftmaxLoss:
        loss_args["sentence_embedding_dimension"] = model.get_sentence_embedding_dimension()
    loss_fn = loss_class(model, **loss_args)
    correct_loss = get_and_assert_loss_from_dataset(model, loss_fn, correct)
    incorrect_loss = get_and_assert_loss_from_dataset(model, loss_fn, incorrect)

    assert correct_loss < incorrect_loss, "Loss should be lower for correct data than for incorrect data"
