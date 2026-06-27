from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.contrastive import ContrastiveLoss

LABEL_WARNING = "ContrastiveLoss expects binary labels.*"


class _EchoModel:
    def __call__(self, sentence_feature: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": sentence_feature["sentence_embedding"]}


def _features() -> list[dict[str, torch.Tensor]]:
    return [
        {"sentence_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
        {"sentence_embedding": torch.tensor([[0.9, 0.1], [1.0, 0.0]])},
    ]


def test_contrastive_loss_warns_once_for_non_binary_labels() -> None:
    loss = ContrastiveLoss(_EchoModel())
    labels = torch.tensor([1.0, 0.25])

    with pytest.warns(UserWarning, match="expects binary labels"):
        first_value = loss(_features(), labels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=LABEL_WARNING, category=UserWarning)
        second_value = loss(_features(), labels)

    assert not caught
    assert torch.isfinite(first_value)
    assert torch.isfinite(second_value)


def test_contrastive_loss_does_not_warn_for_binary_labels() -> None:
    loss = ContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=LABEL_WARNING, category=UserWarning)
        value = loss(_features(), torch.tensor([1.0, 0.0]))

    assert not caught
    assert torch.isfinite(value)
