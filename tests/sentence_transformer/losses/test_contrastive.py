from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.contrastive import ContrastiveLoss
from sentence_transformers.sentence_transformer.losses.online_contrastive import OnlineContrastiveLoss

CONTRASTIVE_WARNING = "ContrastiveLoss expects binary labels.*"
ONLINE_WARNING = "OnlineContrastiveLoss expects binary labels.*"


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

    with pytest.warns(UserWarning, match="ContrastiveLoss expects binary labels"):
        first_value = loss(_features(), labels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        second_value = loss(_features(), labels)

    assert not caught
    assert torch.isfinite(first_value)
    assert torch.isfinite(second_value)


def test_contrastive_loss_does_not_warn_for_binary_labels() -> None:
    loss = ContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        value = loss(_features(), torch.tensor([1.0, 0.0]))

    assert not caught
    assert torch.isfinite(value)


def test_contrastive_loss_only_checks_the_first_batch() -> None:
    # The label check runs once, on the first forward, to avoid a per-step device sync. Non-binary
    # labels that first appear in a later batch are therefore not flagged (labels are fixed per run).
    loss = ContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        loss(_features(), torch.tensor([1.0, 0.0]))
        loss(_features(), torch.tensor([1.0, 0.25]))

    assert not caught


def test_online_contrastive_loss_warns_once_for_non_binary_labels() -> None:
    loss = OnlineContrastiveLoss(_EchoModel())
    labels = torch.tensor([1.0, 0.25])

    with pytest.warns(UserWarning, match="OnlineContrastiveLoss expects binary labels"):
        first_value = loss(_features(), labels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=ONLINE_WARNING, category=UserWarning)
        second_value = loss(_features(), labels)

    assert not caught
    assert torch.isfinite(first_value)
    assert torch.isfinite(second_value)


def test_online_contrastive_loss_does_not_warn_for_binary_labels() -> None:
    loss = OnlineContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=ONLINE_WARNING, category=UserWarning)
        value = loss(_features(), torch.tensor([1.0, 0.0]))

    assert not caught
    assert torch.isfinite(value)
