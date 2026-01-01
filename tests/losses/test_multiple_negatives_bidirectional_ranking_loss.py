from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import InputExample, SentenceTransformer, losses, util


def test_bidirectional_info_nce_manual_formula():
    loss = losses.MultipleNegativesBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        scale=1.0,
        similarity_fct=util.dot_score,
    )
    queries = torch.tensor([[1.0, 2.0], [0.3, -1.2]])
    docs = torch.tensor([[-0.7, 0.5], [1.5, -0.4]])

    computed = loss.compute_loss_from_embeddings([queries, docs], labels=None)
    # Manual check (dot similarity, scale=1.0):
    # For q1=[1,2], d1=[-0.7,0.5], q2=[0.3,-1.2], d2=[1.5,-0.4],
    # the per-sample losses computed from
    #   L_i = -log( exp(s(q_i,d_i)) / Z_i ),
    # with Z_i summing q->d, q->q (j!=i), d->q, d->d (j!=i),
    # are 1.4169083311926611 and 1.1414837565860692, so the mean is:
    expected = 1.2791960438893653

    assert pytest.approx(computed.item(), rel=1e-6) == expected


@pytest.mark.parametrize(
    ["train_samples", "scaler", "precision"],
    [
        (
            [
                InputExample(texts=[q, p])
                for q, p in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                )
            ],
            1.0,
            1e-5,
        ),
    ],
)
def test_cached_bidirectional_info_nce_same_grad(
    train_samples: list[InputExample],
    scaler: float,
    precision: float,
):
    sbert = SentenceTransformer("distilbert-base-uncased")
    sbert.to("cpu")
    optimizer = Adam(sbert.parameters())

    set_seed(42)
    optimizer.zero_grad()
    loss_base = losses.MultipleNegativesBidirectionalRankingLoss(sbert)
    loss_base_value: torch.Tensor = loss_base.forward(*sbert.smart_batching_collate(train_samples)) * scaler
    loss_base_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_base.named_parameters() if p.grad is not None}

    set_seed(42)
    optimizer.zero_grad()
    loss_cached = losses.CachedMultipleNegativesBidirectionalRankingLoss(sbert, mini_batch_size=2)
    loss_cached_value: torch.Tensor = loss_cached.forward(*sbert.smart_batching_collate(train_samples)) * scaler
    loss_cached_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cached.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_base_value.item()) == loss_cached_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    assert nclose == len(grad_expected)
