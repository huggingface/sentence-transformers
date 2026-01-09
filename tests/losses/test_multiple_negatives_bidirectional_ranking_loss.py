from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import InputExample, SentenceTransformer, losses, util


class _DummyModel:
    def __getitem__(self, idx):
        return object()


@pytest.fixture(scope="module")
def shared_sbert() -> SentenceTransformer:
    # Reuse a single model instance to avoid repeated downloads and initialization in CI.
    model = SentenceTransformer("distilbert-base-uncased")
    model.to("cpu")
    return model


def _manual_bidirectional_loss_with_negatives(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: list[torch.Tensor],
) -> torch.Tensor:
    # Manual InfoNCE with q->d, q->q (j!=i), d->q, d->d (j!=i) and hard negatives as extra docs.
    docs = torch.cat([positives] + negatives, dim=0) if negatives else positives
    sim_qd = util.dot_score(queries, docs)
    sim_qq = util.dot_score(queries, queries)
    sim_dd = util.dot_score(docs, docs)

    losses = []
    for i in range(queries.size(0)):
        qd_row = sim_qd[i]
        qq_row = sim_qq[i].clone()
        qq_row[i] = -torch.inf
        qd_col = sim_qd[:, i]
        dd_col = sim_dd[:, i].clone()
        dd_col[i] = -torch.inf
        scores = torch.cat([qd_row, qq_row, qd_col, dd_col], dim=0)
        log_z = torch.logsumexp(scores, dim=0)
        pos_score = sim_qd[i, i]
        losses.append(-(pos_score - log_z))

    return torch.stack(losses).mean()


def test_temperature_default_scale_property():
    loss = losses.MultipleNegativesBidirectionalRankingLoss(model=Mock(spec=SentenceTransformer))
    assert pytest.approx(loss.temperature) == 0.01
    assert pytest.approx(loss.scale) == 100.0

    cached_loss = losses.CachedMultipleNegativesBidirectionalRankingLoss(model=_DummyModel())
    assert pytest.approx(cached_loss.temperature) == 0.01
    assert pytest.approx(cached_loss.scale) == 100.0


def test_bidirectional_info_nce_manual_formula():
    loss = losses.MultipleNegativesBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
    )
    # NOTE: This test uses dot_score (not the default cos_sim) purely to keep the
    # manual calculation simple and exact. The default similarity in the loss is
    # cosine similarity (with L2 normalization).
    queries = torch.tensor([[1.0, 2.0], [0.3, -1.2]])
    docs = torch.tensor([[-0.7, 0.5], [1.5, -0.4]])

    computed = loss.compute_loss_from_embeddings([queries, docs], labels=None)
    # Manual check (dot similarity, temperature=1.0):
    # For q1=[1,2], d1=[-0.7,0.5], q2=[0.3,-1.2], d2=[1.5,-0.4],
    # the per-sample losses computed from
    #   L_i = -log( exp(s(q_i,d_i)) / Z_i ),
    # with Z_i summing q->d, q->q (j!=i), d->q, d->d (j!=i),
    # are 1.4169083311926611 and 1.1414837565860692, so the mean is:
    expected = 1.2791960438893653

    assert pytest.approx(computed.item(), rel=1e-6) == expected


def test_bidirectional_info_nce_temperature_value():
    loss = losses.MultipleNegativesBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=0.01,
        similarity_fct=util.dot_score,
    )
    queries = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    docs = torch.tensor([[0.1, 0.0], [0.0, -0.1]])

    computed = loss.compute_loss_from_embeddings([queries, docs], labels=None)
    # Manual check (dot similarity, temperature=0.01 -> scale=100.0):
    # Per-sample losses are 1.2445918944919965 and 2.5551419846181966,
    # so the mean is:
    expected = 1.8998669395550967

    assert pytest.approx(computed.item(), rel=1e-6) == expected


def test_bidirectional_info_nce_manual_formula_with_hard_negatives():
    loss = losses.MultipleNegativesBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
    )
    queries = torch.tensor([[1.0, 0.5], [-0.3, 0.8]])
    positives = torch.tensor([[0.2, -0.1], [0.4, 0.6]])
    negatives = torch.tensor([[-0.5, 0.7], [0.1, -0.9]])

    computed = loss.compute_loss_from_embeddings([queries, positives, negatives], labels=None)
    # Manual check (dot similarity, temperature=1.0, with hard negatives):
    # L_i = -log( exp(s(q_i,d_i)) / Z_i ),
    # with Z_i summing q->d (including hard negatives), q->q (j!=i), d->q, d->d (j!=i).
    expected = _manual_bidirectional_loss_with_negatives(queries, positives, [negatives])

    assert pytest.approx(computed.item(), rel=1e-6) == expected.item()


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
            1e-4,
        ),
    ],
)
def test_cached_bidirectional_info_nce_same_grad(
    train_samples: list[InputExample],
    scaler: float,
    precision: float,
    shared_sbert: SentenceTransformer,
):
    optimizer = Adam(shared_sbert.parameters())

    set_seed(42)
    optimizer.zero_grad()
    loss_base = losses.MultipleNegativesBidirectionalRankingLoss(shared_sbert)
    loss_base_value: torch.Tensor = loss_base.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_base_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_base.named_parameters() if p.grad is not None}

    set_seed(42)
    optimizer.zero_grad()
    loss_cached = losses.CachedMultipleNegativesBidirectionalRankingLoss(shared_sbert, mini_batch_size=2)
    loss_cached_value: torch.Tensor = loss_cached.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_cached_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cached.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_base_value.item(), rel=precision) == loss_cached_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    assert nclose == len(grad_expected)


def test_cached_bidirectional_info_nce_same_grad_with_hard_negatives(shared_sbert: SentenceTransformer):
    train_samples = [
        InputExample(texts=[q, p, n])
        for q, p, n in zip(
            ["aaa", "bbb", "ccc", "ddd"],
            ["aas", "bbs", "ccs", "dds"],
            ["zzz", "yyy", "xxx", "www"],
        )
    ]
    scaler = 1.0
    # GradCache + hard negatives can introduce tiny numerical differences vs. the non-cached path (order of ops).
    precision = 1e-4
    optimizer = Adam(shared_sbert.parameters())

    set_seed(42)
    optimizer.zero_grad()
    loss_base = losses.MultipleNegativesBidirectionalRankingLoss(shared_sbert)
    loss_base_value: torch.Tensor = loss_base.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_base_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_base.named_parameters() if p.grad is not None}

    set_seed(42)
    optimizer.zero_grad()
    loss_cached = losses.CachedMultipleNegativesBidirectionalRankingLoss(shared_sbert, mini_batch_size=2)
    loss_cached_value: torch.Tensor = loss_cached.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_cached_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cached.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_base_value.item(), rel=precision) == loss_cached_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    assert nclose == len(grad_expected)
