from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)


@pytest.mark.parametrize(
    ["train_samples_mnrl", "train_samples_cmnrl", "same_grad", "scaler", "precision"],
    [
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            True,
            1.0,
            1e-5,
        ),
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["adsa", "czx", "dsada"],
                    ["b", "fas", "xcz"],
                    ["c", "yyy", "asdas"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            False,
            1.0,
            1e-5,
        ),
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            True,
            1000.0,
            1e-3,
        ),
    ],
)
def test_cmnrl_same_grad(
    train_samples_mnrl: list[tuple[str, str, str]],
    train_samples_cmnrl: list[tuple[str, str, str]],
    same_grad: bool,
    scaler: float,
    precision: float,
):
    # Given:
    model = SentenceTransformer("distilbert-base-uncased")
    model.to("cpu")
    optimizer = Adam(model.parameters())

    # When:
    # First run with MNRL
    set_seed(42)
    optimizer.zero_grad()
    loss_mnrl = MultipleNegativesRankingLoss(model)
    queries_mnrl, positives_mnrl, negatives_mnrl = zip(*train_samples_mnrl)
    features_mnrl = [model.preprocess(list(texts)) for texts in (queries_mnrl, positives_mnrl, negatives_mnrl)]
    labels = torch.zeros(len(train_samples_mnrl), dtype=torch.long)
    loss_mnrl_value: torch.Tensor = loss_mnrl(features_mnrl, labels) * scaler
    loss_mnrl_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_mnrl.named_parameters() if p.grad is not None}

    # Then run with this cached version:
    set_seed(42)
    optimizer.zero_grad()
    loss_cmnrl = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    queries_cmnrl, positives_cmnrl, negatives_cmnrl = zip(*train_samples_cmnrl)
    features_cmnrl = [model.preprocess(list(texts)) for texts in (queries_cmnrl, positives_cmnrl, negatives_cmnrl)]
    loss_cmnrl_value = loss_cmnrl(features_cmnrl, labels) * scaler
    loss_cmnrl_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cmnrl.named_parameters() if p.grad is not None}

    # Then:
    if same_grad:
        assert pytest.approx(loss_mnrl_value.item(), rel=precision, abs=precision) == loss_cmnrl_value.item()
    else:
        assert pytest.approx(loss_mnrl_value.item(), rel=precision, abs=precision) != loss_cmnrl_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    if same_grad:
        assert nclose == len(grad_expected)
    else:
        assert nclose != len(grad_expected)


@pytest.mark.parametrize("use_rand_context", [True, False])
def test_rand_context_working(use_rand_context: bool):
    # Given:
    from sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss import RandContext

    a = torch.Tensor(1)
    b = torch.Tensor(1)
    random_state = RandContext(a, b) if use_rand_context else nullcontext()
    expected = torch.rand(1000)
    precision = 1e-6

    # When:
    with random_state:
        # Then:
        if use_rand_context:
            assert torch.allclose(torch.rand(1000), expected, precision, precision)
        else:
            assert not torch.allclose(torch.rand(1000), expected, precision, precision)
