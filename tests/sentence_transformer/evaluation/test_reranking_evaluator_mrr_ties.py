"""Regression tests for RerankingEvaluator MRR with tied scores."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(
    ("positive", "negative", "at_k", "expected_mrr"),
    [
        pytest.param(["mid"], ["mid_tie"] * 4, 10, sum(1 / rank for rank in range(1, 6)) / 5, id="one-positive"),
        pytest.param(
            ["mid"], ["mid_tie"] * 9, 10, sum(1 / rank for rank in range(1, 11)) / 10, id="tie-ends-at-cutoff"
        ),
        pytest.param(
            ["mid"], ["mid_tie"] * 15, 10, sum(1 / rank for rank in range(1, 11)) / 16, id="tie-crosses-cutoff"
        ),
        pytest.param(["mid"], ["mid_tie"] * 20, 10, sum(1 / rank for rank in range(1, 11)) / 21, id="large-tie"),
        pytest.param(["mid"], ["mid_tie"] * 4, 1, 1 / 5, id="top-one"),
        pytest.param(["mid", "mid"], ["mid_tie", "mid_tie"], 10, 13 / 18, id="two-positives"),
        pytest.param(["high", "high"], ["low"], 10, 1.0, id="only-positives-tied"),
        pytest.param(["low", "mid", "mid"], ["high", "mid_tie", "mid_tie", "low"], 10, 29 / 72, id="mixed-scores"),
        pytest.param(["low", "mid", "mid"], ["high", "mid_tie", "mid_tie", "low"], 2, 1 / 4, id="mixed-scores-cutoff"),
        pytest.param(["mid"], ["high", "high", "mid_tie"], 2, 0.0, id="tie-below-cutoff"),
        pytest.param(["mid"], ["high", "low"], 10, 1 / 2, id="distinct-scores"),
        pytest.param(["mid"], ["high", "low"], 1, 0.0, id="distinct-scores-cutoff"),
        pytest.param(["low", "high"], ["mid"], 10, 1.0, id="best-positive-not-first"),
    ],
)
def test_reranking_evaluator_mrr(
    use_batched_encoding: bool, positive: list[str], negative: list[str], at_k: int, expected_mrr: float
) -> None:
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "query": 1, "low": 2, "mid": 3, "mid_tie": 4, "high": 5}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    model = SentenceTransformer(
        modules=[
            StaticEmbedding(
                tokenizer,
                embedding_weights=torch.tensor(
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.6, 0.8], [1.0, 0.0]]
                ),
            )
        ],
        similarity_fn_name="cosine",
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = RerankingEvaluator(
        samples=[{"query": "query", "positive": positive, "negative": negative}],
        at_k=at_k,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
    )
    assert evaluator(model)[f"mrr@{at_k}"] == pytest.approx(expected_mrr)
