"""
Tests the RerankingEvaluator
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import cos_sim

ROWS = [
    {"query": "What sound does a cat make?", "positive": "Cats meow.", "negative": "Dogs bark loudly at night."},
    {
        "query": "What is the capital of France?",
        "positive": "Paris is the capital of France.",
        "negative": "Berlin is in Germany.",
    },
]


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"query": "q", "positive": ["a"], "negative": []}],
        [{"query": "q", "positive": [], "negative": ["a"]}],
    ],
)
def test_reranking_evaluator_without_valid_samples_raises(samples) -> None:
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        RerankingEvaluator(samples)


def test_reranking_evaluator_filters_invalid_samples() -> None:
    valid_sample = {"query": "q", "positive": ["a"], "negative": ["b"]}
    evaluator = RerankingEvaluator(
        [
            {"query": "q", "positive": ["a"], "negative": []},
            valid_sample,
            {"query": "q", "positive": [], "negative": ["b"]},
        ]
    )
    assert evaluator.samples == [valid_sample]


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(("string_positive", "string_negative"), [(True, False), (False, True), (True, True)])
def test_reranking_evaluator_string_documents(
    stsb_bert_tiny_model: SentenceTransformer,
    use_batched_encoding: bool,
    string_positive: bool,
    string_negative: bool,
) -> None:
    """A single positive or negative document given as a string is one document, not a list of characters"""
    as_lists = [{"query": row["query"], "positive": [row["positive"]], "negative": [row["negative"]]} for row in ROWS]
    expected = RerankingEvaluator(as_lists, name="dev", use_batched_encoding=use_batched_encoding)(
        stsb_bert_tiny_model
    )

    rows = [
        {
            **row,
            "positive": row["positive"] if string_positive else [row["positive"]],
            "negative": row["negative"] if string_negative else [row["negative"]],
        }
        for row in ROWS
    ]
    for samples in (rows, Dataset.from_list(rows)):
        evaluator = RerankingEvaluator(samples, name="dev", use_batched_encoding=use_batched_encoding)
        assert evaluator(stsb_bert_tiny_model) == pytest.approx(expected)


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(
    ("similarity_fn_name", "similarity_kwargs", "expected_score"),
    [
        ("dot", {}, 1.0),
        ("dot", {"similarity_fct": None}, 1.0),
        ("euclidean", {}, 1.0),
        ("manhattan", {}, 1.0),
        ("cosine", {}, 0.0),
        ("dot", {"similarity_fct": cos_sim}, 0.0),
    ],
)
def test_reranking_evaluator_similarity(
    use_batched_encoding: bool, similarity_fn_name: str, similarity_kwargs: dict, expected_score: float
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    model = SentenceTransformer(
        modules=[
            StaticEmbedding(
                tokenizer, embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.3], [0.1, 0.0]])
            )
        ],
        similarity_fn_name=similarity_fn_name,
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = RerankingEvaluator(
        samples=[{"query": "alpha", "positive": ["beta"], "negative": ["gamma"]}],
        at_k=1,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
        **similarity_kwargs,
    )

    cosine_model = deepcopy(model)
    cosine_model.similarity_fn_name = "cosine"
    for current_model, score in [(model, expected_score), (cosine_model, 0.0), (model, expected_score)]:
        metrics = evaluator(current_model)
        assert metrics["mrr@1"] == score
        assert metrics["ndcg@1"] == score


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(
    ("positive", "negative", "at_k", "expected_mrr"),
    [
        pytest.param(["mid"], ["mid_tie"] * 4, 10, sum(1 / rank for rank in range(1, 6)) / 5, id="one-positive"),
        pytest.param(["mid"], ["mid_tie"] * 20, 10, sum(1 / rank for rank in range(1, 11)) / 21, id="large-tie"),
        pytest.param(["low", "mid", "mid"], ["high", "mid_tie", "mid_tie", "low"], 10, 29 / 72, id="mixed-scores"),
        pytest.param(["mid"], ["high", "low"], 10, 1 / 2, id="distinct-scores"),
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
    results = evaluator(model)
    assert results[f"mrr@{at_k}"] == pytest.approx(expected_mrr)
    if len(positive) == 1 and len(positive) + len(negative) <= at_k:
        # A single positive's average precision is its reciprocal rank, also when its score is tied
        assert results["map"] == pytest.approx(expected_mrr)
