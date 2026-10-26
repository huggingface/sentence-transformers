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
