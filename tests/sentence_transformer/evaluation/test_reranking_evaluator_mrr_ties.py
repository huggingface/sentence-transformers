"""
Regression tests for RerankingEvaluator MRR with tied scores.

See https://github.com/huggingface/sentence-transformers/issues/4402: torch.argsort
is stable for small inputs, so a positive tied with negatives was always ranked
first (MRR 1.0), while for larger inputs the sort became unstable and MRR flipped
to 0.0 for the same tied scores. MRR now ranks the best-scoring positive
pessimistically, i.e. tied negatives count as ranked above it.
"""

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
    ("num_negatives", "expected_mrr"),
    [
        (4, 0.2),  # rank 5: 1 positive + 4 tied negatives
        (9, 0.1),  # rank 10
        (15, 0.0),  # rank 16 > at_k=10
    ],
)
def test_reranking_evaluator_mrr_tied_scores(
    use_batched_encoding: bool, num_negatives: int, expected_mrr: float
) -> None:
    """Tied scores must not inflate MRR: a positive tied with negatives ranks below them."""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    # "beta" and "gamma" share an embedding, so every document scores 0.0 against "alpha".
    model = SentenceTransformer(
        modules=[
            StaticEmbedding(
                tokenizer, embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
            )
        ],
        similarity_fn_name="cosine",
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = RerankingEvaluator(
        samples=[{"query": "alpha", "positive": ["beta"], "negative": ["gamma"] * num_negatives}],
        at_k=10,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
    )
    assert evaluator(model)["mrr@10"] == expected_mrr


@pytest.mark.parametrize("use_batched_encoding", [True, False])
def test_reranking_evaluator_mrr_distinct_scores_unchanged(use_batched_encoding: bool) -> None:
    """Without ties, MRR is the reciprocal rank of the best-scoring positive, as before."""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    # cosine("alpha", "delta") = 1.0 > cosine("alpha", "beta") = 0.6 > cosine("alpha", "gamma") = 0.0
    model = SentenceTransformer(
        modules=[
            StaticEmbedding(
                tokenizer,
                embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [1.0, 0.0]]),
            )
        ],
        similarity_fn_name="cosine",
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = RerankingEvaluator(
        samples=[{"query": "alpha", "positive": ["beta"], "negative": ["delta", "gamma"]}],
        at_k=10,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
    )
    assert evaluator(model)["mrr@10"] == 0.5
