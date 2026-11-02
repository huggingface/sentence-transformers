from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding
from sentence_transformers.util import cos_sim


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(
    "similarity_fn_name, similarity_fct, expected_score",
    [("dot", None, 1.0), ("cosine", None, 0.0), ("dot", cos_sim, 0.0)],
)
def test_reranking_evaluator_similarity(
    use_batched_encoding: bool, similarity_fn_name: str, similarity_fct, expected_score: float
) -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 2.0, 1.0, 2.0]))],
        similarity_fn_name=similarity_fn_name,
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = SparseRerankingEvaluator(
        samples=[{"query": "alpha beta", "positive": ["alpha beta gamma"], "negative": ["alpha"]}],
        at_k=1,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
        similarity_fct=similarity_fct,
    )

    metrics = evaluator(model)
    assert metrics["mrr@1"] == expected_score
    assert metrics["ndcg@1"] == expected_score
