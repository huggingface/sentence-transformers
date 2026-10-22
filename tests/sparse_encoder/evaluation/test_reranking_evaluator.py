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


@pytest.mark.parametrize("use_batched_encoding", [True, False])
def test_reranking_evaluator_uses_model_similarity(use_batched_encoding: bool) -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    model = SparseEncoder(modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 2.0, 1.0, 2.0]))])
    model.model_card_data.generate_widget_examples = False

    evaluator = SparseRerankingEvaluator(
        samples=[{"query": "alpha beta", "positive": ["alpha beta gamma"], "negative": ["alpha"]}],
        at_k=1,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
    )
    metrics = evaluator(model)

    # The model's dot product ranks the positive first (5 vs 4), cosine similarity would not (0.75 vs 0.89).
    assert model.similarity_fn_name == "dot"
    assert metrics["mrr@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
