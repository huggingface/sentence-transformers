from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("corpus_override", ["model", "sparse", "dense"])
@pytest.mark.parametrize("corpus_chunk_size", [1, 2])
def test_corpus_overrides(positional: bool, corpus_override: str, corpus_chunk_size: int) -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    query_model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 2.0, 1.0]))], device="cpu"
    )
    corpus_model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 0.0, 2.0]))], device="cpu"
    )
    for model in (query_model, corpus_model):
        model.model_card_data.generate_widget_examples = False

    evaluator = SparseInformationRetrievalEvaluator(
        queries={"q": "alpha beta"},
        corpus={"irrelevant": "alpha", "relevant": "beta"},
        relevant_docs={"q": {"relevant"}},
        corpus_chunk_size=corpus_chunk_size,
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        mrr_at_k=[1],
        ndcg_at_k=[1],
        map_at_k=[1],
        write_csv=False,
    )

    if corpus_override == "model":
        args, kwargs = [corpus_model], {"corpus_model": corpus_model}
    else:
        corpus_embeddings = corpus_model.encode_document(["alpha", "beta"], convert_to_sparse_tensor=True)
        if corpus_override == "dense":
            corpus_embeddings = corpus_embeddings.to_dense()
        args, kwargs = [None, corpus_embeddings], {"corpus_embeddings": corpus_embeddings}

    if positional:
        metrics = evaluator(query_model, None, -1, -1, *args)
    else:
        metrics = evaluator(query_model, **kwargs)

    # Only the document encoder ranks beta above alpha for the query.
    assert metrics["dot_ndcg@1"] == 1.0
    assert metrics["dot_recall@1"] == 1.0
    assert metrics["corpus_active_dims"] == 0.5
    assert metrics["avg_flops"] == 0.5

    # Reusing the evaluator without a separate encoder still uses the query model.
    default_metrics = evaluator(query_model)
    assert default_metrics["dot_ndcg@1"] == 0.0
    assert default_metrics["corpus_active_dims"] == 1.0
    assert default_metrics["avg_flops"] == 1.0
