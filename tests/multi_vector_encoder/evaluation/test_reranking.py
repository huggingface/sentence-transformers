from __future__ import annotations

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorRerankingEvaluator,
)


def test_reranking_evaluator(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = mve_bert_tiny_model
    evaluator = MultiVectorRerankingEvaluator(
        samples=[
            {
                "query": "What is the capital of France?",
                "positive": ["Paris is the capital of France."],
                "negative": ["Berlin is the capital of Germany.", "Madrid is the capital of Spain."],
            },
        ],
        name="rerank_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "rerank_smoke_ndcg@10" in results
    assert evaluator.primary_metric == "rerank_smoke_ndcg@10"


def test_reranking_evaluator_token_pooling_and_truncate_dim(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    model = mve_bert_tiny_model
    samples = [
        {
            "query": "What is the capital of France?",
            "positive": ["Paris is the capital of France and its largest city by a wide margin."],
            "negative": ["Van Gogh painted The Starry Night during his stay in the asylum."],
        }
    ]
    evaluator = MultiVectorRerankingEvaluator(
        samples=samples,
        name="rerank_pooling",
        write_csv=False,
        token_pooling=HierarchicalTokenPooling(pool_factor=2),
        truncate_dim=64,
    )
    results = evaluator(model)
    assert 0.0 <= results[evaluator.primary_metric] <= 1.0
    assert evaluator.get_config_dict()["token_pooling"]["config"] == evaluator.token_pooling.get_config_dict()
