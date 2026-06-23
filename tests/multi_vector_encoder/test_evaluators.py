from __future__ import annotations

import pytest

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorDistillationEvaluator,
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorRerankingEvaluator,
    MultiVectorTripletEvaluator,
)


@pytest.fixture(scope="module")
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")


def test_information_retrieval_evaluator(model: MultiVectorEncoder) -> None:
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {
        "d0": "Paris is the capital of France.",
        "d1": "Berlin is the capital of Germany.",
        "d2": "The Mona Lisa was painted by Leonardo da Vinci.",
        "d3": "Van Gogh painted The Starry Night.",
    }
    qrels = {"q0": {"d0"}, "q1": {"d2"}}

    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="ir_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "ir_smoke_maxsim_ndcg@10" in results
    assert evaluator.primary_metric == "ir_smoke_maxsim_ndcg@10"
    assert 0.0 <= results[evaluator.primary_metric] <= 1.0


def test_triplet_evaluator(model: MultiVectorEncoder) -> None:
    evaluator = MultiVectorTripletEvaluator(
        anchors=["What is the capital of France?"],
        positives=["Paris is the capital of France."],
        negatives=["Berlin is the capital of Germany."],
        name="triplet_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "triplet_smoke_maxsim_accuracy" in results
    assert 0.0 <= results["triplet_smoke_maxsim_accuracy"] <= 1.0


def test_reranking_evaluator(model: MultiVectorEncoder) -> None:
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


def test_distillation_evaluator(model: MultiVectorEncoder) -> None:
    evaluator = MultiVectorDistillationEvaluator(
        queries=["What is the capital of France?", "Who painted the Mona Lisa?"],
        documents=["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
        scores=[5.0, 5.0],
        name="distill_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "distill_smoke_spearman" in results
    assert "distill_smoke_kl_divergence" in results
    assert evaluator.primary_metric == "distill_smoke_spearman"
