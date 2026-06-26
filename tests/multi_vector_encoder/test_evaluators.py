from __future__ import annotations

import pytest

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorDistillationEvaluator,
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
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


def test_information_retrieval_evaluator_rejects_xtr_scoring() -> None:
    """XTR's global top-k is incompatible with the evaluator's per-chunk corpus scoring, so an XTR
    scorer in ``score_functions`` must raise at construction rather than emit per-chunk-wrong metrics.
    """
    from functools import partial

    from sentence_transformers.multi_vector_encoder.scoring import XTRScores, xtr_scores

    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France."}
    qrels = {"q0": {"d0"}}
    for scorer in (xtr_scores, XTRScores(k=2), partial(xtr_scores, document_chunk_size=4)):
        with pytest.raises(ValueError, match="XTR"):
            MultiVectorInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=qrels,
                name="x",
                write_csv=False,
                score_functions={"x": scorer},
            )


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


def test_triplet_evaluator_writes_csv_rows(model: MultiVectorEncoder, tmp_path) -> None:
    # Regression: the prior bespoke ``__call__`` registered CSV headers but never wrote rows, so
    # ``write_csv=True`` produced a header-only CSV.
    evaluator = MultiVectorTripletEvaluator(
        anchors=["What is the capital of France?"],
        positives=["Paris is the capital of France."],
        negatives=["Berlin is the capital of Germany."],
        name="csv_smoke",
        write_csv=True,
    )
    evaluator(model, output_path=str(tmp_path))
    csv_path = tmp_path / evaluator.csv_file
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2, f"Expected at least header + 1 row, got {len(lines)} line(s)"


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


def test_nano_beir_evaluator_emits_lowercase_maxsim_key(model: MultiVectorEncoder) -> None:
    # The four training examples set ``metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10"``,
    # so a regression in the lowercase ``maxsim`` segment would break ``load_best_model_at_end``.
    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France.", "d1": "Berlin is the capital of Germany."}
    qrels = {"q0": {"d0"}}

    class _StubNanoBEIR(MultiVectorNanoBEIREvaluator):
        def _load_dataset(self, dataset_name: str, **ir_kwargs):
            return MultiVectorInformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=qrels, name=f"Nano{dataset_name}", **ir_kwargs
            )

    evaluator = _StubNanoBEIR(dataset_names=["msmarco"], write_csv=False)
    results = evaluator(model)
    assert "NanoBEIR_mean_maxsim_ndcg@10" in results
    assert evaluator.primary_metric == "NanoBEIR_mean_maxsim_ndcg@10"


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
