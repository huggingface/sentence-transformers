from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim


@pytest.fixture
def mock_model():
    def mock_encode(sentences: str | list[str], **kwargs) -> torch.Tensor:
        """
        We simply one-hot encode the sentences. If a sentence contains a keyword, the corresponding one-hot
        encoding is added to the sentence embedding.
        """
        one_hot_encodings = {
            "pokemon": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            "car": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
            "vehicle": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
            "fruit": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
            "vegetable": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for sentence in sentences:
            encoding = torch.zeros(5)
            for keyword, one_hot in one_hot_encodings.items():
                if keyword in sentence:
                    encoding += one_hot
            embeddings.append(encoding)
        return torch.stack(embeddings)

    class _MockModelCardData:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class _MockModel:
        similarity_fn_name = "cosine"
        model_card_data = _MockModelCardData()

        def similarity(self, a, b):
            return cos_sim(a, b)

        def encode(self, sentences, **kwargs):
            return mock_encode(sentences, **kwargs)

        def encode_query(self, sentences, **kwargs):
            return mock_encode(sentences, **kwargs)

        def encode_document(self, sentences, **kwargs):
            return mock_encode(sentences, **kwargs)

    return _MockModel()


@pytest.fixture
def test_data():
    queries = {
        "0": "What is a pokemon?",
        "1": "What is a vegetable?",
        "2": "What is a fruit?",
        "3": "What is a vehicle?",
        "4": "What is a car?",
    }
    corpus = {
        "0": "A pokemon is a fictional creature",
        "1": "A vegetable is a plant",
        "2": "A fruit is a plant",
        "3": "A vehicle is a machine",
        "4": "A car is a vehicle",
    }
    relevant_docs = {"0": {"0"}, "1": {"1"}, "2": {"2"}, "3": {"3", "4"}, "4": {"4"}}
    return queries, corpus, relevant_docs


def test_simple(test_data, stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path):
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(model, output_path=str(tmp_path))
    expected_keys = [
        "test_cosine_accuracy@1",
        "test_cosine_accuracy@3",
        "test_cosine_precision@1",
        "test_cosine_precision@3",
        "test_cosine_recall@1",
        "test_cosine_recall@3",
        "test_cosine_ndcg@3",
        "test_cosine_mrr@3",
        "test_cosine_map@5",
    ]
    assert set(results.keys()) == set(expected_keys)


def test_reused_evaluator_follows_the_model_similarity(
    test_data, stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path, caplog
):
    """Without explicit score_functions the scoring is resolved per call, so a reused evaluator
    labels every model with its own similarity rather than with the first one's."""
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test"
    )
    ir_evaluator(model, output_path=str(tmp_path))
    header_count = len(ir_evaluator.csv_headers)

    original = model.similarity_fn_name
    model.similarity_fn_name = "dot"
    try:
        with caplog.at_level("WARNING"):
            results = ir_evaluator(model, output_path=str(tmp_path))
    finally:
        model.similarity_fn_name = original

    assert ir_evaluator.score_function_names == ["dot"]
    assert ir_evaluator.primary_metric == "test_dot_ndcg@10"
    assert {key.split("_")[1] for key in results} == {"dot"}
    assert len(ir_evaluator.csv_headers) == header_count
    # The CSV keeps the header row of the first call, so the appended row is mislabeled.
    assert "labeled with the previous score function" in caplog.text


def test_explicit_score_functions_survive_reuse(test_data, stsb_bert_tiny_model: SentenceTransformer):
    """Explicit score_functions are evaluator configuration, so a model carrying a different
    similarity does not replace them."""
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model
    score_functions = {"custom": cos_sim}

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        score_functions=score_functions,
        write_csv=False,
    )
    ir_evaluator(model)

    original = model.similarity_fn_name
    model.similarity_fn_name = "dot"
    try:
        results = ir_evaluator(model)
    finally:
        model.similarity_fn_name = original

    assert ir_evaluator.score_functions is score_functions
    assert ir_evaluator.score_function_names == ["custom"]
    assert ir_evaluator.primary_metric == "test_custom_ndcg@10"
    assert {key.split("_")[1] for key in results} == {"custom"}


def test_metrics_are_independent_of_corpus_chunk_size(stsb_bert_tiny_model: SentenceTransformer):
    """Duplicate documents produce exact score ties: breaking them by corpus_id keeps every metric
    identical across corpus_chunk_size values."""
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {f"d{idx:02d}": "Paris is the capital of France." for idx in range(20)}
    relevant_docs = {"q0": {"d00", "d15"}, "q1": {"d07"}}

    results_per_chunk_size = {}
    # Every effective chunk length (4, 8, 20, 20) is a multiple of batch_size 4, so every encode
    # batch holds 4 identical texts and the duplicates tie bit-exactly in every chunking (larger
    # batches vary in the last ulp).
    for corpus_chunk_size in [4, 8, 20, 50]:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="chunked",
            corpus_chunk_size=corpus_chunk_size,
            batch_size=4,
            accuracy_at_k=[1, 5],
            precision_recall_at_k=[1, 5],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[10],
            write_csv=False,
        )
        results_per_chunk_size[corpus_chunk_size] = ir_evaluator(stsb_bert_tiny_model)

    baseline = results_per_chunk_size[50]
    for chunk_size, results in results_per_chunk_size.items():
        assert results == baseline, f"corpus_chunk_size={chunk_size} changed the metrics"
    # All 20 documents tie, so ranks follow ascending corpus_id: only q0 has its d00 at rank 1.
    assert baseline["chunked_cosine_accuracy@1"] == 0.5


def test_metrics(test_data, mock_model, tmp_path: Path):
    queries, corpus, relevant_docs = test_data

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(mock_model, output_path=str(tmp_path))
    # We expect test_cosine_precision@3 to be 0.4, since 6 out of 15 (5 queries * 3) are True Positives
    # We expect test_cosine_recall@1 to be 0.9: the average of 4 times a recall of 1 and once a recall of 0.5
    expected_results = {
        "test_cosine_accuracy@1": 1.0,
        "test_cosine_accuracy@3": 1.0,
        "test_cosine_precision@1": 1.0,
        "test_cosine_precision@3": 0.4,
        "test_cosine_recall@1": 0.9,
        "test_cosine_recall@3": 1.0,
        "test_cosine_ndcg@3": 1.0,
        "test_cosine_mrr@3": 1.0,
        "test_cosine_map@5": 1.0,
    }

    for key, expected_value in expected_results.items():
        assert results[key] == pytest.approx(expected_value, abs=1e-9)


def test_tie_reselection_only_builds_keys_for_tied_queries(mock_model, monkeypatch):
    """Only the queries with a tie at the top-k cutoff need the (-score, corpus_id) reselection, so its
    integer keys must cover those rows only, not the whole (queries x corpus chunk) score matrix."""
    corpus = {"d0": "car", "d1": "car", "d2": "pokemon", "d3": "fruit", "d4": "vegetable", "d5": "vehicle"}
    # q0 ties between d0 and d1 at the cutoff; every other query has a single best document
    untied = ["pokemon", "fruit", "vegetable", "vehicle"] * 2
    queries = {"q0": "car", **{f"q{idx + 1}": text for idx, text in enumerate(untied)}}
    relevant_docs = {"q0": {"d1"}, **{f"q{idx + 1}": {"d2"} for idx in range(len(untied))}}

    where_sizes = []
    original_where = torch.where

    def recording_where(*args, **kwargs):
        result = original_where(*args, **kwargs)
        where_sizes.append(result.numel())
        return result

    monkeypatch.setattr(torch, "where", recording_where)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="ties",
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        mrr_at_k=[1],
        ndcg_at_k=[1],
        map_at_k=[1],
        write_csv=False,
    )
    results = ir_evaluator(mock_model)

    assert where_sizes, "the tied query should have been reselected"
    assert max(where_sizes) <= len(corpus), "tie keys were built for untied queries too"
    # d0 wins the tie by corpus_id, so q0 misses its relevant d1 at rank 1; the other 8 queries hit d2 or miss
    assert results["ties_cosine_accuracy@1"] == pytest.approx(2 / 9)


@pytest.mark.parametrize("corpus_chunk_size", [1, 2, 3, 4, 100])
def test_ranking_follows_score_then_corpus_id(mock_model, tmp_path: Path, corpus_chunk_size: int):
    """Mixed tied and untied queries, including chunks smaller than top_k, rank by (-score, corpus_id)."""
    corpus = {
        "d5": "car",
        "d1": "car vehicle",
        "d3": "car",
        "d0": "pokemon",
        "d4": "fruit vegetable",
        "d2": "fruit",
    }
    queries = {"q0": "car", "q1": "pokemon", "q2": "fruit", "q3": "car vehicle"}
    relevant_docs = {query_id: {"d0"} for query_id in queries}
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="order",
        corpus_chunk_size=corpus_chunk_size,
        accuracy_at_k=[3],
        precision_recall_at_k=[3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[3],
        write_csv=False,
        write_predictions=True,
    )
    ir_evaluator(mock_model, output_path=str(tmp_path))

    query_embeddings = mock_model.encode(list(queries.values()))
    corpus_embeddings = mock_model.encode(list(corpus.values()))
    scores = cos_sim(query_embeddings, corpus_embeddings).tolist()
    predictions = [
        json.loads(line)
        for line in (tmp_path / "Information-Retrieval_evaluation_order_predictions_cosine.jsonl").open()
    ]
    for query_idx, prediction in enumerate(predictions):
        expected = sorted(zip(scores[query_idx], corpus), key=lambda pair: (-pair[0], pair[1]))[:3]
        assert [result["corpus_id"] for result in prediction["results"]] == [corpus_id for _, corpus_id in expected]
