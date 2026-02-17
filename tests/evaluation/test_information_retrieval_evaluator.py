from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, PropertyMock

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim


@pytest.fixture
def mock_model():
    def mock_encode(sentences: str | list[str], **kwargs) -> torch.Tensor:
        """
        We simply one-hot encode the sentences; if a sentence contains a keyword, the corresponding one-hot
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

    model = Mock(spec=SentenceTransformer)
    model.similarity_fn_name = "cosine"
    model.similarity.side_effect = cos_sim
    model.encode.side_effect = mock_encode
    model.encode_query.side_effect = mock_encode
    model.encode_document.side_effect = mock_encode
    model.model_card_data = PropertyMock(return_value=Mock())
    return model


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


def test_simple(test_data, tmp_path: Path):
    queries, corpus, relevant_docs = test_data
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")

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


def test_metrices(test_data, mock_model, tmp_path: Path):
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
    # We expect test_cosine_recall@1 to be 0.9; the average of 4 times a recall of 1 and once a recall of 0.5
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
