from __future__ import annotations

import pytest
import torch

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
