from __future__ import annotations

import pytest

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.util import is_training_available


@pytest.mark.parametrize(
    ("revision", "expected_base_revision"),
    [
        ("f3cb857cba53019a20df283396bcca179cf051a4", "f3cb857cba53019a20df283396bcca179cf051a4"),
        ("f3cb857", "f3cb857"),
        ("main", "valid-revision"),
        (None, "valid-revision"),
    ],
)
def test_model_card_data(revision, expected_base_revision) -> None:
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SentenceTransformer(model_name, revision=revision)

    assert model.model_card_data.base_model == model_name
    if expected_base_revision == "valid-revision":
        assert model.model_card_data.base_model_revision
        assert len(model.model_card_data.base_model_revision) == 40
    else:
        assert model.model_card_data.base_model_revision == expected_base_revision


@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_generated_from_trainer_tag(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model

    assert "generated_from_trainer" not in model.model_card_data.tags
    SentenceTransformerTrainer(model)
    assert "generated_from_trainer" in model.model_card_data.tags
