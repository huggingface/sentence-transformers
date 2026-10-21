from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import CrossEncoder, SentenceTransformer


@pytest.fixture(scope="session")
def _distilroberta_base_ce_model() -> CrossEncoder:
    return CrossEncoder("distilbert/distilroberta-base", num_labels=1)


@pytest.fixture()
def distilroberta_base_ce_model(_distilroberta_base_ce_model: CrossEncoder) -> CrossEncoder:
    return deepcopy(_distilroberta_base_ce_model)


@pytest.fixture(scope="session")
def _reranker_bert_tiny_model_v54() -> CrossEncoder:
    return CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce-STv54")


@pytest.fixture()
def reranker_bert_tiny_model_v54(_reranker_bert_tiny_model_v54) -> CrossEncoder:
    return deepcopy(_reranker_bert_tiny_model_v54)


@pytest.fixture(scope="session")
def _multiclass_bert_tiny_model(
    _stsb_bert_tiny_model: SentenceTransformer, tmp_path_factory: pytest.TempPathFactory
) -> CrossEncoder:
    model_path = tmp_path_factory.mktemp("multiclass_bert_tiny")
    _stsb_bert_tiny_model[0].model.save_pretrained(model_path)
    _stsb_bert_tiny_model.tokenizer.save_pretrained(model_path)
    return CrossEncoder(str(model_path), num_labels=3, local_files_only=True)


@pytest.fixture()
def multiclass_bert_tiny_model(_multiclass_bert_tiny_model: CrossEncoder) -> CrossEncoder:
    return deepcopy(_multiclass_bert_tiny_model)
