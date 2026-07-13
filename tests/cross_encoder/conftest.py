from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import CrossEncoder


@pytest.fixture()
def distilroberta_base_ce_model() -> CrossEncoder:
    return CrossEncoder("distilbert/distilroberta-base", num_labels=1)


@pytest.fixture(scope="session")
def _reranker_bert_tiny_model_v54() -> CrossEncoder:
    return CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce-STv54")


@pytest.fixture()
def reranker_bert_tiny_model_v54(_reranker_bert_tiny_model_v54) -> CrossEncoder:
    return deepcopy(_reranker_bert_tiny_model_v54)


@pytest.fixture(scope="session")
def _nli_minilm_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")


@pytest.fixture()
def nli_minilm_model(_nli_minilm_model) -> CrossEncoder:
    return deepcopy(_nli_minilm_model)
