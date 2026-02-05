from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import SparseEncoder


@pytest.fixture(scope="session")
def _csr_bert_tiny_model() -> SparseEncoder:
    model = SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model[-1].k = 16
    model[-1].k_aux = 32
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def csr_bert_tiny_model(_csr_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    return deepcopy(_csr_bert_tiny_model)
