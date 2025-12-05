from __future__ import annotations

import pytest

from sentence_transformers import CrossEncoder


@pytest.fixture()
def distilroberta_base_ce_model() -> CrossEncoder:
    return CrossEncoder("distilroberta-base", num_labels=1)
