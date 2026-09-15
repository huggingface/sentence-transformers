from __future__ import annotations

import pytest

from sentence_transformers.sentence_transformer.losses import ContrastiveTensionDataLoader


@pytest.mark.parametrize("batch_size, ratio", [(8, 8), (16, 8), (4, 2), (4, 1), (1, 1)])
@pytest.mark.parametrize("sentence_count", [0, 1, 7, 8, 15, 16, 30, 31, 60])
def test_contrastive_tension_length_matches_batches(batch_size, ratio, sentence_count):
    loader = ContrastiveTensionDataLoader(
        [str(index) for index in range(sentence_count)], batch_size=batch_size, pos_neg_ratio=ratio
    )
    batches = list(iter(loader))

    assert len(loader) == len(batches)
    for batch in batches:
        assert len(batch) == batch_size
        assert sum(example.label for example in batch) == batch_size // ratio
        assert all((example.texts[0] == example.texts[1]) == bool(example.label) for example in batch)


def test_contrastive_tension_keeps_final_positive_pair():
    loader = ContrastiveTensionDataLoader(["one", "two"], batch_size=2, pos_neg_ratio=1)
    batches = list(iter(loader))
    assert len(batches) == 1
    assert {example.texts[0] for example in batches[0]} == {"one", "two"}
