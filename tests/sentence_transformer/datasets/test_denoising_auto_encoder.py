from __future__ import annotations

import pickle

import pytest
from transformers.utils.import_utils import is_nltk_available

from sentence_transformers.sentence_transformer.datasets.denoising_auto_encoder import DenoisingAutoEncoderDataset

pytestmark = pytest.mark.skipif(not is_nltk_available(), reason="nltk is required for DenoisingAutoEncoderDataset")


def test_default_noise_fn_is_picklable() -> None:
    # DataLoader workers started via "spawn" pickle the dataset
    dataset = DenoisingAutoEncoderDataset(["This is a sentence.", "This is another sentence."])
    restored = pickle.loads(pickle.dumps(dataset))

    assert restored.sentences == dataset.sentences
    assert restored.noise_fn is DenoisingAutoEncoderDataset.delete


def test_custom_noise_fn() -> None:
    dataset = DenoisingAutoEncoderDataset(["a b c"], noise_fn=str.upper)
    assert dataset[0].texts == ["A B C", "a b c"]
