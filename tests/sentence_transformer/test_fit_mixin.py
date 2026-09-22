from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import SoftmaxLoss
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.util import is_datasets_available, is_training_available

if not is_training_available() or not is_datasets_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def test_fit_trains_loss_weights(stsb_bert_tiny_model: SentenceTransformer, tmp_path, monkeypatch) -> None:
    """fit() should also update the weights that live on a loss, e.g. the SoftmaxLoss classifier."""
    # fit() writes its default output directory relative to the working directory
    monkeypatch.chdir(tmp_path)
    model = stsb_bert_tiny_model
    loss = SoftmaxLoss(model, embedding_dimension=model.get_embedding_dimension(), num_labels=3)
    classifier_weight = loss.classifier.weight.detach().clone()

    train_examples = [
        InputExample(texts=[f"This is sentence {idx}", f"This is another sentence {idx}"], label=idx % 3)
        for idx in range(16)
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    model.fit(train_objectives=[(train_dataloader, loss)], epochs=1, warmup_steps=0, show_progress_bar=False)

    assert not torch.equal(loss.classifier.weight.detach(), classifier_weight)
