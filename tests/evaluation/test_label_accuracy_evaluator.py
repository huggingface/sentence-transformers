"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.LabelAccuracyEvaluator import LabelAccuracyEvaluator
from sentence_transformers.losses import SoftmaxLoss
from sentence_transformers.readers import InputExample


@pytest.mark.skip(reason="This test is rather slow, and the LabelAccuracyEvaluator is not commonly used.")
def test_LabelAccuracyEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path) -> None:
    """Tests that the LabelAccuracyEvaluator can be loaded correctly"""
    model = paraphrase_distilroberta_base_v1_model

    max_dev_samples = 100
    nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(
        range(max_dev_samples)
    )

    dev_samples = []
    for row in nli_dataset:
        label_id = int(row["label"])

        dev_samples.append(
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=label_id,
            )
        )

    train_loss = SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )

    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=16)
    evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)
    metrics = evaluator(model, output_path=str(tmp_path))
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.2
