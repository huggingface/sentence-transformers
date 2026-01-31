"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
)


@pytest.mark.skip(reason="This test is rather slow, and the LabelAccuracyEvaluator is not commonly used.")
def test_LabelAccuracyEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path) -> None:
    """Tests that the LabelAccuracyEvaluator can be loaded correctly"""
    model = paraphrase_distilroberta_base_v1_model

    max_dev_samples = 100
    nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(
        range(max_dev_samples)
    )

    # HF dataset label IDs differ from legacy label IDs used by the classifier head,
    # so we remap HF labels to legacy IDs explicitly.
    hf_int2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

    def hf_label_to_legacy(hf_label: int) -> int:
        return label2int[hf_int2label[hf_label]]

    dev_samples = []
    for row in nli_dataset["train"]:
        label_id = hf_label_to_legacy(row["label"])

        dev_samples.append(
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=label_id,
            )
        )

    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label2int),
    )

    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=16)
    evaluator = evaluation.LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)
    metrics = evaluator(model, output_path=str(tmp_path))
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.2
