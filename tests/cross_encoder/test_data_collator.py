from __future__ import annotations

import pytest
import torch

from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)

from datasets import Dataset


@pytest.mark.parametrize("labels", [[1.0, 0.0, 1.0], [2, 0, 1]])
def test_tensor_formatted_scalar_labels(labels: list) -> None:
    collator = CrossEncoderDataCollator(preprocess_fn=lambda inputs, **kwargs: {})
    rows = {"query": ["q1", "q2", "q3"], "document": ["d1", "d2", "d3"], "label": labels}
    expected = collator(list(Dataset.from_dict(rows)))

    dataset = Dataset.from_dict(rows).with_format("torch", columns=["label"], output_all_columns=True)
    batch = collator(list(dataset))

    assert isinstance(batch["label"], torch.Tensor)
    torch.testing.assert_close(batch["label"], expected["label"])
    assert batch["query"] == ["q1", "q2", "q3"]
    assert batch["document"] == ["d1", "d2", "d3"]


def test_tensor_formatted_list_labels() -> None:
    collator = CrossEncoderDataCollator(preprocess_fn=lambda inputs, **kwargs: {})
    rows = {
        "query": ["q1", "q2"],
        "docs": [["d1", "d2", "d3"], ["d4", "d5"]],
        "labels": [[1, 0, 0], [0, 1]],
    }
    dataset = Dataset.from_dict(rows).with_format("torch", columns=["labels"], output_all_columns=True)
    batch = collator(list(dataset))

    assert isinstance(batch["label"], list)
    assert [label.tolist() for label in batch["label"]] == rows["labels"]
    assert batch["docs"] == rows["docs"]
