from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sentence_transformers.sentence_transformer.evaluation import MSEEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize(("source_count", "target_count"), [(1, 2), (2, 1), (2, 3)])
def test_mse_rejects_unequal_sentence_counts(evaluator_cls, source_count, target_count):
    teacher = Mock()
    teacher.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}

    with pytest.raises(ValueError, match="source_sentences.*target_sentences.*same length") as exc_info:
        evaluator_cls(
            source_sentences=["source"] * source_count,
            target_sentences=["target"] * target_count,
            teacher_model=teacher,
        )

    assert str(exc_info.value) == (
        f"source_sentences ({source_count}) and target_sentences ({target_count}) must have the same length."
    )
    teacher.encode.assert_not_called()


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize("sentence_count", [1, 2])
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_mse_equal_sentence_counts_preserve_score(evaluator_cls, sentence_count, offset):
    source_embeddings = np.eye(2, dtype=np.float32)[:sentence_count]
    target_embeddings = source_embeddings + offset
    if evaluator_cls is SparseMSEEvaluator:
        source_embeddings = torch.from_numpy(source_embeddings)
        target_embeddings = torch.from_numpy(target_embeddings)

    teacher = Mock()
    teacher.encode.return_value = source_embeddings
    teacher.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}
    student = Mock()
    student.encode.return_value = target_embeddings
    student.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}
    sources = [f"source {index}" for index in range(sentence_count)]
    targets = [f"target {index}" for index in range(sentence_count)]

    evaluator = evaluator_cls(sources, targets, teacher_model=teacher, name="paired")
    metrics = evaluator(student)

    assert metrics["paired_negative_mse"] == pytest.approx(-100 * offset**2)
    assert evaluator.primary_metric == "paired_negative_mse"
    teacher.encode.assert_called_once()
    student.encode.assert_called_once()
    assert teacher.encode.call_args.args[0] == sources
    assert student.encode.call_args.args[0] == targets
