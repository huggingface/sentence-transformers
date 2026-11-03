from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers.sentence_transformer.evaluation import MSEEvaluator
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("teacher_value,student_value", [(0.0, 1e-4), (0.0, 30.0), (300.0, 0.0), (-60000.0, 60000.0)])
def test_mse_preserves_low_precision_errors(sparse, dtype, teacher_value, student_value):
    def make_model(value):
        if sparse:
            tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "sample": 2}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
            )
            model = SparseEncoder(
                modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, value]))], device="cpu"
            )
        else:
            weights = torch.tensor([[0.0], [value]])
            model = SentenceTransformer(
                modules=[WordEmbeddings(WhitespaceTokenizer(vocab=["pad", "sample"]), weights), Pooling(1)],
                device="cpu",
            )
        model.to(dtype=dtype)
        model.model_card_data.generate_widget_examples = False
        return model

    teacher, student = make_model(teacher_value), make_model(student_value)
    evaluator_cls = SparseMSEEvaluator if sparse else MSEEvaluator
    sentences = ["sample", "sample"]
    evaluator = evaluator_cls(sentences, sentences, teacher_model=teacher)
    original = evaluator.source_embeddings
    snapshot = original.clone() if isinstance(original, torch.Tensor) else original.copy()
    difference = float(torch.tensor(teacher_value, dtype=dtype)) - float(torch.tensor(student_value, dtype=dtype))
    expected = -100 * difference**2 / (3 if sparse else 1)

    metrics = evaluator(student)

    assert np.isfinite(metrics["negative_mse"])
    assert metrics["negative_mse"] == pytest.approx(expected, rel=1e-5, abs=1e-12)
    np.testing.assert_array_equal(
        original.float().numpy() if sparse else original, snapshot.float().numpy() if sparse else snapshot
    )


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


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
def test_mse_keeps_float64_differences(evaluator_cls):
    source = np.array([[1.0]], dtype=np.float64)
    target = np.array([[1.0 + 1e-9]], dtype=np.float64)
    expected = -100 * (target.item() - source.item()) ** 2
    if evaluator_cls is SparseMSEEvaluator:
        source, target = torch.from_numpy(source), torch.from_numpy(target)
    teacher, student = Mock(), Mock()
    teacher.encode.return_value, student.encode.return_value = source, target
    for model in (teacher, student):
        model.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.0}

    metrics = evaluator_cls(["source"], ["target"], teacher_model=teacher)(student)

    assert metrics["negative_mse"] == pytest.approx(expected, rel=1e-12, abs=0)
