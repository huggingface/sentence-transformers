from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.evaluation import BaseEvaluator, SequentialEvaluator
from sentence_transformers.sentence_transformer.evaluation import EmbeddingExportEvaluator
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


@pytest.fixture
def model():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "cat": 1, "dog": 2, "query": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    embedding = StaticEmbedding(
        tokenizer, embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    )
    return SentenceTransformer(modules=[embedding], prompts={"query": "query "}, device="cpu")


def test_export_preserves_embeddings_and_unicode_text(model, tmp_path):
    sentences = ["cat", "dog", "日本語", "cat"]
    evaluator = EmbeddingExportEvaluator(sentences, name="dev", batch_size=2)
    assert evaluator(model, str(tmp_path / "nested"), epoch=0.5, steps=20) == {}
    with np.load(tmp_path / "nested/dev_embeddings_epoch_0.5_steps_20.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["embeddings"], model.encode(sentences))
        assert data["sentences"].tolist() == sentences


def test_export_forwards_encoding_options(model, tmp_path):
    kwargs = {"prompt_name": "query", "normalize_embeddings": True}
    evaluator = EmbeddingExportEvaluator(["cat", "dog"], encode_kwargs=kwargs)
    evaluator(model, str(tmp_path))
    with np.load(tmp_path / "embeddings_epoch_-1_steps_-1.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["embeddings"], model.encode(["cat", "dog"], **kwargs))


def test_export_keeps_checkpoints_and_overwrites_same_step(model, tmp_path):
    evaluator = EmbeddingExportEvaluator(["cat"])
    for step in [10, 20, 20]:
        evaluator(model, str(tmp_path), epoch=1, steps=step)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_missing_output_path_fails_before_encoding():
    model = Mock()
    with pytest.raises(ValueError, match="requires output_path"):
        EmbeddingExportEvaluator(["cat"])(model)
    model.encode.assert_not_called()


@pytest.mark.parametrize("name", ["../dev", "sub/dev", "sub\\dev", ".", ".."])
def test_invalid_name(name):
    with pytest.raises(ValueError, match="filename prefix"):
        EmbeddingExportEvaluator(["cat"], name=name)


def test_empty_sentences():
    with pytest.raises(ValueError, match="must not be empty"):
        EmbeddingExportEvaluator([])


@pytest.mark.parametrize(
    "key", ["convert_to_tensor", "convert_to_numpy", "output_value", "batch_size", "show_progress_bar"]
)
def test_conflicting_output_options(key):
    with pytest.raises(ValueError, match="cannot override"):
        EmbeddingExportEvaluator(["cat"], encode_kwargs={key: True})


class ScoreEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.primary_metric = "score"

    def __call__(self, *args, **kwargs):
        return {"score": 0.75}


@pytest.mark.parametrize("export_first", [True, False])
def test_sequential_export_does_not_change_selected_score(model, tmp_path, export_first):
    evaluators = [EmbeddingExportEvaluator(["cat"]), ScoreEvaluator()]
    if not export_first:
        evaluators.reverse()
    evaluator = SequentialEvaluator(evaluators)
    assert evaluator(model, str(tmp_path)) == {"score": 0.75, "sequential_score": 0.75}


def test_sequential_export_only(model, tmp_path):
    evaluator = SequentialEvaluator([EmbeddingExportEvaluator(["cat"])])
    assert evaluator(model, str(tmp_path)) == {"sequential_score": 0.0}


def test_nonzero_rank_participates_without_writing(monkeypatch):
    monkeypatch.setattr("sentence_transformers.sentence_transformer.evaluation.embedding_export.get_rank", lambda: 1)
    model = Mock()
    assert EmbeddingExportEvaluator(["cat"])(model, output_path=None) == {}
    model.encode.assert_called_once()


def test_export_during_training(model, tmp_path):
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    dataset = Dataset.from_dict({"anchor": ["cat", "dog"], "positive": ["cat dog", "dog cat"]})
    evaluator = SequentialEvaluator([EmbeddingExportEvaluator(["cat", "dog"]), ScoreEvaluator()])
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=2,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=dataset, loss=MultipleNegativesRankingLoss(model), evaluator=evaluator
    )
    trainer.train()
    archives = sorted((tmp_path / "eval").glob("*.npz"))
    assert len(archives) == 2
    with np.load(archives[-1], allow_pickle=False) as data:
        np.testing.assert_allclose(data["embeddings"], model.encode(["cat", "dog"]))
    assert any(log.get("eval_sequential_score") == 0.75 for log in trainer.state.log_history)
