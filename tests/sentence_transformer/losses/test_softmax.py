from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import SoftmaxLoss
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util import is_training_available


@pytest.fixture
def model() -> SentenceTransformer:
    embeddings = WordEmbeddings(
        WhitespaceTokenizer(vocab=["PAD", "red", "blue", "green"]),
        embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        update_embeddings=True,
    )
    return SentenceTransformer(modules=[embeddings, Pooling(2, "mean")], device="cpu")


@pytest.fixture
def features(model: SentenceTransformer) -> list[dict[str, torch.Tensor]]:
    return [model.preprocess(["red", "blue"]), model.preprocess(["green", "red blue"])]


def test_probability_labels_match_cross_entropy_and_gradients(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    labels = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])

    actual = loss(features, labels)
    _, logits = loss(features, None)
    expected = -(labels * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

    torch.testing.assert_close(actual, expected)
    parameters = tuple(loss.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("column_labels", [False, True])
def test_one_hot_labels_match_class_indices(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]], column_labels: bool
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    class_indices = torch.tensor([0, 2])
    one_hot_labels = torch.nn.functional.one_hot(class_indices, num_classes=3).float()
    if column_labels:
        class_indices = class_indices.unsqueeze(-1)

    probability_loss = loss(features, one_hot_labels)
    class_index_loss = loss(features, class_indices)

    torch.testing.assert_close(probability_loss, class_index_loss)
    parameters = tuple(loss.parameters())
    probability_gradients = torch.autograd.grad(probability_loss, parameters)
    class_index_gradients = torch.autograd.grad(class_index_loss, parameters)
    for probability_gradient, class_index_gradient in zip(probability_gradients, class_index_gradients):
        torch.testing.assert_close(probability_gradient, class_index_gradient)


@pytest.mark.parametrize("num_labels", [1, 3])
def test_custom_loss_keeps_flattened_scalar_labels(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]], num_labels: int
) -> None:
    def regression_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(logits[:, 0], labels)

    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=num_labels, loss_fct=regression_loss)

    vector_loss = loss(features, torch.tensor([0.2, 0.8]))
    column_loss = loss(features, torch.tensor([[0.2], [0.8]]))

    torch.testing.assert_close(column_loss, vector_loss)


@pytest.mark.skipif(not is_training_available(), reason="Training requires datasets and accelerate")
def test_trainer_accepts_probability_labels(model: SentenceTransformer, tmp_path) -> None:
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    initial_weights = loss.classifier.weight.detach().clone()
    dataset = Dataset.from_dict(
        {
            "sentence1": ["red", "blue"],
            "sentence2": ["green", "red blue"],
            "label": [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]],
        }
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        loss=loss,
        train_dataset=dataset,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=2,
            use_cpu=True,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
        ),
    )

    result = trainer.train()

    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(loss.classifier.weight.detach(), initial_weights)
