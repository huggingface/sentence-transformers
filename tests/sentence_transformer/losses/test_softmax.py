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


def test_probability_labels_match_cross_entropy(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    labels = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])

    actual = loss(features, labels)
    _, logits = loss(features, None)
    expected = torch.nn.functional.cross_entropy(logits, labels)

    torch.testing.assert_close(actual, expected)


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


def test_validated_probability_targets_match_cross_entropy(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3, validate_probability_targets=True)
    assert loss.get_config_dict()["validate_probability_targets"] is True
    _, logits = loss(features, None)

    for labels in (
        torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]),
        # One-hot float row, and a distribution that contains exact zeros.
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ):
        before = labels.clone()
        actual = loss(features, labels)
        assert torch.equal(labels, before)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected)


def test_validated_low_precision_probability_targets_match_cross_entropy(
    model: SentenceTransformer,
) -> None:
    one_row = [model.preprocess(["red"]), model.preprocess(["green"])]
    specs = [
        (3, torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.float16), 0.999755859375),
        (3, torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.bfloat16), 1.001953125),
        (7, torch.tensor([[1 / 7] * 7], dtype=torch.bfloat16), 0.998046875),
    ]
    for num_labels, labels, expected_sum in specs:
        assert labels.sum(dtype=torch.float64).item() == expected_sum
        loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=num_labels, validate_probability_targets=True)
        before = labels.clone()
        actual = loss(one_row, labels)
        assert torch.equal(labels, before)
        _, logits = loss(one_row, None)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected)


def test_validate_probability_targets_rejects_invalid_rows(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3, validate_probability_targets=True)
    one_row = [model.preprocess(["red"]), model.preprocess(["blue"])]
    bfloat16_row = torch.tensor([[0.5, 0.50390625, 0.0]], dtype=torch.bfloat16)
    assert bfloat16_row.sum(dtype=torch.float64).item() == 1.00390625

    cases = [
        (torch.tensor([[0.2, 0.3, 0.8], [0.6, 0.1, 0.3]]), features, r"row 0.*row-sum constraint"),
        (bfloat16_row, one_row, r"row 0.*row-sum constraint"),
        (torch.tensor([[0.2, 0.3, 0.5], [-0.1, 0.5, 0.6]]), features, r"row 1.*non-negative constraint"),
        (torch.tensor([[float("nan"), 0.0, 0.0], [-0.1, 0.5, 0.6]]), features, r"row 0.*finite constraint"),
        (torch.tensor([[float("inf"), 0.0, 0.0], [-0.1, 0.5, 0.6]]), features, r"row 0.*finite constraint"),
    ]
    for labels, batch_features, message in cases:
        before = labels.clone()
        with pytest.raises(ValueError, match=message):
            loss(batch_features, labels)
        torch.testing.assert_close(labels, before, equal_nan=True)


def test_unchecked_invalid_probability_targets_match_cross_entropy(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    assert loss.get_config_dict()["validate_probability_targets"] is False
    # A row summing to 1.3 and a row containing -0.1 stay unchecked when the flag is omitted.
    labels = torch.tensor([[0.2, 0.3, 0.8], [-0.1, 0.5, 0.6]])
    before = labels.clone()
    actual = loss(features, labels)
    assert torch.equal(labels, before)
    assert torch.isfinite(actual)
    _, logits = loss(features, None)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_validation_flag_matches_default_for_non_probability_labels(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    validated = SoftmaxLoss(model, embedding_dimension=2, num_labels=3, validate_probability_targets=True)
    default = SoftmaxLoss(model, embedding_dimension=2, num_labels=3)
    default.classifier.load_state_dict(validated.classifier.state_dict())

    class_indices = torch.tensor([0, 2])
    torch.testing.assert_close(validated(features, class_indices), default(features, class_indices))
    column_indices = class_indices.unsqueeze(-1)
    torch.testing.assert_close(validated(features, column_indices), default(features, column_indices))

    def regression_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(logits[:, 0], labels)

    for num_labels in (1, 3):
        validated_scalar = SoftmaxLoss(
            model,
            embedding_dimension=2,
            num_labels=num_labels,
            loss_fct=regression_loss,
            validate_probability_targets=True,
        )
        default_scalar = SoftmaxLoss(
            model,
            embedding_dimension=2,
            num_labels=num_labels,
            loss_fct=regression_loss,
        )
        default_scalar.classifier.load_state_dict(validated_scalar.classifier.state_dict())
        vector_labels = torch.tensor([0.2, 0.8])
        column_labels = torch.tensor([[0.2], [0.8]])
        torch.testing.assert_close(validated_scalar(features, vector_labels), default_scalar(features, vector_labels))
        torch.testing.assert_close(validated_scalar(features, column_labels), default_scalar(features, column_labels))
        torch.testing.assert_close(
            validated_scalar(features, column_labels), validated_scalar(features, vector_labels)
        )


def test_validate_probability_targets_accepts_empty_batch(
    model: SentenceTransformer, features: list[dict[str, torch.Tensor]]
) -> None:
    loss = SoftmaxLoss(model, embedding_dimension=2, num_labels=3, validate_probability_targets=True)
    empty_features = [
        {key: value[:0] if isinstance(value, torch.Tensor) else value for key, value in column.items()}
        for column in features
    ]
    labels = torch.empty((0, 3), dtype=torch.float32)
    before = labels.clone()
    actual = loss(empty_features, labels)
    assert torch.equal(labels, before)
    _, logits = loss(empty_features, None)
    assert logits.shape == labels.shape
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(actual, expected, equal_nan=True)
