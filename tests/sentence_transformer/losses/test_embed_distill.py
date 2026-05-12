from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import EmbedDistillLoss, MSELoss


def _features(model: SentenceTransformer, texts: list[str]) -> dict:
    feats = model.preprocess(texts)
    return {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}


def test_invalid_distance_metric_raises(stsb_bert_tiny_model: SentenceTransformer) -> None:
    with pytest.raises(ValueError, match="distance_metric must be one of"):
        EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="kl_div")  # type: ignore[arg-type]


def test_missing_labels_raises(stsb_bert_tiny_model: SentenceTransformer) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="cosine")
    with pytest.raises(ValueError, match="pre-computed teacher embeddings"):
        loss([_features(stsb_bert_tiny_model, ["a", "b"])], labels=None)


@pytest.mark.parametrize("distance_metric", ["cosine", "mse", "l2"])
def test_forward_produces_finite_scalar(stsb_bert_tiny_model: SentenceTransformer, distance_metric: str) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric=distance_metric)
    sentence_features = [_features(stsb_bert_tiny_model, ["a", "b"])]
    labels = torch.randn(2, stsb_bert_tiny_model.get_embedding_dimension(), device=stsb_bert_tiny_model.device)
    value = loss(sentence_features, labels)
    assert value.dim() == 0
    assert torch.isfinite(value)


@pytest.mark.parametrize("distance_metric", ["cosine", "mse", "l2"])
def test_distance_metric_matches_reference(stsb_bert_tiny_model: SentenceTransformer, distance_metric: str) -> None:
    """Per-metric, the loss should match a hand-computed reference on the same embeddings."""
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric=distance_metric)

    student = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    teacher = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2]])

    actual = loss.compute_loss_from_embeddings([student], [teacher])

    if distance_metric == "mse":
        expected = nn.functional.mse_loss(student, teacher)
    elif distance_metric == "l2":
        expected = torch.norm(student - teacher, dim=-1).mean()
    else:  # cosine
        expected = (1 - nn.functional.cosine_similarity(student, teacher, dim=-1)).mean()

    assert torch.allclose(actual, expected)


def test_2d_labels_broadcast_across_columns(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """2D labels with multi-column input = same target broadcast (multilingual-distillation case)."""
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="mse")
    sentence_features = [
        _features(stsb_bert_tiny_model, ["a", "b"]),
        _features(stsb_bert_tiny_model, ["c", "d"]),
    ]
    labels_2d = torch.randn(2, stsb_bert_tiny_model.get_embedding_dimension(), device=stsb_bert_tiny_model.device)
    value = loss(sentence_features, labels_2d)
    assert torch.isfinite(value)


def test_3d_labels_per_column(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """3D labels supply a different teacher embedding per text column."""
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="mse")
    sentence_features = [
        _features(stsb_bert_tiny_model, ["a", "b"]),
        _features(stsb_bert_tiny_model, ["c", "d"]),
    ]
    dim = stsb_bert_tiny_model.get_embedding_dimension()
    labels_3d = torch.randn(2, 2, dim, device=stsb_bert_tiny_model.device)
    value = loss(sentence_features, labels_3d)
    assert torch.isfinite(value)


def test_3d_labels_column_count_mismatch_raises(stsb_bert_tiny_model: SentenceTransformer) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="mse")
    sentence_features = [_features(stsb_bert_tiny_model, ["a", "b"])]
    dim = stsb_bert_tiny_model.get_embedding_dimension()
    labels_3d_wrong = torch.randn(2, 3, dim, device=stsb_bert_tiny_model.device)
    with pytest.raises(ValueError, match="Number of label columns"):
        loss(sentence_features, labels_3d_wrong)


def test_projection_layer_changes_during_training(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The projection layer should receive gradients and update its weights."""
    student_dim = stsb_bert_tiny_model.get_embedding_dimension()
    teacher_dim = student_dim + 16
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="cosine", projection_dim=teacher_dim)
    loss = loss.to(stsb_bert_tiny_model.device)  # the trainer normally handles this
    assert loss.projection is not None

    optimizer = torch.optim.SGD(loss.projection.parameters(), lr=1.0)
    initial_weight = loss.projection.weight.detach().clone()

    sentence_features = [_features(stsb_bert_tiny_model, ["a", "b"])]
    labels = torch.randn(2, teacher_dim, device=stsb_bert_tiny_model.device)

    optimizer.zero_grad()
    value = loss(sentence_features, labels)
    value.backward()
    optimizer.step()

    assert not torch.allclose(initial_weight, loss.projection.weight)


def test_projection_dim_matches_teacher(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """projection_dim should produce a Linear that outputs teacher_dim."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() * 2
    loss = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    assert loss.projection is not None
    assert loss.projection.out_features == teacher_dim
    assert loss.projection.in_features == stsb_bert_tiny_model.get_embedding_dimension()


def test_no_projection_by_default(stsb_bert_tiny_model: SentenceTransformer) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model)
    assert loss.projection is None
    assert loss.projection_dim is None


def test_get_config_dict(stsb_bert_tiny_model: SentenceTransformer) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="l2", projection_dim=128)
    assert loss.get_config_dict() == {"distance_metric": "l2", "projection_dim": 128}


def test_mseloss_subclass_matches_embed_distill_mse(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """`MSELoss(model)` should produce the same value as `EmbedDistillLoss(model, distance_metric='mse')`."""
    mse_loss = MSELoss(stsb_bert_tiny_model)
    embed_loss = EmbedDistillLoss(stsb_bert_tiny_model, distance_metric="mse")

    student = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    teacher = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2]])

    assert torch.allclose(
        mse_loss.compute_loss_from_embeddings([student], [teacher]),
        embed_loss.compute_loss_from_embeddings([student], [teacher]),
    )


def test_mseloss_is_subclass_of_embed_distill() -> None:
    assert issubclass(MSELoss, EmbedDistillLoss)


def test_mseloss_accepts_projection_dim(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The upgraded MSELoss should accept a projection_dim for cross-dim distillation."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 8
    loss = MSELoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    assert loss.projection is not None
    assert loss.projection.out_features == teacher_dim
    assert loss.distance_metric == "mse"


def test_mseloss_get_config_dict_exposes_projection_dim(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """MSELoss's config should expose `projection_dim` (the only knob it has) so model
    cards reflect cross-dim distillation runs."""
    loss = MSELoss(stsb_bert_tiny_model, projection_dim=64)
    assert loss.get_config_dict() == {"projection_dim": 64}


def test_save_projection_without_projection_raises(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    loss = EmbedDistillLoss(stsb_bert_tiny_model)
    with pytest.raises(ValueError, match="No projection layer to save"):
        loss.save_projection(tmp_path / "proj.safetensors")


def test_save_load_projection_round_trip(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    """save_projection -> load_projection should preserve weights bit-for-bit."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 16
    src = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    assert src.projection is not None

    # Mutate weights so we know we're not just matching a fresh random init.
    with torch.no_grad():
        src.projection.weight.add_(0.1)
        src.projection.bias.add_(0.2)

    path = tmp_path / "proj.safetensors"
    src.save_projection(path)

    dst = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    dst.load_projection(path)

    assert torch.equal(src.projection.weight, dst.projection.weight)
    assert torch.equal(src.projection.bias, dst.projection.bias)


def test_load_projection_without_projection_dim(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    """`load_projection` should construct the layer from the saved shape when none was configured."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 8
    src = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    with torch.no_grad():
        src.projection.weight.fill_(0.5)
        src.projection.bias.fill_(0.25)

    path = tmp_path / "proj.safetensors"
    src.save_projection(path)

    dst = EmbedDistillLoss(stsb_bert_tiny_model)
    assert dst.projection is None
    dst.load_projection(path)

    assert dst.projection is not None
    assert dst.projection_dim == teacher_dim
    assert torch.equal(src.projection.weight, dst.projection.weight)
    assert torch.equal(src.projection.bias, dst.projection.bias)


def test_load_projection_matches_model_device(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    """Lazy-constructed projection should land on the model's device so that
    `load_projection` works when called after the loss has been moved."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 8
    src = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    path = tmp_path / "proj.safetensors"
    src.save_projection(path)

    dst = EmbedDistillLoss(stsb_bert_tiny_model)
    dst.load_projection(path)

    assert dst.projection.weight.device == stsb_bert_tiny_model.device


def test_load_projection_dim_mismatch_raises(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    """Loading into a loss whose projection has a different shape should raise."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 16
    src = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    path = tmp_path / "proj.safetensors"
    src.save_projection(path)

    dst = EmbedDistillLoss(stsb_bert_tiny_model, projection_dim=teacher_dim + 1)
    with pytest.raises(ValueError, match="does not match the configured"):
        dst.load_projection(path)


def test_mseloss_save_load_projection(stsb_bert_tiny_model: SentenceTransformer, tmp_path) -> None:
    """MSELoss should inherit `save_projection` / `load_projection` from the parent."""
    teacher_dim = stsb_bert_tiny_model.get_embedding_dimension() + 8
    src = MSELoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    with torch.no_grad():
        src.projection.weight.fill_(0.3)

    path = tmp_path / "proj.safetensors"
    src.save_projection(path)

    dst = MSELoss(stsb_bert_tiny_model, projection_dim=teacher_dim)
    dst.load_projection(path)
    assert torch.equal(src.projection.weight, dst.projection.weight)
