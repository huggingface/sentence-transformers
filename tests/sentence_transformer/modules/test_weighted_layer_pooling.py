from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentence_transformers.sentence_transformer.modules import WeightedLayerPooling


@pytest.mark.parametrize("safe_serialization", [False, True])
@pytest.mark.parametrize("as_parameter", [False, True])
def test_custom_layer_weights_round_trip(tmp_path: Path, safe_serialization: bool, as_parameter: bool) -> None:
    weights = torch.tensor([0.2, 0.8])
    if as_parameter:
        weights = torch.nn.Parameter(weights)
    model = WeightedLayerPooling(3, num_hidden_layers=1, layer_start=0, layer_weights=weights)
    hidden_states = (torch.zeros(2, 4, 3), torch.ones(2, 4, 3))
    expected = model({"all_layer_embeddings": hidden_states})["token_embeddings"]

    model.save(str(tmp_path), safe_serialization=safe_serialization)
    restored = WeightedLayerPooling.load(str(tmp_path))

    torch.testing.assert_close(restored.layer_weights, weights)
    torch.testing.assert_close(restored({"all_layer_embeddings": hidden_states})["token_embeddings"], expected)


def test_custom_tensor_layer_weights_are_trainable() -> None:
    model = WeightedLayerPooling(1, num_hidden_layers=1, layer_start=0, layer_weights=torch.tensor([0.2, 0.8]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_weights = model.layer_weights.detach().clone()

    output = model({"all_layer_embeddings": (torch.zeros(1, 1, 1), torch.ones(1, 1, 1))})["token_embeddings"]
    (output - 0.3).square().sum().backward()
    optimizer.step()

    assert model.layer_weights.grad is not None
    assert torch.isfinite(model.layer_weights.grad).all()
    assert not torch.equal(model.layer_weights, original_weights)


def test_custom_tensor_layer_weights_follow_module_dtype() -> None:
    model = WeightedLayerPooling(1, num_hidden_layers=1, layer_start=0, layer_weights=torch.tensor([0.2, 0.8]))

    model.double()

    assert model.layer_weights.dtype == torch.float64


@pytest.mark.parametrize("requires_grad", [False, True])
def test_custom_parameter_preserves_identity_and_requires_grad(requires_grad: bool) -> None:
    weights = torch.nn.Parameter(torch.tensor([0.2, 0.8]), requires_grad=requires_grad)

    model = WeightedLayerPooling(1, num_hidden_layers=1, layer_start=0, layer_weights=weights)

    assert model.layer_weights is weights
    assert model.layer_weights.requires_grad is requires_grad
