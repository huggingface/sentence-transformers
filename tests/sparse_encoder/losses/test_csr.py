from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sentence_transformers.sparse_encoder.losses.csr import normalized_mean_squared_error


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 2.0, 3.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[0.1, 0.1, 0.1]] * 7,
        [[0.0], [1e-23]],
    ],
    ids=["batch_size_one", "constant_batch", "constant_batch_rounding", "variance_underflow"],
)
@pytest.mark.parametrize("offset", [0.0, 0.5], ids=["perfect", "imperfect"])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 autocast"),
        ),
    ],
)
def test_normalized_mean_squared_error_zero_variance(
    values: list[list[float]], offset: float, device: str, dtype: torch.dtype
) -> None:
    original_input = torch.tensor(values, device=device, dtype=dtype, requires_grad=True)
    reconstruction = (original_input.detach() + offset).requires_grad_()

    with torch.autocast(device, dtype=dtype, enabled=dtype == torch.float16):
        loss = normalized_mean_squared_error(reconstruction, original_input)
        expected_loss = F.mse_loss(reconstruction, original_input)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected_loss)

    loss.backward()
    expected_gradients = torch.autograd.grad(expected_loss, (reconstruction, original_input))
    for tensor, expected_gradient in zip((reconstruction, original_input), expected_gradients):
        assert torch.isfinite(tensor.grad).all()
        torch.testing.assert_close(tensor.grad, expected_gradient)


@pytest.mark.parametrize("scale", [1.0, 1e-4], ids=["ordinary_variance", "small_variance"])
def test_normalized_mean_squared_error_nonzero_variance(scale: float) -> None:
    original_input = (torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]) * scale).requires_grad_()
    reconstruction = (original_input.detach() + 0.5 * scale).requires_grad_()

    loss = normalized_mean_squared_error(reconstruction, original_input)
    expected_loss = F.mse_loss(reconstruction, original_input) / F.mse_loss(
        original_input.mean(dim=0, keepdim=True).expand_as(original_input), original_input
    )

    torch.testing.assert_close(loss, expected_loss)

    loss.backward()
    expected_gradients = torch.autograd.grad(expected_loss, (reconstruction, original_input))
    for tensor, expected_gradient in zip((reconstruction, original_input), expected_gradients):
        torch.testing.assert_close(tensor.grad, expected_gradient)
