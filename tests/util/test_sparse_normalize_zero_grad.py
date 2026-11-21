from __future__ import annotations

import pytest
import torch

from sentence_transformers.util import normalize_embeddings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duplicate_entries", [False, True])
def test_sparse_normalization_zero_rows_have_finite_gradients(dtype, duplicate_entries):
    if duplicate_entries:
        indices = torch.tensor([[0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1]])
        data = [2.0, -2.0, 3.0, -3.0, 3.0, 4.0]
        expected_grad = [1.0, 1.0, 1.0, 1.0, 0.032, -0.024]
    else:
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        data = [0.0, 0.0, 3.0, 4.0]
        expected_grad = [1.0, 1.0, 0.032, -0.024]
    values = torch.tensor(data, dtype=dtype, requires_grad=True)
    embeddings = torch.sparse_coo_tensor(indices, values, (3, 2))

    result = normalize_embeddings(embeddings)

    assert result.is_sparse
    assert result.dtype == dtype
    torch.testing.assert_close(result.to_dense(), torch.tensor([[0.0, 0.0], [0.6, 0.8], [0.0, 0.0]], dtype=dtype))
    result.to_dense().sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor(expected_grad, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_normalization_nonzero_rows_match_dense_gradients(dtype):
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    values = torch.tensor([1.0, 2.0, -3.0, 4.0], dtype=dtype, requires_grad=True)
    dense = values.detach().clone().reshape(2, 2).requires_grad_()
    weights = torch.tensor([[2.0, -1.0], [3.0, 5.0]], dtype=dtype)
    sparse = torch.sparse_coo_tensor(indices, values, (2, 2))

    result = normalize_embeddings(sparse).to_dense()
    expected = torch.nn.functional.normalize(dense, dim=1)
    torch.testing.assert_close(result, expected)
    (result * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(values.grad.reshape(2, 2), dense.grad)
