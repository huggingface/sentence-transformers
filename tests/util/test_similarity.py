from __future__ import annotations

import time

import numpy as np
import pytest
import sklearn
import torch

from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.util.similarity import (
    cos_sim,
    dot_score,
    euclidean_sim,
    manhattan_sim,
    pairwise_angle_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
    pytorch_cos_sim,
)


def test_pytorch_cos_sim() -> None:
    """Tests the correct computation of pytorch_cos_sim"""
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    sklearn_pairwise = sklearn.metrics.pairwise.cosine_similarity(a, b)
    pytorch_cos_scores = pytorch_cos_sim(a, b).numpy()
    for i in range(len(sklearn_pairwise)):
        for j in range(len(sklearn_pairwise[i])):
            assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001


def test_pairwise_cos_sim() -> None:
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    # Pairwise cos
    sklearn_pairwise = 1 - sklearn.metrics.pairwise.paired_cosine_distances(a, b)
    pytorch_cos_scores = pairwise_cos_sim(a, b).numpy()

    assert np.allclose(sklearn_pairwise, pytorch_cos_scores)


def test_pairwise_euclidean_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    euclidean_expected = np.array([-1.0, -np.sqrt(2.0)])
    euclidean_calculated = pairwise_euclidean_sim(a, b).numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_pairwise_manhattan_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    manhattan_expected = np.array([-1.0, -2.0])
    manhattan_calculated = pairwise_manhattan_sim(a, b).numpy()

    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_pairwise_dot_score_cos_sim() -> None:
    a = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([1.0, 0.0, -1.0])
    cosine_calculated = pairwise_cos_sim(a, b)
    dot_calculated = pairwise_dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)


def test_euclidean_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    euclidean_expected = np.array([[-1.0, -np.sqrt(2.0)], [-1.0, 0.0]])
    euclidean_calculated = euclidean_sim(a, b).detach().numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_manhattan_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    manhattan_expected = np.array([[-1.0, -2.0], [-1.0, 0]])
    manhattan_calculated = manhattan_sim(a, b).detach().numpy()
    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_dot_score_cos_sim() -> None:
    a = np.array([[1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([[1.0, 0.0, -1.0]])
    cosine_calculated = cos_sim(a, b)
    dot_calculated = dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)


def create_sparse_tensor(rows, cols, num_nonzero, seed=None):
    """Create a sparse tensor of shape (rows, cols) with num_nonzero values per row."""
    if seed is not None:
        torch.manual_seed(seed)

    indices = []
    values = []

    for i in range(rows):
        row_indices = torch.stack(
            [torch.full((num_nonzero,), i, dtype=torch.long), torch.randint(0, cols, (num_nonzero,))]
        )
        row_values = torch.randn(num_nonzero)

        indices.append(row_indices)
        values.append(row_values)

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    return torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce()


@pytest.fixture
def sparse_tensors():
    """Create two large sparse tensors of shape (50, 100) each."""
    rows, cols = 50, 1000
    num_nonzero = 10  # per row

    tensor1 = create_sparse_tensor(rows, cols, num_nonzero, seed=42)
    tensor2 = create_sparse_tensor(rows, cols, num_nonzero, seed=1337)
    if torch.cuda.is_available():
        return tensor1.to("cuda"), tensor2.to("cuda")
    else:
        return tensor1, tensor2


def test_cos_sim_sparse(sparse_tensors):
    """Test cosine similarity between sparse and dense representations."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = cos_sim(tensor1, tensor2)
    sim_dense = cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_dot_score_sparse(sparse_tensors):
    """Test dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    # Convert to dense before computing
    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = dot_score(tensor1, tensor2)
    score_dense = dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_manhattan_sim_sparse(sparse_tensors):
    """Test Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = manhattan_sim(tensor1, tensor2)
    sim_dense = manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_euclidean_sim_sparse(sparse_tensors):
    """Test Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = euclidean_sim(tensor1, tensor2)
    sim_dense = euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_cos_sim_sparse(sparse_tensors):
    """Test pairwise cosine similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_cos_sim(tensor1, tensor2)
    sim_dense = pairwise_cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_dot_score_sparse(sparse_tensors):
    """Test pairwise dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = pairwise_dot_score(tensor1, tensor2)
    score_dense = pairwise_dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_manhattan_sim_sparse(sparse_tensors):
    """Test pairwise Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_manhattan_sim(tensor1, tensor2)
    sim_dense = pairwise_manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_euclidean_sim_sparse(sparse_tensors):
    """Test pairwise Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_euclidean_sim(tensor1, tensor2)
    sim_dense = pairwise_euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_performance_with_large_vectors():
    """Test performance (time) for all similarity functions with large sparse vectors vs dense."""

    # Set dimensions for large sparse vectors
    rows = 500  # Just a few vectors to compare
    cols = 100000  # Large dimensionality
    num_nonzero = 128  # 128 non-null elements per vector

    print("\nPerformance test with large sparse vs. dense vectors")
    print(f"Shape: ({rows}, {cols}), Non-zeros per vector: {num_nonzero}")

    # Create large sparse tensors
    print("Creating sparse tensors...")
    tensor1_sparse = create_sparse_tensor(rows, cols, num_nonzero, seed=42)
    tensor2_sparse = create_sparse_tensor(rows, cols, num_nonzero, seed=1337)

    # Convert to dense for comparison
    print("Converting to dense tensors...")
    tensor1_dense = tensor1_sparse.to_dense()
    tensor2_dense = tensor2_sparse.to_dense()

    # List of functions to test
    similarity_functions = [
        ("cos_sim", cos_sim),
        ("dot_score", dot_score),
        ("manhattan_sim", manhattan_sim),  # Comment until the function is implemented in a fast way
        ("euclidean_sim", euclidean_sim),
        ("pairwise_cos_sim", pairwise_cos_sim),
        ("pairwise_dot_score", pairwise_dot_score),
        ("pairwise_manhattan_sim", pairwise_manhattan_sim),
        ("pairwise_euclidean_sim", pairwise_euclidean_sim),
    ]

    results = []

    for name, func in similarity_functions:
        # Time sparse operation
        start_time = time.time()
        _ = func(tensor1_sparse, tensor2_sparse)
        sparse_time = time.time() - start_time

        # Time dense operation
        start_time = time.time()
        _ = func(tensor1_dense, tensor2_dense)
        dense_time = time.time() - start_time

        # Calculate speedup ratio
        speedup_ratio = dense_time / sparse_time if sparse_time > 0 else float("inf")

        results.append(
            {"function": name, "sparse_time": sparse_time, "dense_time": dense_time, "speedup_ratio": speedup_ratio}
        )

    # Print results in a table
    print("\nPerformance Results:")
    print(f"{'Function':<25} | {'Sparse Time (s)':<15} | {'Dense Time (s)':<15} | {'Speedup Ratio':<15}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['function']:<25} | {r['sparse_time']:<15.6f} | {r['dense_time']:<15.6f} | {r['speedup_ratio']:<15.2f}"
        )

    # Create performance summary
    sparse_time_avg = np.mean([r["sparse_time"] for r in results])
    dense_time_avg = np.mean([r["dense_time"] for r in results])
    avg_speedup = np.mean([r["speedup_ratio"] for r in results])

    print("\nAverage Performance:")
    print(f"Time - Sparse: {sparse_time_avg:.6f}s")
    print(f"Time - Dense: {dense_time_avg:.6f}s")
    print(f"Average speedup: {avg_speedup:.2f}x")

    assert sparse_time_avg < 0.1, "Sparse operations took too long!"


def test_pairwise_angle_sim_even_and_odd_sparse_embeddings(splade_bert_tiny_model: SparseEncoder) -> None:
    """Ensure pairwise_angle_sim works for even and artificially odd dims."""

    model = splade_bert_tiny_model
    sentences = [
        "The weather is nice today.",
        "It's sunny outside.",
        "I love going for walks in the park.",
        "Let's have a picnic this weekend.",
    ]
    embeddings_even_dense = model.encode(
        sentences,
        convert_to_tensor=True,
        convert_to_sparse_tensor=False,
    )
    assert embeddings_even_dense.shape[1] % 2 == 0, "Test setup error: expected even-dimensional sparse embeddings."

    # Baseline with the model's normal (even-dimensional) sparse embeddings
    sim_even = pairwise_angle_sim(embeddings_even_dense[:2].to_sparse(), embeddings_even_dense[2:].to_sparse())

    # Convert to dense, append a trailing zero column to make the embedding
    # dimension odd while keeping the representation identical, then convert
    # back to sparse so we continue testing the sparse input path.
    embeddings_odd = torch.nn.functional.pad(embeddings_even_dense, (0, 1), mode="constant", value=0)
    assert embeddings_odd.shape[1] % 2 == 1, "Test setup error: expected odd-dimensional sparse embeddings."

    sim_odd = pairwise_angle_sim(embeddings_odd[:2].to_sparse(), embeddings_odd[2:].to_sparse())

    assert sim_even.shape == sim_odd.shape
    assert torch.allclose(sim_even, sim_odd, rtol=1e-5, atol=1e-5)
