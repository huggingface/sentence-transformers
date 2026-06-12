from __future__ import annotations

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
    maxsim,
    maxsim_pairwise,
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


def test_maxsim_ragged_matrix_hand_computed() -> None:
    """MaxSim of ragged (variable-length) inputs, with a hand-computed (2, 2) score matrix."""
    queries = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # 2 tokens
        torch.tensor([[1.0, 1.0]]),  # 1 token
    ]
    documents = [
        torch.tensor([[1.0, 0.0], [0.0, 0.5]]),  # 2 tokens
        torch.tensor([[0.0, 1.0]]),  # 1 token
    ]
    # res[i][j] = sum over query tokens of the max similarity to any document token, e.g.
    # res[0][0] = max(1, 0) + max(0, 0.5) = 1.5; res[0][1] = max(0) + max(1) = 1.0
    expected = torch.tensor([[1.5, 1.0], [1.0, 1.0]])
    scores = maxsim(queries, documents)
    assert scores.shape == (2, 2)
    assert torch.allclose(scores, expected)


def test_maxsim_list_matches_masked_padded_tensor() -> None:
    """The ragged-list path auto-derives a length mask; it must match an explicitly padded + masked 3D tensor."""
    queries = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 1.0]])]
    documents = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0], [0.5, 0.5]])]
    from_list = maxsim(queries, documents)

    queries_padded = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
    documents_padded = torch.nn.utils.rnn.pad_sequence(documents, batch_first=True)
    query_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    document_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    from_padded = maxsim(queries_padded, documents_padded, a_mask=query_mask, b_mask=document_mask)

    assert torch.allclose(from_list, from_padded)


def test_maxsim_document_mask_excludes_padding_tokens() -> None:
    """A masked-out document token must not participate in the max, even with the highest similarity."""
    query = torch.tensor([[[1.0, 0.0]]])  # (1 document-query, 1 token, dim 2)
    document_real_only = torch.tensor([[[0.8, 0.6]]])  # similarity 0.8
    document_with_pad = torch.tensor([[[0.8, 0.6], [5.0, 0.0]]])  # 2nd token would otherwise dominate the max
    mask = torch.tensor([[1.0, 0.0]])

    base = maxsim(query, document_real_only)
    masked = maxsim(query, document_with_pad, b_mask=mask)
    assert torch.allclose(base, masked)
    # Without the mask the high-similarity padding token wins, inflating the score.
    assert maxsim(query, document_with_pad).item() > base.item()


def test_maxsim_negative_similarities_survive_padding() -> None:
    """A padding token (from a shorter document) must not inflate a negative MaxSim score to 0."""
    query = [torch.tensor([[1.0, 0.0]])]  # 1 query, 1 token
    documents = [
        torch.tensor([[-0.5, 0.0]]),  # 1 token: best (only) similarity -0.5
        torch.tensor([[-0.5, 0.0], [-0.3, 0.0]]),  # 2 tokens: best similarity -0.3
    ]
    # The first document is padded to length 2; the padding must be excluded from the max so the score
    # stays -0.5 instead of the 0 a zero-valued padding token would otherwise produce.
    scores = maxsim(query, documents)
    assert torch.allclose(scores, torch.tensor([[-0.5, -0.3]]))


def test_maxsim_query_mask_excludes_query_tokens() -> None:
    """MaxSim sums over query tokens; masking a query token drops its contribution from the sum."""
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # 2 query tokens
    document = torch.tensor([[[1.0, 0.0]]])  # token0 similarity 1, token1 similarity 0
    assert torch.allclose(maxsim(query, document), torch.tensor([[1.0]]))
    mask = torch.tensor([[0.0, 1.0]])  # keep only the second (zero-similarity) query token
    assert torch.allclose(maxsim(query, document, a_mask=mask), torch.tensor([[0.0]]))


def test_maxsim_pairwise_matches_maxsim_diagonal() -> None:
    """maxsim_pairwise(q, d) must equal the diagonal of the full maxsim(q, d) matrix, even with padding."""
    # randn gives mixed-sign similarities and the documents have different lengths, so the padded maxsim
    # path and the unpadded pairwise path only agree if masked padding tokens are excluded from the max.
    torch.manual_seed(0)
    queries = [torch.randn(3, 8), torch.randn(2, 8)]
    documents = [torch.randn(4, 8), torch.randn(6, 8)]
    full = maxsim(queries, documents)
    pairwise = maxsim_pairwise(queries, documents)
    assert pairwise.shape == (2,)
    assert torch.allclose(pairwise, torch.diagonal(full), atol=1e-5)


def test_maxsim_pairwise_tensor_path_matches_list_path() -> None:
    """maxsim_pairwise should give the same result for a 3D tensor and the equivalent list of 2D tensors."""
    queries = torch.rand(2, 3, 8)
    documents = torch.rand(2, 4, 8)
    from_tensor = maxsim_pairwise(queries, documents)
    from_list = maxsim_pairwise([queries[0], queries[1]], [documents[0], documents[1]])
    assert from_tensor.shape == (2,)
    assert torch.allclose(from_tensor, from_list, atol=1e-5)


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
