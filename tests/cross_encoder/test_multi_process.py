from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import CrossEncoder

# These tests fail if optimum.intel.openvino is imported, because openvinotoolkit/nncf
# patches torch._C._nn.gelu in a way that breaks pickling. As a result, we may have issues
# when running both backend tests and multi-process tests in the same session.


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", (False, True))
@pytest.mark.parametrize("apply_softmax", (False, True))
def test_predict_multi_process(
    reranker_bert_tiny_model: CrossEncoder, convert_to_tensor: bool, apply_softmax: bool
) -> None:
    model = reranker_bert_tiny_model
    pairs = [[f"This is sentence {i}", f"This is another sentence {i}"] for i in range(40)]

    # Start the multi-process pool on e.g. two CPU devices & compute the scores using the pool
    pool = model.start_multi_process_pool(["cpu", "cpu"])
    scores = model.predict(pairs, pool=pool, convert_to_tensor=convert_to_tensor, apply_softmax=apply_softmax)
    model.stop_multi_process_pool(pool)

    if convert_to_tensor:
        assert isinstance(scores, torch.Tensor)
        assert scores.shape[0] == len(pairs)
    else:
        assert isinstance(scores, np.ndarray)
        assert scores.shape[0] == len(pairs)

    # Make sure the scores aren't just all 0
    assert scores.sum() != 0.0

    # Compare against normal predictions
    scores_normal = model.predict(pairs, convert_to_tensor=convert_to_tensor, apply_softmax=apply_softmax)
    if convert_to_tensor:
        diff = torch.max(torch.abs(scores - scores_normal))
        assert diff < 1e-3
    else:
        diff = np.max(np.abs(scores - scores_normal))
        assert diff < 1e-3


@pytest.mark.slow
def test_multi_process_predict_same_as_standard_predict(reranker_bert_tiny_model: CrossEncoder):
    model = reranker_bert_tiny_model
    # Test that multi-process prediction gives the same result as standard prediction
    pairs = [
        ["First sentence.", "Second sentence."],
        ["Second sentence.", "Third sentence."],
        ["Third sentence.", "Fourth sentence."],
    ] * 5

    # Standard predict
    scores_standard = model.predict(pairs)

    # Multi-process predict with device=["cpu"] * 2
    scores_multi = model.predict(pairs, device=["cpu"] * 2)

    # Should produce the same scores
    assert np.allclose(scores_standard, scores_multi, atol=1e-6)


@pytest.mark.slow
def test_multi_process_pool(reranker_bert_tiny_model: CrossEncoder):
    # Test the start_multi_process_pool and stop_multi_process_pool functions
    model = reranker_bert_tiny_model
    pairs = [
        ["First sentence.", "Second sentence."],
        ["Second sentence.", "Third sentence."],
        ["Third sentence.", "Fourth sentence."],
    ] * 5

    # Standard predict
    scores_standard = model.predict(pairs)

    pool = model.start_multi_process_pool(["cpu"] * 2)
    try:
        # Predict using the pool
        scores_multi = model.predict(pairs, pool=pool)

    finally:
        model.stop_multi_process_pool(pool)

    # Should be numpy array with correct shape and the same scores
    assert isinstance(scores_multi, np.ndarray)
    assert scores_multi.shape == scores_standard.shape
    assert np.allclose(scores_standard, scores_multi, atol=1e-6)


@pytest.mark.slow
def test_multi_process_chunk_size(reranker_bert_tiny_model: CrossEncoder):
    # Test explicit chunk_size parameter for predict
    model = reranker_bert_tiny_model
    pairs = [
        ["First sentence.", "Second sentence."],
        ["Second sentence.", "Third sentence."],
        ["Third sentence.", "Fourth sentence."],
    ] * 10

    # Test with explicit chunk size
    scores = model.predict(pairs, device=["cpu"] * 2, chunk_size=5)

    # Should produce correct scores
    assert isinstance(scores, np.ndarray)
    assert scores.shape[0] == len(pairs)


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
def test_multi_process_with_empty_pairs(
    reranker_bert_tiny_model: CrossEncoder,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
):
    # Test predicting with empty pairs
    model = reranker_bert_tiny_model
    pairs: list[list[str]] = []

    # Predict with empty pairs
    scores_standard = model.predict(pairs, convert_to_tensor=convert_to_tensor, convert_to_numpy=convert_to_numpy)
    scores_multi = model.predict(
        pairs,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )

    # Should return empty arrays, identical types as without multi-processing
    assert type(scores_standard) is type(scores_multi)
    if convert_to_tensor:
        assert isinstance(scores_standard, torch.Tensor)
        assert scores_standard.numel() == 0
    elif convert_to_numpy:
        assert isinstance(scores_standard, np.ndarray)
        assert scores_standard.size == 0
    else:
        assert isinstance(scores_standard, list)
        assert len(scores_standard) == 0


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
def test_multi_process_with_single_pair(
    reranker_bert_tiny_model: CrossEncoder,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
):
    # Test predicting with a single pair
    model = reranker_bert_tiny_model
    pair = ["This is a single sentence.", "This is another sentence."]

    # Predict with single pair
    scores_standard = model.predict(pair, convert_to_tensor=convert_to_tensor, convert_to_numpy=convert_to_numpy)
    scores_multi = model.predict(
        pair,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )

    # Assert that the scores are the same type and shape
    assert type(scores_standard) is type(scores_multi)
    if isinstance(scores_standard, (np.ndarray, torch.Tensor)):
        assert scores_standard.shape == scores_multi.shape
    else:
        # Scalar outputs for num_labels=1
        assert np.allclose(scores_standard, scores_multi, atol=1e-6)


@pytest.mark.slow
def test_multi_process_more_workers_than_pairs(reranker_bert_tiny_model: CrossEncoder):
    # Test with more workers than pairs
    model = reranker_bert_tiny_model
    pairs = [["First sentence.", "Second sentence."], ["Second sentence.", "Third sentence."]]

    scores = model.predict(pairs, device=["cpu"] * 3)

    # Should be numpy array with correct shape
    assert isinstance(scores, np.ndarray)
    assert scores.shape[0] == len(pairs)


@pytest.mark.slow
def test_multi_process_with_large_chunk_size(reranker_bert_tiny_model: CrossEncoder):
    # Test with a large chunk size
    model = reranker_bert_tiny_model
    pairs = [["First sentence.", "Second sentence."]] * 20  # 20 pairs

    # Use a large chunk size
    scores = model.predict(pairs, device=["cpu"] * 2, chunk_size=30)

    # Should produce correct scores
    assert isinstance(scores, np.ndarray)
    assert scores.shape[0] == len(pairs)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available to experiment with 2 separate devices"
)
def test_multi_process_output_tensors_two_devices(reranker_bert_tiny_model: CrossEncoder):
    # Test with two separate devices
    model = reranker_bert_tiny_model
    pairs = [["First sentence.", "Second sentence."], ["Second sentence.", "Third sentence."]]

    # Ensure that scores are moved to CPU so they can be concatenated
    scores = model.predict(pairs, device=["cpu", "cuda"], convert_to_tensor=True)
    assert isinstance(scores, torch.Tensor)
    assert scores.device.type == "cpu"
    assert scores.shape[0] == len(pairs)

    # But the default is still just numpy
    scores = model.predict(pairs, device=["cpu", "cuda"])
    assert isinstance(scores, np.ndarray)
    assert scores.shape[0] == len(pairs)
