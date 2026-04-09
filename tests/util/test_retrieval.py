from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util.retrieval import community_detection, paraphrase_mining, semantic_search
from sentence_transformers.util.similarity import pytorch_cos_sim


def test_semantic_search() -> None:
    """Tests util.semantic_search function"""
    num_queries = 20
    num_k = 10

    doc_emb = torch.tensor(np.random.randn(1000, 100))
    q_emb = torch.tensor(np.random.randn(num_queries, 100))
    hits = semantic_search(q_emb, doc_emb, top_k=num_k, query_chunk_size=5, corpus_chunk_size=17)
    assert len(hits) == num_queries
    assert len(hits[0]) == num_k

    # Sanity Check of the results
    cos_scores = pytorch_cos_sim(q_emb, doc_emb)
    cos_scores_values, cos_scores_idx = cos_scores.topk(num_k)
    cos_scores_values = cos_scores_values.cpu().tolist()
    cos_scores_idx = cos_scores_idx.cpu().tolist()

    for qid in range(num_queries):
        for hit_num in range(num_k):
            assert hits[qid][hit_num]["corpus_id"] == cos_scores_idx[qid][hit_num]
            assert np.abs(hits[qid][hit_num]["score"] - cos_scores_values[qid][hit_num]) < 0.001


def test_paraphrase_mining() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = [
        "This is a test",
        "This is a test!",
        "The cat sits on mat",
        "The cat sits on the mat",
        "On the mat a cat sits",
        "A man eats pasta",
        "A woman eats pasta",
        "A man eats spaghetti",
    ]
    duplicates = paraphrase_mining(model, sentences)

    for score, a, b in duplicates:
        if score > 0.5:
            assert (a, b) in [(0, 1), (2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 7)]


def test_community_detection_two_clear_communities():
    """Test case with two clear communities."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Point 0
            [0.9, 0.1, 0.0],  # Point 1
            [0.8, 0.2, 0.0],  # Point 2
            [0.1, 0.9, 0.0],  # Point 3
            [0.0, 1.0, 0.0],  # Point 4
            [0.2, 0.8, 0.0],  # Point 5
        ]
    )
    expected = [
        [0, 1, 2],  # Community 1
        [3, 4, 5],  # Community 2
    ]
    result = community_detection(embeddings, threshold=0.8, min_community_size=2)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])


def test_community_detection_no_communities_high_threshold():
    """Test case where no communities are found due to a high threshold."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = []
    result = community_detection(embeddings, threshold=0.99, min_community_size=2)
    assert result == expected


def test_community_detection_all_points_in_one_community():
    """Test case where all points form a single community due to a low threshold."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
    )
    expected = [
        [0, 1, 2],  # Single community
    ]
    result = community_detection(embeddings, threshold=0.5, min_community_size=2)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])


def test_community_detection_min_community_size_filtering():
    """Test case where communities are filtered based on minimum size."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    expected = [
        [0, 1, 2],  # Only one community meets the min size requirement
    ]
    result = community_detection(embeddings, threshold=0.8, min_community_size=3)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])


def test_community_detection_overlapping_communities():
    """Test case with overlapping communities (resolved by the function)."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Point 0
            [0.9, 0.1, 0.0],  # Point 1
            [0.8, 0.2, 0.0],  # Point 2
            [0.7, 0.3, 0.0],  # Point 3 (overlaps with both communities)
            [0.1, 0.9, 0.0],  # Point 4
            [0.0, 1.0, 0.0],  # Point 5
        ]
    )
    expected = [
        [0, 1, 2, 3],  # Community 1 (includes overlapping point 3)
        [4, 5],  # Community 2
    ]
    result = community_detection(embeddings, threshold=0.8, min_community_size=2)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])


def test_community_detection_numpy_input():
    """Test case where input is a numpy array instead of a torch tensor."""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
    )
    expected = [
        [0, 1, 2],  # Single community
    ]
    result = community_detection(embeddings, threshold=0.8, min_community_size=2)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])


def test_community_detection_large_batch_size():
    """Test case with a large dataset and batching."""
    embeddings = torch.rand(1000, 128)  # Random embeddings
    result = community_detection(embeddings, threshold=0.8, min_community_size=10, batch_size=256)
    # Check that all communities meet the minimum size requirement
    assert all(len(community) >= 10 for community in result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_community_detection_gpu_support():
    """Test case for GPU support (if available)."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
    ).cuda()
    expected = [
        [0, 1, 2],  # Single community
    ]
    result = community_detection(embeddings, threshold=0.8, min_community_size=2)
    assert sorted([sorted(community) for community in result]) == sorted([sorted(community) for community in expected])
