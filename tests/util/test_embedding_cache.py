"""Tests for the EmbeddingCache utility."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import EmbeddingCache


@pytest.fixture
def cache_directory():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def embedding_cache(cache_directory: Path) -> EmbeddingCache:
    """Create an EmbeddingCache instance."""
    return EmbeddingCache(cache_directory)


class TestEmbeddingCache:
    """Test suite for EmbeddingCache class."""

    def test_cache_miss_computes_embeddings(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test that cache miss triggers embedding computation."""
        sentences = ["Hello world", "How are you"]
        
        # First call should compute embeddings
        embeddings = embedding_cache.encode(stsb_bert_tiny_model, sentences)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == stsb_bert_tiny_model.get_sentence_embedding_dimension()

    def test_cache_hit_returns_cached(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test that cache hit returns stored embeddings without recomputation."""
        sentences = ["Hello world", "How are you"]
        
        # First call - compute and cache
        embeddings1 = embedding_cache.encode(stsb_bert_tiny_model, sentences)
        
        # Second call - should return from cache
        embeddings2 = embedding_cache.encode(stsb_bert_tiny_model, sentences)
        
        # Should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)

    def test_partial_cache_hit(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test that partial cache hits only compute missing sentences."""
        # First call with 2 sentences
        sentences1 = ["Hello world", "How are you"]
        embeddings1 = embedding_cache.encode(stsb_bert_tiny_model, sentences1)
        
        # Second call with 1 cached + 1 new sentence
        sentences2 = ["Hello world", "New sentence"]
        embeddings2 = embedding_cache.encode(stsb_bert_tiny_model, sentences2)
        
        # First embedding should match between calls
        np.testing.assert_array_almost_equal(embeddings1[0], embeddings2[0])
        
        # Second embedding should be different
        assert not np.allclose(embeddings2[0], embeddings2[1])

    def test_single_sentence_input(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test encoding a single sentence (string input)."""
        sentence = "Hello world"
        
        embedding = embedding_cache.encode(stsb_bert_tiny_model, sentence)
        
        # Should return 1D array for single sentence
        assert embedding.ndim == 1
        assert embedding.shape[0] == stsb_bert_tiny_model.get_sentence_embedding_dimension()

    def test_get_method(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test the get() method for checking cache contents."""
        sentences = ["Hello world", "Missing sentence"]
        
        # Cache only the first sentence
        embedding_cache.encode(stsb_bert_tiny_model, ["Hello world"])
        
        # Get should return embedding for cached, None for missing
        cached = embedding_cache.get(stsb_bert_tiny_model, sentences)
        
        assert cached["Hello world"] is not None
        assert cached["Missing sentence"] is None

    def test_add_method(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test the add() method for manually adding embeddings."""
        sentences = ["Manual sentence"]
        embeddings = np.random.rand(1, stsb_bert_tiny_model.get_sentence_embedding_dimension()).astype(np.float32)
        
        # Manually add to cache
        embedding_cache.add(stsb_bert_tiny_model, sentences, embeddings)
        
        # Should be retrievable
        cached = embedding_cache.get(stsb_bert_tiny_model, sentences)
        np.testing.assert_array_almost_equal(cached["Manual sentence"], embeddings[0])

    def test_clear_cache(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test clearing the cache."""
        sentences = ["Hello world"]
        
        # Add to cache
        embedding_cache.encode(stsb_bert_tiny_model, sentences)
        assert len(embedding_cache) > 0
        
        # Clear cache
        embedding_cache.clear()
        
        # Should be empty
        assert len(embedding_cache) == 0

    def test_clear_model_specific(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test clearing cache for a specific model."""
        sentences = ["Hello world"]
        
        # Add to cache
        embedding_cache.encode(stsb_bert_tiny_model, sentences)
        
        # Clear for specific model
        embedding_cache.clear(model=stsb_bert_tiny_model)
        
        # Should no longer be cached
        cached = embedding_cache.get(stsb_bert_tiny_model, sentences)
        assert cached["Hello world"] is None

    def test_stats(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test cache statistics."""
        sentences = ["Hello world", "How are you", "Nice day"]
        
        # Initially empty
        stats = embedding_cache.stats()
        assert stats["num_embeddings"] == 0
        
        # Add some embeddings
        embedding_cache.encode(stsb_bert_tiny_model, sentences)
        
        # Should show 3 embeddings
        stats = embedding_cache.stats()
        assert stats["num_embeddings"] == 3
        assert stats["total_size_bytes"] > 0

    def test_custom_model_id(
        self, stsb_bert_tiny_model: SentenceTransformer, cache_directory: Path
    ):
        """Test using a custom model ID."""
        cache = EmbeddingCache(cache_directory, model_id="my-custom-model")
        sentences = ["Hello world"]
        
        cache.encode(stsb_bert_tiny_model, sentences)
        
        # Should use custom model ID in path
        assert (cache_directory / "my-custom-model").exists()

    def test_encode_kwargs_passed_through(
        self, stsb_bert_tiny_model: SentenceTransformer, embedding_cache: EmbeddingCache
    ):
        """Test that encode kwargs are passed to the model."""
        sentences = ["Hello world"]
        
        # Should work with various encode parameters
        embeddings = embedding_cache.encode(
            stsb_bert_tiny_model,
            sentences,
            batch_size=1,
            show_progress_bar=False,
        )
        
        assert embeddings.shape[0] == 1
