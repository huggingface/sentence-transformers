"""Embedding cache utility for avoiding recomputation of sentence embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache for storing and retrieving sentence embeddings to avoid recomputation.

    This utility helps speed up workflows where the same sentences are encoded multiple times,
    such as in RAG applications or semantic search systems that restart frequently.

    Args:
        cache_dir: Directory path where cached embeddings will be stored.
        model_id: Optional model identifier. If not provided, it will be auto-detected
            from the model when `encode()` is called.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import EmbeddingCache

            model = SentenceTransformer("all-MiniLM-L6-v2")
            cache = EmbeddingCache("./embeddings_cache")

            # First call - computes and caches embeddings
            embeddings = cache.encode(model, ["Hello world", "How are you"])

            # Second call - loads from cache (instant)
            embeddings = cache.encode(model, ["Hello world", "How are you"])

            # Mixed - only computes new sentences
            embeddings = cache.encode(model, ["Hello world", "New sentence"])
    """

    def __init__(self, cache_dir: str | Path, model_id: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self._initialized = False

    def _get_model_cache_dir(self, model: "SentenceTransformer") -> Path:
        """Get the cache directory for a specific model."""
        if self.model_id:
            model_name = self.model_id
        else:
            # Try to get model name from various sources
            model_name = getattr(model, "model_card_data", {})
            if hasattr(model_name, "model_id"):
                model_name = model_name.model_id
            elif hasattr(model, "_model_card_text"):
                model_name = "unknown_model"
            else:
                # Fallback: use a hash of the model's config
                model_name = self._get_model_hash(model)

        # Sanitize model name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(model_name))
        return self.cache_dir / safe_name

    def _get_model_hash(self, model: "SentenceTransformer") -> str:
        """Generate a hash for a model based on its configuration."""
        try:
            config_str = str(model.get_sentence_embedding_dimension())
            return hashlib.md5(config_str.encode()).hexdigest()[:12]
        except Exception:
            return "default"

    def _hash_sentence(self, sentence: str) -> str:
        """Create a unique hash key for a sentence."""
        return hashlib.sha256(sentence.encode("utf-8")).hexdigest()

    def _get_embedding_path(self, model_cache_dir: Path, sentence_hash: str) -> Path:
        """Get the file path for a cached embedding."""
        # Use first 2 chars as subdirectory to avoid too many files in one folder
        subdir = model_cache_dir / sentence_hash[:2]
        return subdir / f"{sentence_hash}.npy"

    def _ensure_initialized(self, model_cache_dir: Path) -> None:
        """Ensure the cache directory structure exists."""
        if not self._initialized:
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def get(self, model: "SentenceTransformer", sentences: list[str]) -> dict[str, np.ndarray | None]:
        """
        Get cached embeddings for sentences.

        Args:
            model: The SentenceTransformer model (used to determine cache location).
            sentences: List of sentences to look up.

        Returns:
            Dictionary mapping sentences to their embeddings (None if not cached).
        """
        model_cache_dir = self._get_model_cache_dir(model)
        result = {}

        for sentence in sentences:
            sentence_hash = self._hash_sentence(sentence)
            embedding_path = self._get_embedding_path(model_cache_dir, sentence_hash)

            if embedding_path.exists():
                try:
                    result[sentence] = np.load(embedding_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    result[sentence] = None
            else:
                result[sentence] = None

        return result

    def add(self, model: "SentenceTransformer", sentences: list[str], embeddings: np.ndarray) -> None:
        """
        Add embeddings to the cache.

        Args:
            model: The SentenceTransformer model (used to determine cache location).
            sentences: List of sentences that were encoded.
            embeddings: The embedding matrix (shape: [num_sentences, embedding_dim]).
        """
        model_cache_dir = self._get_model_cache_dir(model)
        self._ensure_initialized(model_cache_dir)

        for i, sentence in enumerate(sentences):
            sentence_hash = self._hash_sentence(sentence)
            embedding_path = self._get_embedding_path(model_cache_dir, sentence_hash)

            # Ensure subdirectory exists
            embedding_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                np.save(embedding_path, embeddings[i])
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

    def encode(
        self,
        model: "SentenceTransformer",
        sentences: str | list[str],
        **encode_kwargs: Any,
    ) -> np.ndarray:
        """
        Encode sentences with caching support.

        This method checks the cache for existing embeddings, computes only the missing ones,
        and returns the complete embedding matrix in the original order.

        Args:
            model: The SentenceTransformer model to use for encoding.
            sentences: A sentence or list of sentences to encode.
            **encode_kwargs: Additional keyword arguments passed to model.encode().

        Returns:
            Numpy array of embeddings with shape [num_sentences, embedding_dim].
        """
        # Handle single sentence input
        input_was_string = isinstance(sentences, str)
        if input_was_string:
            sentences = [sentences]

        sentences = list(sentences)  # Ensure it's a list
        num_sentences = len(sentences)

        # Check cache for existing embeddings
        cached = self.get(model, sentences)

        # Find sentences that need to be computed
        sentences_to_compute = []
        indices_to_compute = []
        for i, sentence in enumerate(sentences):
            if cached[sentence] is None:
                sentences_to_compute.append(sentence)
                indices_to_compute.append(i)

        # Log cache statistics
        cache_hits = num_sentences - len(sentences_to_compute)
        if cache_hits > 0:
            logger.info(f"EmbeddingCache: {cache_hits}/{num_sentences} cache hits")

        # Compute missing embeddings
        if sentences_to_compute:
            # Ensure we get numpy output for caching
            kwargs = encode_kwargs.copy()
            kwargs["convert_to_numpy"] = True
            kwargs["convert_to_tensor"] = False

            new_embeddings = model.encode(sentences_to_compute, **kwargs)

            # Add to cache
            self.add(model, sentences_to_compute, new_embeddings)

            # Update cached dict with new embeddings
            for i, sentence in enumerate(sentences_to_compute):
                cached[sentence] = new_embeddings[i]

        # Build result array in original order
        embedding_dim = next(iter(cached.values())).shape[0]
        result = np.zeros((num_sentences, embedding_dim), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            result[i] = cached[sentence]

        if input_was_string:
            return result[0]

        return result

    def clear(self, model: "SentenceTransformer | None" = None) -> None:
        """
        Clear cached embeddings.

        Args:
            model: If provided, only clear cache for this model.
                If None, clear the entire cache directory.
        """
        import shutil

        if model is not None:
            model_cache_dir = self._get_model_cache_dir(model)
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)
                logger.info(f"Cleared cache for model at: {model_cache_dir}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleared entire cache at: {self.cache_dir}")

        self._initialized = False

    def __len__(self) -> int:
        """Return the approximate number of cached embeddings."""
        count = 0
        if self.cache_dir.exists():
            for npy_file in self.cache_dir.rglob("*.npy"):
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including size and file count.
        """
        total_size = 0
        file_count = 0
        models = set()

        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    models.add(item.name)

            for npy_file in self.cache_dir.rglob("*.npy"):
                file_count += 1
                total_size += npy_file.stat().st_size

        return {
            "cache_dir": str(self.cache_dir),
            "num_embeddings": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "num_models": len(models),
            "models": list(models),
        }
