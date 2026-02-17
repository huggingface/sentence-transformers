from __future__ import annotations

from sentence_transformers.util.similarity import SimilarityFunction

# TODO: Can I remove this file by e.g. adding SimilarityFunction to __init__.py
# or using the deprecated import logic?
__all__ = ["SimilarityFunction"]
