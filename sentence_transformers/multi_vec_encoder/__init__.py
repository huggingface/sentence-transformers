from __future__ import annotations

# LateInteractionPooling must be imported first to avoid circular import
# (MultiVectorEncoder imports LateInteractionPooling from this package)
from sentence_transformers.multi_vec_encoder.LateInteractionPooling import LateInteractionPooling
from sentence_transformers.multi_vec_encoder.MultiVectorEncoder import MultiVectorEncoder

__all__ = [
    "MultiVectorEncoder",
    "LateInteractionPooling",
]
