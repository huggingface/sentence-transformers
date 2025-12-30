from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch.nn.functional as F
from torch import Tensor

from sentence_transformers.base.models.Module import Module


class Normalize(Module):
    """This layer normalizes embeddings to unit length"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        return

    @classmethod
    def load(cls, *args, **kwargs) -> Self:
        return cls()
