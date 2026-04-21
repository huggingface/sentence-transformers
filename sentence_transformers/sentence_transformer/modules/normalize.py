from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch.nn.functional as F
from torch import Tensor

from sentence_transformers.base.modules.module import Module


class Normalize(Module):
    """This layer normalizes embeddings to unit length"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        sentence_embedding = features.get("sentence_embedding")
        if sentence_embedding is not None:
            features["sentence_embedding"] = F.normalize(sentence_embedding, p=2, dim=1)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        return

    @classmethod
    def load(cls, *args, **kwargs) -> Self:
        return cls()
