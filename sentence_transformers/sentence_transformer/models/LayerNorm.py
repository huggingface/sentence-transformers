from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn

from sentence_transformers.base.models.Module import Module


class LayerNorm(Module):
    config_keys: list[str] = ["dimension"]

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: dict[str, Tensor]):
        features["sentence_embedding"] = self.norm(features["sentence_embedding"])
        return features

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def save(self, output_path, safe_serialization: bool = True) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
