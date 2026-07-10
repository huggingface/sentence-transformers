from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch.nn.functional as F
from torch import Tensor

from sentence_transformers.base.modules.module import Module


# TODO: Perhaps move to base?
class Normalize(Module):
    """L2-normalizes the embeddings under one of the feature keys to have unit length.

    By default operates on the pooled ``sentence_embedding`` produced by a
    :class:`~sentence_transformers.sentence_transformer.modules.Pooling` layer. Pass
    ``module_input_name="token_embeddings"`` to normalize token-level embeddings instead — useful for
    multi-vector / ColBERT-style models that score with MaxSim (which expects L2-normalized rows so
    inner products give cosine similarities in ``[-1, 1]``).

    Args:
        module_input_name: The key in the features dict to normalize. Defaults to ``"sentence_embedding"``.
        module_output_name: The key to write the normalized tensor to. Defaults to ``module_input_name``.
    """

    config_keys: list[str] = ["module_input_name", "module_output_name"]

    def __init__(
        self,
        module_input_name: str = "sentence_embedding",
        module_output_name: str | None = None,
    ) -> None:
        super().__init__()
        self.module_input_name = module_input_name
        self.module_output_name = module_output_name if module_output_name is not None else module_input_name

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        x = features.get(self.module_input_name)
        if x is not None:
            features[self.module_output_name] = F.normalize(x, p=2, dim=-1)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str = "",
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        if not model_name_or_path:
            # Backward compatibility with the old no-arg signature: returns the default Normalize.
            return cls()
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(**config)
