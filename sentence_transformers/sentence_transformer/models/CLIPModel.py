from __future__ import annotations

from sentence_transformers.base.models.Transformer import Transformer


# For backwards compatibility, we ensure that the legacy `CLIPModel` alias points to the updated `Transformer` class.
class CLIPModel(Transformer):
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch32", *args, **kwargs) -> None:
        super().__init__(model_name_or_path=model_name_or_path, *args, **kwargs)
