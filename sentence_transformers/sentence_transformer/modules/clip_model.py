from __future__ import annotations

import logging
from typing import Any

from sentence_transformers.base.modules.transformer import ModalityConfig, Transformer

logger = logging.getLogger(__name__)


class CLIPModel(Transformer):
    """Legacy module kept only for backward compatibility with older saved models.

    New multimodal models should use :class:`~sentence_transformers.base.modules.Transformer` directly,
    which handles CLIP-style models and other multimodal architectures natively via modality auto-detection.

    .. deprecated:: 5.4
        Use :class:`~sentence_transformers.base.modules.Transformer` instead.
    """

    def __init__(
        self, model_name_or_path: str = "openai/clip-vit-base-patch32", _from_auto_load: bool = False, **kwargs
    ) -> None:
        if not _from_auto_load:
            logger.warning("The CLIPModel module is deprecated. Please use the Transformer module instead.")
        if "processor_name" in kwargs:
            kwargs["tokenizer_name_or_path"] = kwargs.pop("processor_name")
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs) -> CLIPModel:
        init_kwargs = cls._load_init_kwargs(model_name_or_path=model_name_or_path, **kwargs)
        return cls(model_name_or_path=model_name_or_path, _from_auto_load=True, **init_kwargs)

    @staticmethod
    def _get_default_modality_config(config: dict[str, Any]) -> tuple[ModalityConfig, str]:
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[MODALITY_CONFIG, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        from transformers import CLIPModel as TransformersCLIPModel

        # Use Transformer._infer_method_output_name to check whether the method outputs a BaseModelOutputWithPooling
        # with a pooler_output, or just a Tensor output
        modality_config: ModalityConfig = {
            "text": {
                "method": "get_text_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", TransformersCLIPModel.get_text_features
                ),
            },
            "image": {
                "method": "get_image_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", TransformersCLIPModel.get_image_features
                ),
            },
        }
        module_output_name = "sentence_embedding"
        return modality_config, module_output_name
