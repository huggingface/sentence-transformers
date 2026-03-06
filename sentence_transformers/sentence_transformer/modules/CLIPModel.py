from __future__ import annotations

from typing import Any

from sentence_transformers.base.modules.Transformer import ModalityConfig, Transformer


# For backwards compatibility, we ensure that the legacy `CLIPModel` alias points to the updated `Transformer` class.
class CLIPModel(Transformer):
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch32", **kwargs) -> None:
        if "processor_name" in kwargs:
            kwargs["tokenizer_name_or_path"] = kwargs.pop("processor_name")
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    @staticmethod
    def _get_default_modality_config(config: dict[str, Any]) -> tuple[ModalityConfig, str]:
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[MODALITY_CONFIG, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        from transformers import CLIPModel

        # Use Transformer._infer_method_output_name to check whether the method outputs a BaseModelOutputWithPooling
        # with a pooler_output, or just a Tensor output
        modality_config: ModalityConfig = {
            "text": {
                "method": "get_text_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", CLIPModel.get_text_features
                ),
            },
            "image": {
                "method": "get_image_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", CLIPModel.get_image_features
                ),
            },
        }
        module_output_name = "sentence_embedding"
        return modality_config, module_output_name
