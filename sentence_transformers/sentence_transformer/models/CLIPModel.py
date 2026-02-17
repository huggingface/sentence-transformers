from __future__ import annotations

from sentence_transformers.base.models.Transformer import ModalityConfig, Transformer


# For backwards compatibility, we ensure that the legacy `CLIPModel` alias points to the updated `Transformer` class.
class CLIPModel(Transformer):
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch32", **kwargs) -> None:
        if "processor_name" in kwargs:
            kwargs["tokenizer_name_or_path"] = kwargs.pop("processor_name")
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    def _get_default_modality_config(self) -> tuple[ModalityConfig, str]:
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[MODALITY_CONFIG, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        modality_config: ModalityConfig = {
            "text": {
                "method": "get_text_features",
                "method_output_name": None,
            },
            "image": {
                "method": "get_image_features",
                "method_output_name": None,
            },
        }
        module_output_name = "sentence_embedding"
        return modality_config, module_output_name
