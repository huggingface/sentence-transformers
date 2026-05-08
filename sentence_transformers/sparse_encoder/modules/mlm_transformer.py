from __future__ import annotations

import logging
from typing import Any

from sentence_transformers.base.modules.transformer import (
    TRANSFORMER_TASK_DEFAULTS,
    ModalityConfig,
    Transformer,
)

logger = logging.getLogger(__name__)


class MLMTransformer(Transformer):
    """Legacy module kept only for backward compatibility with older saved models.

    New SPLADE models should use :class:`~sentence_transformers.base.modules.Transformer` with
    ``transformer_task="fill-mask"`` instead.

    .. deprecated:: 5.4
        Use :class:`~sentence_transformers.base.modules.Transformer` with ``transformer_task="fill-mask"`` instead.
    """

    def __init__(self, *args: Any, _from_auto_load: bool = False, **kwargs: Any) -> None:
        if not _from_auto_load:
            logger.warning(
                "The MLMTransformer module is deprecated. Please use the Transformer module with "
                '`transformer_task="fill-mask"` instead.'
            )
        transformer_task = kwargs.pop("transformer_task", "fill-mask")
        super().__init__(*args, transformer_task=transformer_task, **kwargs)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs) -> MLMTransformer:
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
        return TRANSFORMER_TASK_DEFAULTS[config.get("transformer_task", "fill-mask")]
