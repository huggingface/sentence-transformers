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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "MLMTransformer is deprecated and will be removed in a future release. "
            "Please use sentence_transformers.sentence_transformer.modules.Transformer with "
            '`transformer_task="fill-mask"` instead.'
        )
        transformer_task = kwargs.pop("transformer_task", "fill-mask")
        super().__init__(*args, transformer_task=transformer_task, **kwargs)

    @staticmethod
    def _get_default_modality_config(config: dict[str, Any]) -> tuple[ModalityConfig, str]:
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[MODALITY_CONFIG, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        return TRANSFORMER_TASK_DEFAULTS[config.get("transformer_task", "fill-mask")]
