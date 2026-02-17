from __future__ import annotations

import logging
from typing import Any

from sentence_transformers.base.models import Transformer

logger = logging.getLogger(__name__)


class MLMTransformer(Transformer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "MLMTransformer is deprecated and will be removed in a future release. "
            "Please use sentence_transformers.sentence_transformer.models.Transformer with "
            '`transformer_task="fill-mask"` instead.'
        )
        transformer_task = kwargs.pop("transformer_task", "fill-mask")
        super().__init__(*args, transformer_task=transformer_task, **kwargs)
