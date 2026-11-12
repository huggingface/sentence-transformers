from __future__ import annotations

import re
from typing import Any

from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling, HierarchicalTokenPooling
from sentence_transformers.util import fullname


def pooling_config(token_pooling: BaseTokenPooling) -> dict[str, Any]:
    """Describe the pooling class and its complete declared configuration."""
    return {
        "type": fullname(token_pooling),
        "config": token_pooling.get_config_dict(),
    }


def pooling_name_suffix(token_pooling: BaseTokenPooling) -> str:
    """Short pooling label for metrics. Use evaluator names to distinguish finer configuration differences."""
    if isinstance(token_pooling, HierarchicalTokenPooling):
        if token_pooling.distance_threshold is not None:
            return f"distance_threshold_{token_pooling.distance_threshold}"
        if token_pooling.max_tokens is not None:
            return f"max_tokens_{token_pooling.max_tokens}"
        return f"pool_factor_{token_pooling.pool_factor:g}"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(token_pooling).__name__).lower()
