from __future__ import annotations

import sys

from .classification import CrossEncoderClassificationEvaluator
from .correlation import CrossEncoderCorrelationEvaluator
from .deprecated import (
    CEBinaryAccuracyEvaluator,
    CEBinaryClassificationEvaluator,
    CECorrelationEvaluator,
    CEF1Evaluator,
    CERerankingEvaluator,
    CESoftmaxAccuracyEvaluator,
)
from .nano_evaluator import CrossEncoderNanoEvaluator
from .nano_beir import CrossEncoderNanoBEIREvaluator
from .reranking import CrossEncoderRerankingEvaluator

# Ensure that imports using deprecated paths still work
# Although importing via `from sentence_transformers.cross_encoder.evaluation import ...` is recommended
deprecated_modules = [
    "sentence_transformers.cross_encoder.evaluation.CEBinaryAccuracyEvaluator",
    "sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator",
    "sentence_transformers.cross_encoder.evaluation.CEF1Evaluator",
    "sentence_transformers.cross_encoder.evaluation.CESoftmaxAccuracyEvaluator",
    "sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator",
    "sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator",
]
for module in deprecated_modules:
    sys.modules[module] = sys.modules["sentence_transformers.cross_encoder.evaluation.deprecated"]

__all__ = [
    "CrossEncoderClassificationEvaluator",
    "CrossEncoderCorrelationEvaluator",
    "CrossEncoderNanoEvaluator",
    "CrossEncoderRerankingEvaluator",
    "CrossEncoderNanoBEIREvaluator",
    # Deprecated:
    "CERerankingEvaluator",
    "CECorrelationEvaluator",
    "CEBinaryAccuracyEvaluator",
    "CEBinaryClassificationEvaluator",
    "CEF1Evaluator",
    "CESoftmaxAccuracyEvaluator",
]
