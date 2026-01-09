from __future__ import annotations

import importlib

import pytest

from sentence_transformers.cross_encoder.evaluation.deprecated import (
    CEBinaryAccuracyEvaluator,
    CEBinaryClassificationEvaluator,
    CECorrelationEvaluator,
    CEF1Evaluator,
    CERerankingEvaluator,
    CESoftmaxAccuracyEvaluator,
)


@pytest.mark.parametrize(
    ("module_names", "module_attributes"),
    [
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEBinaryAccuracyEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEBinaryAccuracyEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEBinaryClassificationEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEF1Evaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEF1Evaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CESoftmaxAccuracyEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CESoftmaxAccuracyEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CERerankingEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CECorrelationEvaluator,
            ],
        ),
    ],
)
def test_import(module_names: list[str], module_attributes: list[object]) -> None:
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for module_attribute in module_attributes:
            obj = getattr(module, module_attribute.__name__, None)
            assert obj is module_attribute
