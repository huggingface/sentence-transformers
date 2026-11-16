"""
Deprecated evaluator wrappers retained for compatibility with existing code.
Use the CrossEncoder evaluators named in each class's deprecation notice instead.

Maintenance is limited to compatibility and security fixes. New features are out of scope.
"""

from __future__ import annotations

from typing_extensions import deprecated

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation.classification import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation.correlation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEBinaryAccuracyEvaluator(CrossEncoderClassificationEvaluator):
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEBinaryClassificationEvaluator(CrossEncoderClassificationEvaluator):
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEF1Evaluator(CrossEncoderClassificationEvaluator):
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CESoftmaxAccuracyEvaluator(CrossEncoderClassificationEvaluator):
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "The CECorrelationEvaluator has been renamed to CrossEncoderCorrelationEvaluator. "
    "Please use CrossEncoderCorrelationEvaluator instead."
)
class CECorrelationEvaluator(CrossEncoderCorrelationEvaluator):
    pass


@deprecated(
    "The CERerankingEvaluator has been renamed to CrossEncoderRerankingEvaluator. "
    "Please use CrossEncoderRerankingEvaluator instead."
)
class CERerankingEvaluator(CrossEncoderRerankingEvaluator):
    pass
