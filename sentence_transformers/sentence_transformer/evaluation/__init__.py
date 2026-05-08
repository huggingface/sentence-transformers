from __future__ import annotations

# Re-exported from base so the deprecated path `sentence_transformers.evaluation` (which
# redirects here) can still resolve `SentenceEvaluator` and `SequentialEvaluator` as attributes.
from ...base.evaluation.evaluator import BaseEvaluator, SentenceEvaluator
from ...base.evaluation.sequential import SequentialEvaluator
from ...util.similarity import SimilarityFunction
from .binary_classification import BinaryClassificationEvaluator
from .embedding_similarity import EmbeddingSimilarityEvaluator
from .information_retrieval import InformationRetrievalEvaluator
from .label_accuracy import LabelAccuracyEvaluator
from .mse import MSEEvaluator
from .mse_from_dataframe import MSEEvaluatorFromDataFrame
from .nano_beir import NanoBEIREvaluator
from .paraphrase_mining import ParaphraseMiningEvaluator
from .reranking import RerankingEvaluator
from .translation import TranslationEvaluator
from .triplet import TripletEvaluator

__all__ = [
    "BaseEvaluator",
    "SentenceEvaluator",
    "SimilarityFunction",
    "BinaryClassificationEvaluator",
    "EmbeddingSimilarityEvaluator",
    "InformationRetrievalEvaluator",
    "LabelAccuracyEvaluator",
    "MSEEvaluator",
    "MSEEvaluatorFromDataFrame",
    "ParaphraseMiningEvaluator",
    "SequentialEvaluator",
    "TranslationEvaluator",
    "TripletEvaluator",
    "RerankingEvaluator",
    "NanoBEIREvaluator",
]
