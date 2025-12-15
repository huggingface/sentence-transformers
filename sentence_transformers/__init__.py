from __future__ import annotations

__version__ = "5.3.0.dev0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

import importlib
import os
import sys
import warnings

from sentence_transformers.backend import (
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
    export_static_quantized_openvino_model,
)
from sentence_transformers.base.sampler import DefaultBatchSampler, MultiDatasetDefaultBatchSampler
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.LoggingHandler import LoggingHandler
from sentence_transformers.sentence_transformer.datasets import ParallelSentencesDataset, SentencesDataset
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.util.quantization import quantize_embeddings
from sentence_transformers.util.similarity_functions import SimilarityFunction

# Backward compatibility: make SentenceTransformer available at the old import path
# TODO: This might need extending + it needs testing + reducing
# TODO: How to update e.g. `from sentence_transformers.trainer import SentenceTransformerTrainer`
# so it works like `from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer`
deprecated_modules = [
    ("sentence_transformers.SentenceTransformer", "sentence_transformers.sentence_transformer.model"),
    ("sentence_transformers.models", "sentence_transformers.sentence_transformer.models"),
    (
        "sentence_transformers.models.tokenizer.WordTokenizer",
        "sentence_transformers.sentence_transformer.models.tokenizer.WordTokenizer",
    ),
    (
        "sentence_transformers.models.tokenizer.PhraseTokenizer",
        "sentence_transformers.sentence_transformer.models.tokenizer.PhraseTokenizer",
    ),
    (
        "sentence_transformers.models.tokenizer.WhitespaceTokenizer",
        "sentence_transformers.sentence_transformer.models.tokenizer.WhitespaceTokenizer",
    ),
    ("sentence_transformers.models.tokenizer", "sentence_transformers.sentence_transformer.models.tokenizer"),
    ("sentence_transformers.models.BoW", "sentence_transformers.sentence_transformer.models.BoW"),
    ("sentence_transformers.models.CLIPModel", "sentence_transformers.sentence_transformer.models.CLIPModel"),
    ("sentence_transformers.models.CNN", "sentence_transformers.sentence_transformer.models.CNN"),
    ("sentence_transformers.models.Dense", "sentence_transformers.sentence_transformer.models.Dense"),
    ("sentence_transformers.models.Dropout", "sentence_transformers.sentence_transformer.models.Dropout"),
    ("sentence_transformers.models.LayerNorm", "sentence_transformers.sentence_transformer.models.LayerNorm"),
    ("sentence_transformers.models.LSTM", "sentence_transformers.sentence_transformer.models.LSTM"),
    ("sentence_transformers.models.Normalize", "sentence_transformers.sentence_transformer.models.Normalize"),
    ("sentence_transformers.models.Pooling", "sentence_transformers.sentence_transformer.models.Pooling"),
    ("sentence_transformers.models.Router", "sentence_transformers.base.models.Router"),
    ("sentence_transformers.models.Asym", "sentence_transformers.base.models.Router"),
    ("sentence_transformers.models.InputModule", "sentence_transformers.base.models.InputModule"),
    ("sentence_transformers.models.Module", "sentence_transformers.base.models.Module"),
    ("sentence_transformers.models.Transformer", "sentence_transformers.base.models.Transformer"),
    ("sentence_transformers.models.modality_utils", "sentence_transformers.base.models.modality_utils"),
    (
        "sentence_transformers.models.StaticEmbedding",
        "sentence_transformers.sentence_transformer.models.StaticEmbedding",
    ),
    (
        "sentence_transformers.models.WeightedLayerPooling",
        "sentence_transformers.sentence_transformer.models.WeightedLayerPooling",
    ),
    (
        "sentence_transformers.models.WordEmbeddings",
        "sentence_transformers.sentence_transformer.models.WordEmbeddings",
    ),
    ("sentence_transformers.models.WordWeights", "sentence_transformers.sentence_transformer.models.WordWeights"),
    ("sentence_transformers.evaluation", "sentence_transformers.sentence_transformer.evaluation"),
    (
        "sentence_transformers.evaluation.BinaryClassificationEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.BinaryClassificationEvaluator",
    ),
    (
        "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.EmbeddingSimilarityEvaluator",
    ),
    (
        "sentence_transformers.evaluation.InformationRetrievalEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator",
    ),
    (
        "sentence_transformers.evaluation.LabelAccuracyEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.LabelAccuracyEvaluator",
    ),
    (
        "sentence_transformers.evaluation.MSEEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.MSEEvaluator",
    ),
    (
        "sentence_transformers.evaluation.MSEEvaluatorFromDataFrame",
        "sentence_transformers.sentence_transformer.evaluation.MSEEvaluatorFromDataFrame",
    ),
    (
        "sentence_transformers.evaluation.NanoBEIREvaluator",
        "sentence_transformers.sentence_transformer.evaluation.NanoBEIREvaluator",
    ),
    (
        "sentence_transformers.evaluation.ParaphraseMiningEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.ParaphraseMiningEvaluator",
    ),
    (
        "sentence_transformers.evaluation.RerankingEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.RerankingEvaluator",
    ),
    (
        "sentence_transformers.evaluation.SimilarityFunction",
        "sentence_transformers.sentence_transformer.evaluation.SimilarityFunction",
    ),
    (
        "sentence_transformers.evaluation.TranslationEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.TranslationEvaluator",
    ),
    (
        "sentence_transformers.evaluation.TripletEvaluator",
        "sentence_transformers.sentence_transformer.evaluation.TripletEvaluator",
    ),
    ("sentence_transformers.losses", "sentence_transformers.sentence_transformer.losses"),
    ("sentence_transformers.losses.CoSENTLoss", "sentence_transformers.sentence_transformer.losses.CoSENTLoss"),
    (
        "sentence_transformers.losses.CachedGISTEmbedLoss",
        "sentence_transformers.sentence_transformer.losses.CachedGISTEmbedLoss",
    ),
    (
        "sentence_transformers.losses.CachedMultipleNegativesRankingLoss",
        "sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss",
    ),
    (
        "sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss",
        "sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesSymmetricRankingLoss",
    ),
    (
        "sentence_transformers.losses.AdaptiveLayerLoss",
        "sentence_transformers.sentence_transformer.losses.AdaptiveLayerLoss",
    ),
    ("sentence_transformers.losses.AnglELoss", "sentence_transformers.sentence_transformer.losses.AnglELoss"),
    (
        "sentence_transformers.losses.BatchHardTripletLoss",
        "sentence_transformers.sentence_transformer.losses.BatchHardTripletLoss",
    ),
    (
        "sentence_transformers.losses.BatchAllTripletLoss",
        "sentence_transformers.sentence_transformer.losses.BatchAllTripletLoss",
    ),
    (
        "sentence_transformers.losses.BatchHardSoftMarginTripletLoss",
        "sentence_transformers.sentence_transformer.losses.BatchHardSoftMarginTripletLoss",
    ),
    (
        "sentence_transformers.losses.BatchSemiHardTripletLoss",
        "sentence_transformers.sentence_transformer.losses.BatchSemiHardTripletLoss",
    ),
    (
        "sentence_transformers.losses.ContrastiveLoss",
        "sentence_transformers.sentence_transformer.losses.ContrastiveLoss",
    ),
    (
        "sentence_transformers.losses.ContrastiveTensionLoss",
        "sentence_transformers.sentence_transformer.losses.ContrastiveTensionLoss",
    ),
    (
        "sentence_transformers.losses.CosineSimilarityLoss",
        "sentence_transformers.sentence_transformer.losses.CosineSimilarityLoss",
    ),
    (
        "sentence_transformers.losses.DenoisingAutoEncoderLoss",
        "sentence_transformers.sentence_transformer.losses.DenoisingAutoEncoderLoss",
    ),
    (
        "sentence_transformers.losses.DistillKLDivLoss",
        "sentence_transformers.sentence_transformer.losses.DistillKLDivLoss",
    ),
    ("sentence_transformers.losses.GISTEmbedLoss", "sentence_transformers.sentence_transformer.losses.GISTEmbedLoss"),
    ("sentence_transformers.losses.MarginMSELoss", "sentence_transformers.sentence_transformer.losses.MarginMSELoss"),
    (
        "sentence_transformers.losses.MatryoshkaLoss",
        "sentence_transformers.sentence_transformer.losses.MatryoshkaLoss",
    ),
    (
        "sentence_transformers.losses.Matryoshka2dLoss",
        "sentence_transformers.sentence_transformer.losses.Matryoshka2dLoss",
    ),
    (
        "sentence_transformers.losses.MegaBatchMarginLoss",
        "sentence_transformers.sentence_transformer.losses.MegaBatchMarginLoss",
    ),
    ("sentence_transformers.losses.MSELoss", "sentence_transformers.sentence_transformer.losses.MSELoss"),
    (
        "sentence_transformers.losses.MultipleNegativesRankingLoss",
        "sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss",
    ),
    (
        "sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss",
        "sentence_transformers.sentence_transformer.losses.MultipleNegativesSymmetricRankingLoss",
    ),
    (
        "sentence_transformers.losses.OnlineContrastiveLoss",
        "sentence_transformers.sentence_transformer.losses.OnlineContrastiveLoss",
    ),
    ("sentence_transformers.losses.SoftmaxLoss", "sentence_transformers.sentence_transformer.losses.SoftmaxLoss"),
    ("sentence_transformers.losses.TripletLoss", "sentence_transformers.sentence_transformer.losses.TripletLoss"),
    ("sentence_transformers.readers", "sentence_transformers.sentence_transformer.readers"),
    ("sentence_transformers.readers.InputExample", "sentence_transformers.sentence_transformer.readers.InputExample"),
    (
        "sentence_transformers.readers.LabelSentenceReader",
        "sentence_transformers.sentence_transformer.readers.LabelSentenceReader",
    ),
    (
        "sentence_transformers.readers.NLIDataReader",
        "sentence_transformers.sentence_transformer.readers.NLIDataReader",
    ),
    (
        "sentence_transformers.readers.STSDataReader",
        "sentence_transformers.sentence_transformer.readers.STSDataReader",
    ),
    (
        "sentence_transformers.readers.TripletReader",
        "sentence_transformers.sentence_transformer.readers.TripletReader",
    ),
    ("sentence_transformers.datasets", "sentence_transformers.sentence_transformer.datasets"),
    (
        "sentence_transformers.datasets.DenoisingAutoEncoderDataset",
        "sentence_transformers.sentence_transformer.datasets.DenoisingAutoEncoderDataset",
    ),
    (
        "sentence_transformers.datasets.NoDuplicatesDataLoader",
        "sentence_transformers.sentence_transformer.datasets.NoDuplicatesDataLoader",
    ),
    (
        "sentence_transformers.datasets.ParallelSentencesDataset",
        "sentence_transformers.sentence_transformer.datasets.ParallelSentencesDataset",
    ),
    (
        "sentence_transformers.datasets.SentenceLabelDataset",
        "sentence_transformers.sentence_transformer.datasets.SentenceLabelDataset",
    ),
    (
        "sentence_transformers.datasets.SentencesDataset",
        "sentence_transformers.sentence_transformer.datasets.SentencesDataset",
    ),
]
for deprecated_module_name, new_module_name in deprecated_modules:
    sys.modules[deprecated_module_name] = sys.modules[new_module_name]

# If codecarbon is installed and the log level is not defined,
# automatically overwrite the default to "error"
if importlib.util.find_spec("codecarbon") and "CODECARBON_LOG_LEVEL" not in os.environ:
    os.environ["CODECARBON_LOG_LEVEL"] = "error"

# Globally silence PyTorch sparse CSR tensor beta warning
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

__all__ = [
    "LoggingHandler",
    "SentencesDataset",
    "ParallelSentencesDataset",
    "SentenceTransformer",
    "SimilarityFunction",
    "InputExample",
    "CrossEncoder",
    "CrossEncoderTrainer",
    "CrossEncoderTrainingArguments",
    "CrossEncoderModelCardData",
    "SentenceTransformerTrainer",
    "SentenceTransformerTrainingArguments",
    "SentenceTransformerModelCardData",
    "SparseEncoder",
    "SparseEncoderTrainer",
    "SparseEncoderTrainingArguments",
    "SparseEncoderModelCardData",
    "quantize_embeddings",
    "export_optimized_onnx_model",
    "export_dynamic_quantized_onnx_model",
    "export_static_quantized_openvino_model",
    "DefaultBatchSampler",
    "MultiDatasetDefaultBatchSampler",
    "mine_hard_negatives",
]
