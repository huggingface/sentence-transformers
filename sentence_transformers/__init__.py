from __future__ import annotations

__version__ = "5.3.0.dev0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

import importlib
import os
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
from sentence_transformers.util.deprecated_import import setup_deprecated_module_imports
from sentence_transformers.util.logging import LoggingHandler
from sentence_transformers.util.quantization import quantize_embeddings
from sentence_transformers.util.similarity import SimilarityFunction

# TODO: Rename 'models' to 'modules' finally?

# Set up backward compatibility for moved/renamed modules
setup_deprecated_module_imports()

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
