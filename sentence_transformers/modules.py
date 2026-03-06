"""
This module provides a collection of all modules across SentenceTransformer, CrossEncoder, and SparseEncoder
for convenient importing.
"""

from __future__ import annotations

from .base.modules import InputModule, Module, Router, Transformer
from .cross_encoder.modules import CausalScoreHead
from .sentence_transformer.modules import (
    CNN,
    LSTM,
    BoW,
    CLIPModel,
    Dense,
    Dropout,
    LayerNorm,
    Normalize,
    Pooling,
    StaticEmbedding,
    WeightedLayerPooling,
    WordEmbeddings,
    WordWeights,
)
from .sparse_encoder.modules import MLMTransformer, SparseAutoEncoder, SparseStaticEmbedding, SpladePooling

__all__ = [
    # Base modules
    "InputModule",
    "Module",
    "Router",
    "Transformer",
    # SentenceTransformer modules
    "BoW",
    "CLIPModel",
    "CNN",
    "Dense",
    "Dropout",
    "LayerNorm",
    "LSTM",
    "Normalize",
    "Pooling",
    "StaticEmbedding",
    "WeightedLayerPooling",
    "WordEmbeddings",
    "WordWeights",
    # CrossEncoder modules
    "CausalScoreHead",
    # SparseEncoder modules
    "MLMTransformer",
    "SparseAutoEncoder",
    "SparseStaticEmbedding",
    "SpladePooling",
]
