from __future__ import annotations

import sys

from .BoW import BoW
from .CLIPModel import CLIPModel
from .CNN import CNN
from .Dense import Dense
from .Dropout import Dropout
from .InputModule import InputModule
from .LandmarkTransformer import LandmarkTransformer
from .LayerNorm import LayerNorm
from .LSTM import LSTM
from .Module import Module
from .Normalize import Normalize
from .Pooling import Pooling
from .Router import Asym, Router
from .StaticEmbedding import StaticEmbedding
from .Transformer import Transformer
from .WeightedLayerPooling import WeightedLayerPooling
from .WordEmbeddings import WordEmbeddings
from .WordWeights import WordWeights

sys.modules["sentence_transformers.models.Asym"] = sys.modules["sentence_transformers.models.Router"]

__all__ = [
    "Transformer",
    "StaticEmbedding",
    "Asym",
    "BoW",
    "CNN",
    "Dense",
    "Dropout",
    "LandmarkTransformer",
    "LayerNorm",
    "LSTM",
    "Normalize",
    "Pooling",
    "WeightedLayerPooling",
    "WordEmbeddings",
    "WordWeights",
    "CLIPModel",
    "Module",
    "InputModule",
    "Router",
]
