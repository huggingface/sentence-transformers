from __future__ import annotations

import sys

from ...base.models.InputModule import InputModule
from ...base.models.Module import Module
from ...base.models.Router import Asym, Router
from ...base.models.Transformer import Transformer
from .BoW import BoW
from .CLIPModel import CLIPModel
from .CNN import CNN
from .Dense import Dense
from .Dropout import Dropout
from .LayerNorm import LayerNorm
from .LSTM import LSTM
from .Normalize import Normalize
from .Pooling import Pooling
from .StaticEmbedding import StaticEmbedding
from .WeightedLayerPooling import WeightedLayerPooling
from .WordEmbeddings import WordEmbeddings
from .WordWeights import WordWeights

sys.modules["sentence_transformers.models.Asym"] = sys.modules["sentence_transformers.base.models.Router"]
sys.modules["sentence_transformers.sentence_transformer.models.Asym"] = sys.modules[
    "sentence_transformers.base.models.Router"
]
sys.modules["sentence_transformers.models.Router"] = sys.modules["sentence_transformers.base.models.Router"]
sys.modules["sentence_transformers.sentence_transformer.models.Router"] = sys.modules[
    "sentence_transformers.base.models.Router"
]

__all__ = [
    "Transformer",
    "StaticEmbedding",
    "Asym",
    "BoW",
    "CNN",
    "Dense",
    "Dropout",
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
