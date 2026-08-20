from __future__ import annotations

# Base modules are re-exported here so all modules used to build a SparseEncoder
# can be imported from this single path.
from ...base.modules.dense import Dense
from ...base.modules.input_module import InputModule
from ...base.modules.module import Module
from ...base.modules.router import Asym, Router
from ...base.modules.transformer import Transformer
from .mlm_transformer import MLMTransformer
from .sparse_auto_encoder import SparseAutoEncoder
from .sparse_auto_encoder_token_encoder import SparseAutoEncoderTokenEncoder
from .sparse_static_embedding import SparseStaticEmbedding
from .sparse_token_pooling import SparseTokenPooling
from .splade_pooling import SpladePooling

__all__ = [
    "Dense",
    "InputModule",
    "Module",
    "Asym",
    "Router",
    "Transformer",
    "MLMTransformer",
    "SparseAutoEncoder",
    "SparseAutoEncoderTokenEncoder",
    "SparseStaticEmbedding",
    "SparseTokenPooling",
    "SpladePooling",
]
