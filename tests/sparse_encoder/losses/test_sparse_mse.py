from __future__ import annotations

import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SparseMSELoss


def test_sparse_mse_load_projection(splade_bert_tiny_model: SparseEncoder, tmp_path) -> None:
    """SparseMSELoss inherits load_projection, which must size the projection from the SparseEncoder's dimension."""
    student_dim = splade_bert_tiny_model.get_embedding_dimension()
    source = SparseMSELoss(splade_bert_tiny_model)
    source.projection = torch.nn.Linear(student_dim, 8)
    source.save_projection(tmp_path / "proj.safetensors")

    loss = SparseMSELoss(splade_bert_tiny_model)
    loss.load_projection(tmp_path / "proj.safetensors")

    assert loss.projection.in_features == student_dim
    assert loss.projection.out_features == 8
