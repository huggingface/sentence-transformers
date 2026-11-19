from __future__ import annotations

import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SparseMSELoss


def test_sparse_mse_load_projection(splade_bert_tiny_model: SparseEncoder, tmp_path) -> None:
    """SparseMSELoss inherits load_projection, which must size the projection from the SparseEncoder's dimension."""
    student_dim = splade_bert_tiny_model.get_embedding_dimension()
    source = SparseMSELoss(splade_bert_tiny_model)
    source.projection = torch.nn.Linear(student_dim, 8, device=splade_bert_tiny_model.device)
    source.save_projection(tmp_path / "proj.safetensors")

    loss = SparseMSELoss(splade_bert_tiny_model)
    loss.load_projection(tmp_path / "proj.safetensors")

    torch.testing.assert_close(loss.projection.state_dict(), source.projection.state_dict(), rtol=0, atol=0)
