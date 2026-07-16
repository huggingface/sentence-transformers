from __future__ import annotations

import torch

from sentence_transformers import SparseEncoder, util
from sentence_transformers.sparse_encoder.losses import SparseCoSENTLoss


def _sparse_embeddings() -> list[torch.Tensor]:
    torch.manual_seed(12)
    return [torch.randn(4, 8).relu().to_sparse(), torch.randn(4, 8).relu().to_sparse()]


def test_sparse_cosent_loss_default_similarity_fct_is_pairwise(
    splade_bert_tiny_model: SparseEncoder,
) -> None:
    # CoSENT compares one similarity per input pair, so the default must be a pairwise
    # function, as this loss' own docstring and the parent CoSENTLoss both say.
    loss = SparseCoSENTLoss(splade_bert_tiny_model)

    assert loss.similarity_fct is util.pairwise_cos_sim


def test_sparse_cosent_loss_default_matches_explicit_pairwise(
    splade_bert_tiny_model: SparseEncoder,
) -> None:
    # A full cos_sim matrix makes the score difference a 3-D tensor that the 2-D label
    # mask silently broadcasts over, so the loss stays finite while optimizing a
    # different objective. The default must give the same loss as the pairwise function.
    embeddings = _sparse_embeddings()
    labels = torch.tensor([0.9, 0.1, 0.8, 0.2])

    default = SparseCoSENTLoss(splade_bert_tiny_model)
    reference = SparseCoSENTLoss(splade_bert_tiny_model, similarity_fct=util.pairwise_cos_sim)

    assert torch.allclose(
        default.compute_loss_from_embeddings(embeddings, labels),
        reference.compute_loss_from_embeddings(embeddings, labels),
    )
