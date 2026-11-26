from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder, util
from sentence_transformers.sparse_encoder.losses import SparseAnglELoss, SparseCoSENTLoss


def test_sparse_cosent_loss_default_matches_explicit_pairwise(
    splade_bert_tiny_model: SparseEncoder,
) -> None:
    # CoSENT needs one similarity per input pair. The old cos_sim default returned a
    # full similarity matrix that silently broadcast into a different objective.
    default = SparseCoSENTLoss(splade_bert_tiny_model)
    reference = SparseCoSENTLoss(splade_bert_tiny_model, similarity_fct=util.pairwise_cos_sim)

    assert default.similarity_fct is util.pairwise_cos_sim
    assert reference.similarity_fct is util.pairwise_cos_sim

    torch.manual_seed(12)
    embeddings = [torch.randn(4, 8).relu().to_sparse(), torch.randn(4, 8).relu().to_sparse()]
    labels = torch.tensor([0.9, 0.1, 0.8, 0.2])

    assert torch.allclose(
        default.compute_loss_from_embeddings(embeddings, labels),
        reference.compute_loss_from_embeddings(embeddings, labels),
    )


@pytest.mark.parametrize("loss_class", [SparseCoSENTLoss, SparseAnglELoss])
@pytest.mark.parametrize("sparse", [False, True])
def test_sparse_cosent_matches_pair_matrix_with_tied_labels(loss_class, sparse):
    torch.manual_seed(12)
    dense = [torch.rand(6, 8).requires_grad_(not sparse) for _ in range(2)]
    reference = [x.detach().clone().requires_grad_(not sparse) for x in dense]
    labels = torch.tensor([0.0, 1.0, 0.5, 1.0, 0.0, 0.5])
    loss_fn = loss_class(torch.nn.Identity())

    actual = loss_fn.compute_loss_from_embeddings([x.to_sparse() for x in dense] if sparse else dense, labels)
    scores = loss_fn.similarity_fct(*([x.to_sparse() for x in reference] if sparse else reference)) * loss_fn.scale
    valid_differences = (scores[:, None] - scores[None, :])[labels[:, None] < labels[None, :]]
    expected = torch.logsumexp(torch.cat((scores.new_zeros(1), valid_differences)), dim=0)
    torch.testing.assert_close(actual, expected)
    if not sparse:  # SparseEncoder produces dense embeddings during training.
        actual.backward()
        expected.backward()
        for actual_embedding, reference_embedding in zip(dense, reference):
            torch.testing.assert_close(actual_embedding.grad, reference_embedding.grad)
