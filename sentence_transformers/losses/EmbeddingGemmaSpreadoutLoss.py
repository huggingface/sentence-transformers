from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer


class EmbeddingGemmaSpreadoutLoss(nn.Module):
    def __init__(self, model: SentenceTransformer) -> None:
        """
        This class implements the Spread-out loss(based on Global Orthogonal Regularizer) from the EmbeddingGemma paper (Eq. 4).

        This loss encourages EmbeddingGemma to produce embeddings that are spread out over the embedding space, to fully utilize the expressive power of the embedding space. This also intends to ensure that
            - the model is robust to quantization (especially embedding quantization), and that
            - the embeddings produced by the model can be retrieved efficiently in vector databases using approximate nearest neighbor (ANN) algorithms.

        It uses only the “second moment” term of the regularizer's original definition, as it finds this also sufficiently pushes the “mean” term towards its target value.

        Args:
            model: SentenceTransformer model

        References:
            - For further details, see: https://arxiv.org/abs/2509.20354

        Inputs:
            +----------------------+----------------------+
            | Texts                | Labels               |
            +======================+======================+
            | (query, passage)     | Ignored (None)       |
            +----------------------+----------------------+
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(
        self,
        embeddings: list[Tensor],
    ) -> Tensor:
        # extractquery embeddings and positive passage embeddings
        q, p = embeddings  # (B, D)

        # normalize embeddings to unit length
        q = nn.functional.normalize(q, dim=1)
        p = nn.functional.normalize(p, dim=1)

        # batch size
        B = q.size(0)

        # pairwise cosine similarity matrices
        S_qq = q @ q.T
        S_pp = p @ p.T

        # squared similarities (second moment)
        S_qq2 = S_qq**2
        S_pp2 = S_pp**2

        # remove diagonal terms (i = j)
        S_qq2.fill_diagonal_(0.0)
        S_pp2.fill_diagonal_(0.0)
        # so the sum is now over all pairs of embeddings -> i!=j

        # normalize using batch size and sum
        loss_q = S_qq2.sum() / (B * (B - 1))  # query loss
        loss_p = S_pp2.sum() / (B * (B - 1))  # positive passage loss

        return loss_q + loss_p  # total loss

    @property
    def citation(self) -> str:
        return """
@article{embeddinggemma2025,
  title   = {EmbeddingGemma: Powerful and Lightweight Text Representations},
  author  = {Google Deepmind},
  year    = {2025},
}
"""
