from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import cos_sim


class GlobalOrthogonalRegularizationLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        similarity_fct=cos_sim,
        mean_weight: float = 1.0,
        second_moment_weight: float = 1.0,
    ) -> None:
        """
        Global Orthogonal Regularization (GOR) Loss that encourages embeddings to be well-distributed
        in the embedding space by penalizing high mean similarities and high variance in similarities.

        The loss consists of two terms:

        1. Mean term: Penalizes when the mean similarity across embeddings is far from zero
        2. Second moment term: Penalizes when the variance of similarities is too high

        The loss is called independently on each input column (e.g., queries and passages) and averages the results.
        This is why the loss can be used on any dataset configuration (e.g., single inputs, pairs, triplets, etc.).

        It's recommended to combine this loss with a primary loss function, such as :class:`MultipleNegativesRankingLoss`.

        Args:
            model: SentenceTransformer model
            similarity_fct: Function to compute similarity between embeddings (default: cosine similarity)
            mean_weight: Weight for the mean term loss component (default: 1.0)
            second_moment_weight: Weight for the second moment term loss component (default: 1.0)

        References:
            - For further details, see: https://arxiv.org/abs/1708.06320 or https://arxiv.org/abs/2509.20354.
              The latter paper uses the equivalent of GOR with ``mean_weight=0.0``.

        Inputs:
            +-------+--------+
            | Texts | Labels |
            +=======+========+
            | any   | none   |
            +-------+--------+
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.mean_weight = mean_weight
        self.second_moment_weight = second_moment_weight

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(
        self, embeddings: list[Tensor], labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        """
        Compute the GOR loss from pre-computed embeddings.

        Args:
            embeddings: List of embedding tensors, one for each input column (e.g., [queries, passages])
            labels: Not used, kept for compatibility

        Returns:
            Dictionary containing the weighted mean term and second moment term losses
        """
        mean_terms, second_moment_terms = zip(*[self.compute_gor(embedding) for embedding in embeddings])
        results = {}
        if self.mean_weight:
            results["gor_mean"] = self.mean_weight * torch.stack(mean_terms).mean()
        if self.second_moment_weight:
            results["gor_second_moment"] = self.second_moment_weight * torch.stack(second_moment_terms).mean()
        return results

    def compute_gor(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the Global Orthogonal Regularization terms for a batch of embeddings.

        The GOR loss encourages embeddings to be well-distributed by:
        1. Mean term (M_1^2): Penalizes high mean similarity, pushing embeddings apart
        2. Second moment term (M_2 - 1/d): Penalizes high variance, ensuring uniform distribution

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)

        Returns:
            Tuple of (mean_term, second_moment_term) losses (unweighted)
        """
        batch_size = embeddings.size(0)
        hidden_dim = embeddings.size(1)

        # Compute pairwise similarity matrix between all embeddings, and exclude self-similarities
        sim_matrix = self.similarity_fct(embeddings, embeddings)
        sim_matrix.fill_diagonal_(0.0)
        num_off_diagonal = batch_size * (batch_size - 1)

        # Mean term: M_1^2 where M_1 = mean of off-diagonal similarities
        # Penalizes high similarities across inputs from the same column (e.g., queries vs other queries)
        mean_term = (sim_matrix.sum() / num_off_diagonal).pow(2)

        # Second moment term: M_2 - 1/d where M_2 = mean of squared off-diagonal similarities and d is embedding dimension
        # Pushes the second moment close to 1/d, encouraging a more uniform distribution
        second_moment = sim_matrix.pow(2).sum() / num_off_diagonal
        second_moment_term = torch.relu(second_moment - (1.0 / hidden_dim))

        return mean_term, second_moment_term

    @property
    def citation(self) -> str:
        return """
@misc{zhang2017learningspreadoutlocalfeature,
      title={Learning Spread-out Local Feature Descriptors},
      author={Xu Zhang and Felix X. Yu and Sanjiv Kumar and Shih-Fu Chang},
      year={2017},
      eprint={1708.06320},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1708.06320},
}
"""
