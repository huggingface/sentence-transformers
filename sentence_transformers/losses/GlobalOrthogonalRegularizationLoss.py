from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import cos_sim


class GlobalOrthogonalRegularizationLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        similarity_fct=cos_sim,
        mean_weight: float | None = 1.0,
        second_moment_weight: float | None = 1.0,
        aggregation: Literal["mean", "sum"] = "mean",
    ) -> None:
        """
        Global Orthogonal Regularization (GOR) Loss that encourages embeddings to be well-distributed
        in the embedding space by penalizing high mean similarities and high second moments of similarities
        across unrelated inputs.

        The loss consists of two terms:

        1. Mean term: Penalizes when the mean similarity across unrelated embeddings is high
        2. Second moment term: Penalizes when the second moment of similarities is high

        A high second moment indicates that some embeddings have very high similarities, suggesting clustering
        or concentration in certain regions of the embedding space. A low second moment indicates that
        similarities are more uniformly distributed.

        The loss is called independently on each input column (e.g., queries and passages) and combines the results
        using either mean or sum aggregation. This is why the loss can be used on any dataset configuration
        (e.g., single inputs, pairs, triplets, etc.).

        It's recommended to combine this loss with a primary loss function, such as :class:`MultipleNegativesRankingLoss`.

        Args:
            model: SentenceTransformer model
            similarity_fct: Function to compute similarity between embeddings (default: cosine similarity)
            mean_weight: Weight for the mean term loss component. None or 0 can be used to disable this term (default: 1.0)
            second_moment_weight: Weight for the second moment term loss component. None or 0 can be used to disable this term (default: 1.0)
            aggregation: How to combine losses across input columns. Either "mean" or "sum" (default: "mean").
                The EmbeddingGemma paper uses "sum".

        References:
            - For further details, see: https://arxiv.org/abs/1708.06320 or https://arxiv.org/abs/2509.20354.
              The latter paper uses the equivalent of GOR with ``mean_weight=0.0`` and ``aggregation="sum"``.

        Inputs:
            +-------+--------+
            | Texts | Labels |
            +=======+========+
            | any   | none   |
            +-------+--------+

        Example:
            ::

                import torch
                from datasets import Dataset
                from torch import Tensor
                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from sentence_transformers.losses import GlobalOrthogonalRegularizationLoss, MultipleNegativesRankingLoss
                from sentence_transformers.util import cos_sim

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })

                class InfoNCEGORLoss(torch.nn.Module):
                    def __init__(self, model: SentenceTransformer, similarity_fct=cos_sim, scale=20.0) -> None:
                        super().__init__()
                        self.model = model
                        self.info_nce_loss = MultipleNegativesRankingLoss(model, similarity_fct=similarity_fct, scale=scale)
                        self.gor_loss = GlobalOrthogonalRegularizationLoss(model, similarity_fct=similarity_fct)

                    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
                        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
                        info_nce_loss: dict[str, Tensor] = {
                            "info_nce": self.info_nce_loss.compute_loss_from_embeddings(embeddings, labels)
                        }
                        gor_loss: dict[str, Tensor] = self.gor_loss.compute_loss_from_embeddings(embeddings, labels)
                        return {**info_nce_loss, **gor_loss}

                loss = InfoNCEGORLoss(model)
                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            Alternatively, you can use multi-task learning to train with both losses:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from sentence_transformers.losses import GlobalOrthogonalRegularizationLoss, MultipleNegativesRankingLoss
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                mnrl_loss = MultipleNegativesRankingLoss(model)
                gor_loss = GlobalOrthogonalRegularizationLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset={"main": train_dataset, "gor": train_dataset},
                    loss={"main": mnrl_loss, "gor": gor_loss},
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.mean_weight = mean_weight
        self.second_moment_weight = second_moment_weight
        if not mean_weight and not second_moment_weight:
            raise ValueError("At least one of mean_weight or second_moment_weight must be non-zero")
        if aggregation not in ["mean", "sum"]:
            raise ValueError(f"aggregation must be 'mean' or 'sum', got '{aggregation}'")
        self.aggregation = aggregation

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
            stacked_mean = torch.stack(mean_terms)
            aggregated_mean = stacked_mean.sum() if self.aggregation == "sum" else stacked_mean.mean()
            results["gor_mean"] = self.mean_weight * aggregated_mean
        if self.second_moment_weight:
            stacked_second_moment = torch.stack(second_moment_terms)
            aggregated_second_moment = (
                stacked_second_moment.sum() if self.aggregation == "sum" else stacked_second_moment.mean()
            )
            results["gor_second_moment"] = self.second_moment_weight * aggregated_second_moment
        return results

    def compute_gor(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the Global Orthogonal Regularization terms for a batch of embeddings.

        The GOR loss encourages embeddings to be well-distributed by:
        1. Mean term (M_1^2): Penalizes high mean similarity, pushing embeddings apart
        2. Second moment term (M_2 - 1/d): Penalizes when the second moment exceeds 1/d, encouraging uniform distribution

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

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "similarity_fct": self.similarity_fct.__name__,
            "mean_weight": self.mean_weight,
            "second_moment_weight": self.second_moment_weight,
            "aggregation": self.aggregation,
        }

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
