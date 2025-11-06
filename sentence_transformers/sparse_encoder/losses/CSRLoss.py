from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss import (
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


def normalized_mean_squared_error(reconstruction: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    original_input_mean = original_input.mean(dim=0)
    loss = F.mse_loss(reconstruction, original_input) / F.mse_loss(
        original_input_mean[None, :].broadcast_to(original_input.shape), original_input
    )
    return loss


class CSRReconstructionLoss(nn.Module):
    def __init__(self, model: SparseEncoder, beta: float = 1.0) -> None:
        """
        CSRReconstructionLoss implements the reconstruction loss component for Contrastive Sparse Representation (CSR) models.

        This loss ensures that the sparse encoding can accurately reconstruct the original model embeddings through
        three components:

        1. A primary reconstruction loss (L_k) that measures the error between the original embedding and its
           reconstruction using the top-k sparse components.
        2. A secondary reconstruction loss (L_4k) that measures the error using the top-4k sparse components.
        3. An auxiliary loss (L_aux) that helps to learn residual information.

        Args:
            model: SparseEncoder model with autoencoder components
            beta: Weight for the auxiliary loss component (L_aux)

        References:
            - For more details, see the paper "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation"
              https://huggingface.co/papers/2503.01776

        Requirements:
            1. The model must be configured to output the necessary reconstruction components
            2. Used with SparseEncoder models that implement compositional sparse autoencoding

        Relations:
            - Used as a component within :class:`CSRLoss` combined with a contrastive loss

        Example:
            - This loss is never used standalone, but instead used within the :class:`CSRLoss` class. See that loss for more details.
        """
        super().__init__()
        self.model = model
        self.beta = beta

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "CSRReconstructionLoss is not intended to be used standalone. Use it within CSRLoss instead."
        )

    def compute_loss_from_embeddings(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Compute the CSRReconstruction loss from embeddings.

        Args:
            outputs: List of dictionaries containing sentence embeddings and their sparse representations

        Returns:
            total_loss: The total reconstruction loss value
        """
        # Initialize loss components
        total_L_k = 0.0
        total_L_4k = 0.0
        total_L_aux = 0.0

        # Process each sentence feature
        for features in outputs:
            x = features["sentence_embedding_backbone"]
            recons_k = features["decoded_embedding_k"]
            recons_4k = features["decoded_embedding_4k"]
            recons_aux = features["decoded_embedding_aux"]
            reconsk_pre_bias = features["decoded_embedding_k_pre_bias"]

            # L(k) = ||f(x) - f(dx)_k||₂²
            L_k = F.mse_loss(x, recons_k)

            # L(4k) = ||f(x) - f(dx)_4k||₂²
            L_4k = F.mse_loss(x, recons_4k)

            # L_aux = ||e - ê||₂²
            L_aux = normalized_mean_squared_error(recons_aux, x - reconsk_pre_bias.detach())

            # Accumulate losses
            total_L_k += L_k
            total_L_4k += L_4k
            total_L_aux += L_aux

        # Average losses over batch
        num_columns = len(outputs)
        if num_columns > 0:
            total_L_k /= num_columns
            total_L_4k /= num_columns
            total_L_aux /= num_columns

        # return the total losses as a dictionary, they'll be summed for a final reconstruction loss
        return {
            "reconstruction_loss_k": total_L_k,
            "reconstruction_loss_4k": total_L_4k / 8.0,
            "reconstruction_loss_aux": self.beta * total_L_aux,
        }

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"beta": self.beta}


class CSRLoss(nn.Module):
    def __init__(self, model: SparseEncoder, loss: nn.Module | None = None, beta: float = 0.1, gamma: float = 1.0):
        """
        CSRLoss implements a combined loss function for Contrastive Sparse Representation (CSR) models.

        This loss combines two components:

        1. A reconstruction loss :class:`CSRReconstructionLoss` that ensures the sparse representation can faithfully
            reconstruct the original embedding.
        2. A main loss, which in the paper is a :class:`SparseMultipleNegativesRankingLoss` that ensures semantically
            similar sentences have similar representations.

        The total loss is linear combination of the two losses.

        Args:
            model: SparseEncoder model
            loss: The principal loss function to use can be any of the SparseEncoder losses except flops loss and CSRReconstruction loss.
                If None, the default loss is used, which is the SparseMultipleNegativesRankingLoss.
            beta: Weight for the L_aux component in the reconstruction loss. Default is 0.1.
            gamma: Weight for the main loss component (MNRL a.k.a. InfoNCE by default). Default is 1.0.

        References:
            - For more details, see the paper "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation"
            https://huggingface.co/papers/2503.01776

        Requirements:
            1. Input requirements depend on the chosen loss
            2. Uses autoencoder components of the SparseEncoder model

        Relations:
            - Uses :class:`CSRReconstructionLoss` for the reconstruction component

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("sentence-transformers/all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                        "negative": ["It's quite rainy, sadly.", "She walked to the store."],
                    }
                )
                loss = losses.CSRLoss(model, beta=0.1, gamma=1.0)

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma

        # Initialize the component losses
        self.reconstruction_loss = CSRReconstructionLoss(model=model, beta=beta)
        self.loss = loss if loss is not None else SparseMultipleNegativesRankingLoss(model=model)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]
        sentence_embedding = [output["sentence_embedding"] for output in outputs]

        losses = self.reconstruction_loss.compute_loss_from_embeddings(outputs)
        base_loss = self.loss.compute_loss_from_embeddings(sentence_embedding, labels)
        # Handle the two cases: dictionary of losses or a single loss value
        if isinstance(base_loss, dict):
            for key, value in base_loss.items():
                losses[key] = value * self.gamma
        else:
            losses["base_loss"] = base_loss * self.gamma

        return losses

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"beta": self.beta, "gamma": self.gamma, "loss": self.loss}

    @property
    def citation(self) -> str:
        return """
@misc{wen2025matryoshkarevisitingsparsecoding,
      title={Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation},
      author={Tiansheng Wen and Yifei Wang and Zequn Zeng and Zhong Peng and Yudi Su and Xinyang Liu and Bo Chen and Hongwei Liu and Stefanie Jegelka and Chenyu You},
      year={2025},
      eprint={2503.01776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01776},
}
"""
