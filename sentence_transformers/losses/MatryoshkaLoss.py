from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    CachedMultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


def shrink(tensor: Tensor, dim: int) -> Tensor:
    tensor_dim = tensor.shape[-1]
    if dim > tensor_dim:
        raise ValueError(
            f"Dimension {dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
        )
    tensor = tensor[..., :dim]
    tensor = F.normalize(tensor, p=2, dim=-1)
    return tensor


class ForwardDecorator:
    """
    This decorator is used to cache the output of the Sentence Transformer's forward pass,
    so that it can be shrank and reused for multiple loss calculations. This prevents the
    model from recalculating the embeddings for each desired Matryoshka dimensionality.

    This decorator is applied to `SentenceTransformer.forward`.
    """

    def __init__(self, fn) -> None:
        self.fn = fn

        self.dim = None
        self.cache = []
        self.cache_dim = None
        self.idx = 0

    def set_dim(self, dim) -> None:
        self.dim = dim
        self.idx = 0

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Growing cache:
        if self.cache_dim is None or self.cache_dim == self.dim:
            output = self.fn(features)
            self.cache.append(output)
            self.cache_dim = self.dim
        # Using cache:
        else:
            output = self.cache[self.idx]
        if "token_embeddings" in output:
            output["token_embeddings"] = shrink(output["token_embeddings"], self.dim)
        output["sentence_embedding"] = shrink(output["sentence_embedding"], self.dim)
        self.idx += 1
        return output


class CachedLossDecorator:
    """
    This decorator is used with the Cached... losses to compute the underlying loss function
    for each Matryoshka dimensionality. This is done by shrinking the pre-computed embeddings
    to the desired dimensionality and then passing them to the underlying loss function once
    for each desired dimensionality.

    This decorator is applied to the `calculate_loss` method of the Cached... losses.
    """

    def __init__(
        self,
        fn,
        matryoshka_dims: Sequence[int],
        matryoshka_weights: Sequence[float] | Sequence[int],
        n_dims_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        self.n_dims_per_step = n_dims_per_step

    def __call__(self, reps: list[list[Tensor]], *args, **kwargs) -> Tensor:
        dim_indices = range(len(self.matryoshka_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)

        loss = 0.0
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]

            truncated = [[shrink(r, dim) for r in minibatch] for minibatch in reps]
            compute_gradients = torch.is_grad_enabled()
            # we need to detach the truncated embeddings,
            # otherwise the first backward pass of the underlying function will clear the computation graph of the embedding truncation
            if compute_gradients:
                matryoshka_reps = [[r.detach().requires_grad_() for r in minibatch] for minibatch in truncated]
            else:
                matryoshka_reps = truncated
            loss += weight * self.fn(matryoshka_reps, *args, **kwargs)
            # After computing the gradients in minibatches, we need to continue the backward pass through the truncation calculation
            # the gradients must be multiplied with the weights because otherwise the matryoshka weights are not considered in the backward pass
            if compute_gradients:
                for t_minibatch, d_minibatch in zip(truncated, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        t.backward(weight * d.grad)
        return loss


class MatryoshkaLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_dims: Sequence[int],
        matryoshka_weights: Sequence[float] | Sequence[int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        """
        The MatryoshkaLoss can be seen as a loss *modifier* that allows you to use other loss functions at various
        different embedding dimensions. This is useful for when you want to train a model where users have the option
        to lower the embedding dimension to improve their embedding comparison speed and costs.

        This loss is also compatible with the Cached... losses, which are in-batch negative losses that allow for
        higher batch sizes. The higher batch sizes allow for more negatives, and often result in a stronger model.

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            matryoshka_dims: A list of embedding dimensions to be used
                for the loss function, e.g. [768, 512, 256, 128, 64].
            matryoshka_weights: A list of weights to be used for the
                loss function, e.g. [1, 1, 1, 1, 1]. If None, then the
                weights will be set to 1 for all dimensions.
            n_dims_per_step: The number of dimensions to use per step.
                If -1, then all dimensions are used. If > 0, then a
                random sample of n_dims_per_step dimensions are used per
                step. The default value is -1.

        References:
            - The concept was introduced in this paper: https://huggingface.co/papers/2205.13147
            - `Matryoshka Embeddings <../../../examples/sentence_transformer/training/matryoshka/README.html>`_

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`Matryoshka2dLoss` uses this loss in combination with :class:`AdaptiveLayerLoss` which allows for
                layer reduction for faster inference.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)
                loss = losses.MatryoshkaLoss(model, loss, [768, 512, 256, 128, 64])

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.loss = loss

        if not matryoshka_dims:
            raise ValueError("You must provide at least one dimension in matryoshka_dims.")
        if any(dim <= 0 for dim in matryoshka_dims):
            raise ValueError("All dimensions passed to a matryoshka loss must be > 0.")
        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)
        elif len(matryoshka_weights) != len(matryoshka_dims):
            raise ValueError("matryoshka_weights must be the same length as matryoshka_dims.")

        model_embedding_dim = model.get_sentence_embedding_dimension()
        if model_embedding_dim is not None:
            if any(d > model_embedding_dim for d in matryoshka_dims):
                raise ValueError(
                    f"Dimensions in matryoshka_dims cannot exceed the model's embedding dimension of {model_embedding_dim}."
                )
            if model_embedding_dim not in matryoshka_dims:
                logger.warning(
                    f"The model's embedding dimension {model_embedding_dim} is not included in matryoshka_dims: {matryoshka_dims}. "
                    "This means that the full model dimension won't be trained, which may lead to degraded performance "
                    "when using the model without specifying a lower truncation dimension. It is strongly recommended to include "
                    f"{model_embedding_dim} in matryoshka_dims."
                )

        # Sort the dimensions and weights in descending order
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims: tuple[int, ...]
        self.matryoshka_weights: tuple[float, ...] | tuple[int, ...]
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step

        # The Cached... losses require a special treatment as their backward pass is incompatible with the
        # ForwardDecorator approach. Instead, we use a CachedLossDecorator to compute the loss for each
        # Matryoshka dimensionality given pre-computed embeddings passed to `calculate_loss`.
        self.cached_losses = (
            CachedMultipleNegativesRankingLoss,
            CachedGISTEmbedLoss,
            CachedMultipleNegativesSymmetricRankingLoss,
        )
        if isinstance(loss, self.cached_losses):
            loss.calculate_loss = CachedLossDecorator(
                loss.calculate_loss, self.matryoshka_dims, self.matryoshka_weights
            )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # For the Cached... losses, the CachedLossDecorator has been applied to the `calculate_loss` method,
        # so we can directly call the loss function.
        if isinstance(self.loss, self.cached_losses):
            return self.loss(sentence_features, labels)

        # Otherwise, we apply the ForwardDecorator to the model's forward pass, which will cache the output
        # embeddings for each Matryoshka dimensionality, allowing it to be reused for the smaller dimensions.
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
                dim_indices = random.sample(dim_indices, self.n_dims_per_step)
                dim_indices.sort()

            loss = 0.0
            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)
                # If the labels seem to be embeddings, truncate them to match the soon-to-be-truncated predicted embeddings
                # This allows for MatryoshkaLoss with a direct distillation loss
                dim_labels = labels
                if (
                    isinstance(labels, torch.Tensor)
                    and labels.ndim == 2
                    and labels.size(-1) == self.model.get_sentence_embedding_dimension()
                ):
                    dim_labels = labels[:, :dim]

                loss += weight * self.loss(sentence_features, dim_labels)
        finally:
            self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }

    @property
    def citation(self) -> str:
        return """
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
