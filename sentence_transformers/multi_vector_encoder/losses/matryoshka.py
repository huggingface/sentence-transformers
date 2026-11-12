from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from typing import Any

from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import resolve_gradient_cache_loss, uses_gradient_cache
from sentence_transformers.multi_vector_encoder.losses.token_pooling import MultiVectorTokenPoolingLoss
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.sentence_transformer.losses.matryoshka import CachedLossDecorator, shrink

logger = logging.getLogger(__name__)


class ForwardDecorator:
    """Reuse token embeddings across Matryoshka dimensions."""

    def __init__(self, fn) -> None:
        self.fn = fn

        self.dim = None
        self.cache = []
        self.cache_dim = None
        self.idx = 0

    def set_dim(self, dim) -> None:
        self.dim = dim
        self.idx = 0

    def __call__(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        if self.cache_dim is None or self.cache_dim == self.dim:
            output = self.fn(features, **kwargs)
            self.cache.append(output)
            self.cache_dim = self.dim
        else:
            output = self.cache[self.idx]
        # Descending dimensions make iterative truncation and normalization exact.
        output["token_embeddings"] = shrink(output["token_embeddings"], self.dim)
        if "sentence_embedding" in output:
            output["sentence_embedding"] = shrink(output["sentence_embedding"], self.dim)
        self.idx += 1
        return output


class MultiVectorMatryoshkaLoss(nn.Module):
    def __init__(
        self,
        model: MultiVectorEncoder,
        loss: nn.Module,
        matryoshka_dims: Sequence[int],
        matryoshka_weights: Sequence[float] | Sequence[int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        """
        Train for reduced token embedding dimensions by summing a wrapped loss over dimensions.
        Token embeddings are truncated and L2-normalized before scoring. Apply the same truncation
        and normalization at inference. Cached losses are supported.

        Args:
            model: MultiVectorEncoder model
            loss: The loss function to be used, e.g.
                :class:`MultiVectorMultipleNegativesRankingLoss`,
                :class:`MultiVectorDistillKLDivLoss`, etc.
            matryoshka_dims: A list of token embedding dimensions to be used for the loss function,
                e.g. [128, 96, 64, 32].
            matryoshka_weights: A list of weights to be used for the loss function, e.g.
                [1, 1, 1, 1]. If None, then the weights will be set to 1 for all dimensions.
            n_dims_per_step: The number of dimensions to use per step. If -1, then all dimensions
                are used. If > 0, then a random sample of n_dims_per_step dimensions are used per
                step. The default value is -1.

        References:
            - The concept was introduced in this paper: https://huggingface.co/papers/2205.13147
            - Jina-ColBERT-v2 applied it to late-interaction models: https://huggingface.co/papers/2408.16829
            - `Matryoshka Embeddings <../../../examples/sentence_transformer/training/matryoshka/README.html>`_

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss` is the
              single-vector counterpart.
            - Wrap this loss with :class:`MultiVectorTokenPoolingLoss` so pooling precedes
              truncation, matching ``MultiVectorEncoder.encode``.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
                from sentence_transformers.multi_vector_encoder.losses import (
                    MultiVectorMatryoshkaLoss,
                    MultiVectorMultipleNegativesRankingLoss,
                )

                model = MultiVectorEncoder("answerdotai/ModernBERT-base")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                        "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                        "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                    }
                )
                loss = MultiVectorMultipleNegativesRankingLoss(model)
                loss = MultiVectorMatryoshkaLoss(model, loss, [128, 96, 64, 32])

                trainer = MultiVectorEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        if isinstance(loss, MultiVectorTokenPoolingLoss):
            raise ValueError(
                "MultiVectorTokenPoolingLoss must wrap MultiVectorMatryoshkaLoss so token pooling "
                "happens before truncation, matching MultiVectorEncoder.encode()."
            )
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

        model_embedding_dim = model._get_untruncated_embedding_dimension()
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

        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims: tuple[int, ...]
        self.matryoshka_weights: tuple[float, ...] | tuple[int, ...]
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step

        # Cached backward runs after the forward decorator is removed, so truncate inside the loss.
        self.uses_gradient_cache = uses_gradient_cache(loss)
        if self.uses_gradient_cache:
            gradient_cache_loss = resolve_gradient_cache_loss(loss)
            gradient_cache_loss.calculate_loss = CachedLossDecorator(
                gradient_cache_loss.calculate_loss, self.matryoshka_dims, self.matryoshka_weights, self.n_dims_per_step
            )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        sentence_features = list(sentence_features)
        if self.uses_gradient_cache:
            return self.loss(sentence_features, labels)

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
                decorated_forward.set_dim(self.matryoshka_dims[idx])
                loss += self.matryoshka_weights[idx] * self.loss(sentence_features, labels)
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
