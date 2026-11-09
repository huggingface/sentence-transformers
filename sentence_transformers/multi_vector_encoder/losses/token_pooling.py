from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import resolve_gradient_cache_loss, uses_gradient_cache
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling


def _pooling_name(pooling: nn.Module | None) -> str | None:
    """Describe a pooling variant for configs and model cards."""
    if pooling is None:
        return None
    config = pooling.get_config_dict() if hasattr(pooling, "get_config_dict") else {}
    if not config:
        return pooling.__class__.__name__
    return f"{pooling.__class__.__name__}({', '.join(f'{key}={value!r}' for key, value in config.items())})"


class ForwardDecorator:
    """Reuse token embeddings and tasks across pooling variants."""

    def __init__(self, fn) -> None:
        self.fn = fn

        self.pooling: BaseTokenPooling | None = None
        self.cache: list[tuple[dict[str, Tensor], str | None]] = []
        self.cache_pooling: BaseTokenPooling | None = None
        self.idx = 0

    def set_pooling(self, pooling: BaseTokenPooling | None) -> None:
        self.pooling = pooling
        self.idx = 0

    def __call__(self, features: dict[str, Tensor], task: str | None = None, **kwargs) -> dict[str, Tensor]:
        if not self.cache or self.cache_pooling is self.pooling:
            output = self.fn(features, task=task, **kwargs)
            self.cache.append((output, task))
            self.cache_pooling = self.pooling
        else:
            output, task = self.cache[self.idx]
        self.idx += 1
        # Pooling mutates the output dict, so preserve the cached version.
        output = dict(output)
        return self.pooling(output, task=task) if self.pooling is not None else output


class CachedLossDecorator:
    """Apply pooling variants to cached embeddings before loss calculation."""

    def __init__(
        self,
        fn,
        token_poolings: Sequence[BaseTokenPooling | None],
        weights: Sequence[float] | Sequence[int],
        n_poolings_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.token_poolings = token_poolings
        self.weights = weights
        self.n_poolings_per_step = n_poolings_per_step
        self.column_tasks: list[str | None] | None = None

    def __call__(self, reps: list[list[Tensor]], masks_chunks: list[list[Tensor]], *args, **kwargs) -> Tensor:
        pooling_indices = range(len(self.token_poolings))
        if self.n_poolings_per_step > 0 and self.n_poolings_per_step < len(pooling_indices):
            pooling_indices = random.sample(pooling_indices, self.n_poolings_per_step)

        column_tasks = self.column_tasks
        if column_tasks is None:
            column_tasks = ["query"] + ["document"] * (len(reps) - 1)

        compute_gradients = torch.is_grad_enabled()
        loss = 0.0
        for pooling_idx in pooling_indices:
            pooling = self.token_poolings[pooling_idx]
            weight = self.weights[pooling_idx]

            pooled_reps: list[list[Tensor]] = []
            pooled_masks: list[list[Tensor]] = []
            for task, rep_mbs, mask_mbs in zip(column_tasks, reps, masks_chunks):
                pooled_rep_mbs: list[Tensor] = []
                pooled_mask_mbs: list[Tensor] = []
                for rep_mb, mask_mb in zip(rep_mbs, mask_mbs):
                    pooled = {"token_embeddings": rep_mb, "attention_mask": mask_mb}
                    if pooling is not None:
                        pooled = pooling(pooled, task=task)
                    pooled_rep_mbs.append(pooled["token_embeddings"])
                    pooled_mask_mbs.append(pooled["attention_mask"].bool())
                pooled_reps.append(pooled_rep_mbs)
                pooled_masks.append(pooled_mask_mbs)

            # Detach so the cached loss's backward preserves the pooling graph.
            if compute_gradients:
                variant_reps = [[r.detach().requires_grad_() for r in mbs] for mbs in pooled_reps]
            else:
                variant_reps = pooled_reps
            loss += weight * self.fn(variant_reps, pooled_masks, *args, **kwargs)
            if compute_gradients:
                for pooled_mbs, variant_mbs in zip(pooled_reps, variant_reps):
                    for pooled_mb, variant_mb in zip(pooled_mbs, variant_mbs):
                        pooled_mb.backward(weight * variant_mb.grad)
        return loss


class MultiVectorTokenPoolingLoss(nn.Module):
    def __init__(
        self,
        model: MultiVectorEncoder,
        loss: nn.Module,
        token_poolings: Sequence[BaseTokenPooling | None],
        *,
        weights: Sequence[float] | Sequence[int] | None = None,
        n_poolings_per_step: int = -1,
    ) -> None:
        """
        Train for token pooling by summing a wrapped loss over pooling variants. Input embeddings
        are computed once, and pooling applies to documents only by default. Cached losses are
        supported. Hierarchical pooling uses straight-through gradients through cluster means.

        Args:
            model: MultiVectorEncoder model
            loss: The loss function to be used, e.g.
                :class:`MultiVectorMultipleNegativesRankingLoss`,
                :class:`MultiVectorDistillKLDivLoss`, etc.
            token_poolings: Pooling modules, one per variant. Include None to also train unpooled
                retrieval. Custom modules must preserve gradients through the input embeddings.
                Cached losses require poolers without trainable parameters.
            weights: A list of weights for the per-variant losses, summed like the Matryoshka
                losses do. If None, then the weights will be set to 1 for all variants.
            n_poolings_per_step: Number of randomly sampled variants per step. Defaults to -1
                (all variants).

        References:
            - Learn to Pool, the concept this loss implements: https://stefan-jo.github.io/learn-to-pool
            - LightOn applied the recipe to LateOn models: https://huggingface.co/blog/lightonai/lateon-hpool-regularization

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Recommendations:
            - Wrap a model without a pooling module in its pipeline. Pooling is not idempotent, so
              a baked-in pooling module would compound with every variant.

        Relations:
            - Wrap :class:`MultiVectorMatryoshkaLoss` with this loss so pooling
              precedes truncation, matching ``MultiVectorEncoder.encode``.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
                from sentence_transformers.multi_vector_encoder.losses import (
                    MultiVectorMultipleNegativesRankingLoss,
                    MultiVectorTokenPoolingLoss,
                )
                from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

                model = MultiVectorEncoder("answerdotai/ModernBERT-base")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                        "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                        "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                    }
                )
                loss = MultiVectorMultipleNegativesRankingLoss(model)
                token_poolings = [None, HierarchicalTokenPooling(pool_factor=2), HierarchicalTokenPooling(pool_factor=4)]
                loss = MultiVectorTokenPoolingLoss(model, loss, token_poolings=token_poolings)

                trainer = MultiVectorEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.loss = loss

        if not token_poolings:
            raise ValueError("You must provide at least one token pooling in token_poolings.")
        for pooling in token_poolings:
            if pooling is not None and not isinstance(pooling, BaseTokenPooling):
                raise ValueError(
                    f"Every entry of token_poolings must be a BaseTokenPooling module or None, "
                    f"got {type(pooling).__name__}."
                )

        if weights is None:
            weights = [1] * len(token_poolings)
        elif len(weights) != len(token_poolings):
            raise ValueError("weights must have one weight per pooling variant.")

        self.token_poolings = nn.ModuleList()
        for idx, pooling in enumerate(token_poolings):
            self.token_poolings.add_module(str(idx), pooling)
        self.weights: tuple[float, ...] | tuple[int, ...] = tuple(weights)
        self.n_poolings_per_step = n_poolings_per_step

        # Cached backward runs after the forward decorator is removed, so pool inside the loss.
        self.uses_gradient_cache = uses_gradient_cache(loss)
        if self.uses_gradient_cache:
            if any(parameter.requires_grad for parameter in self.token_poolings.parameters()):
                raise ValueError(
                    "Cached losses do not support token_poolings with trainable parameters. "
                    "Freeze the pooler with pooling.requires_grad_(False), or use a non-cached loss."
                )
            gradient_cache_loss = resolve_gradient_cache_loss(loss)
            self._cached_loss_decorator = CachedLossDecorator(
                gradient_cache_loss.calculate_loss, self.token_poolings, self.weights, self.n_poolings_per_step
            )
            gradient_cache_loss.calculate_loss = self._cached_loss_decorator

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        sentence_features = list(sentence_features)
        if self.uses_gradient_cache:
            self._cached_loss_decorator.column_tasks = [
                sentence_feature.get("task", "query" if idx == 0 else "document")
                for idx, sentence_feature in enumerate(sentence_features)
            ]
            return self.loss(sentence_features, labels)

        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            pooling_indices = range(len(self.token_poolings))
            if self.n_poolings_per_step > 0 and self.n_poolings_per_step < len(pooling_indices):
                pooling_indices = random.sample(pooling_indices, self.n_poolings_per_step)
                pooling_indices.sort()

            loss = 0.0
            for idx in pooling_indices:
                decorated_forward.set_pooling(self.token_poolings[idx])
                loss += self.weights[idx] * self.loss(sentence_features, labels)
        finally:
            self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "token_poolings": [_pooling_name(pooling) for pooling in self.token_poolings],
            "weights": self.weights,
            "n_poolings_per_step": self.n_poolings_per_step,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{josef2026learntopool,
    title={Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression},
    author={Stefan Josef},
    booktitle={1st Late Interaction Workshop (LIR) at ECIR 2026},
    year={2026},
}
"""
