from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    CachedMultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


def quantize_embeddings_torch(embeddings: Tensor, precision: str) -> Tensor:
    """
    Differentiable quantization for PyTorch tensors.
    Uses straight-through estimators to allow gradients to flow.

    Args:
        embeddings: Input tensor to quantize
        precision: Quantization precision ("float32", "int8", "uint8", "binary", "ubinary")

    Returns:
        Quantized tensor (as float32) with gradient support
    """
    if precision == "float32":
        return embeddings

    if precision in ("int8", "uint8"):
        # Compute per-dimension min/max for quantization ranges
        # Using the current batch for range calculation (in practice, you'd want to use calibration data)
        mins = embeddings.min(dim=0, keepdim=True)[0]
        maxs = embeddings.max(dim=0, keepdim=True)[0]

        # Avoid division by zero
        scales = (maxs - mins) / 255.0
        scales = torch.clamp(scales, min=1e-8)

        # Quantize: map to [0, 255]
        normalized = (embeddings - mins) / scales

        # Round with straight-through estimator (forward: round, backward: identity)
        quantized = normalized + (torch.round(normalized) - normalized).detach()

        if precision == "uint8":
            # Map back to original range
            quantized = torch.clamp(quantized, 0, 255)
            dequantized = quantized * scales + mins
        else:  # int8
            # Shift to [-128, 127]
            quantized = quantized - 128
            quantized = torch.clamp(quantized, -128, 127)
            dequantized = (quantized + 128) * scales + mins

        return dequantized

    if precision in ("binary", "ubinary"):
        # Binary quantization: threshold at 0
        # Using sign function with straight-through estimator
        if precision == "binary":  # TODO: should still be 0/+1
            # Signed binary: -1 or +1
            quantized = torch.sign(embeddings)
            # Handle zeros (map to +1)
            quantized = torch.where(quantized == 0, torch.ones_like(quantized), quantized)
        else:  # ubinary
            # Unsigned binary: 0 or +1
            quantized = (embeddings > 0).float()

        # Straight-through: forward uses quantized, backward uses original
        return embeddings + (quantized - embeddings).detach()

    raise ValueError(f"Unsupported precision: {precision}")


class ForwardDecorator:
    """
    This decorator is used to cache the output of the Sentence Transformer's forward pass,
    so that it can be quantized and reused for multiple loss calculations. This prevents the
    model from recalculating the embeddings for each desired quantization precision.

    This decorator is applied to `SentenceTransformer.forward`.
    """

    def __init__(self, fn) -> None:
        self.fn = fn

        self.precision = None
        self.cache = []
        self.caching_mode = True  # First call caches, subsequent calls use cache
        self.idx = 0

    def set_precision(self, precision: str | None) -> None:
        self.precision = precision
        self.idx = 0

    def start_caching(self) -> None:
        """Start caching mode - compute embeddings and store them"""
        self.caching_mode = True
        self.cache = []
        self.idx = 0

    def use_cache(self) -> None:
        """Use cache mode - retrieve embeddings from cache"""
        self.caching_mode = False
        self.idx = 0

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Growing cache (first pass with no quantization):
        if self.caching_mode:
            output = self.fn(features)
            self.cache.append(output)
        # Using cache (subsequent passes with quantization):
        else:
            # Copy to keep original float32 cached output intact for reuse
            output = self.cache[self.idx].copy()

        # Quantize the embeddings if precision is specified
        if self.precision is not None:
            if "token_embeddings" in output:
                output["token_embeddings"] = self._quantize(output["token_embeddings"], self.precision)
            output["sentence_embedding"] = self._quantize(output["sentence_embedding"], self.precision)

        self.idx += 1
        return output

    def _quantize(self, tensor: Tensor, precision: str) -> Tensor:
        """Quantize a tensor to the specified precision (differentiable)"""
        return quantize_embeddings_torch(tensor, precision)


class CachedLossDecorator:
    """
    This decorator is used with the Cached... losses to compute the underlying loss function
    for each quantization precision. This is done by quantizing the pre-computed embeddings
    to the desired precision and then passing them to the underlying loss function once
    for each desired precision.

    This decorator is applied to the `calculate_loss` method of the Cached... losses.
    """

    def __init__(
        self,
        fn,
        quantization_precisions: Sequence[str],
        quantization_weights: Sequence[float] | Sequence[int],
        n_precisions_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.quantization_precisions = quantization_precisions
        self.quantization_weights = quantization_weights
        self.n_precisions_per_step = n_precisions_per_step

    def __call__(self, reps: list[list[Tensor]], *args, **kwargs) -> Tensor:
        # Select which precisions to use this step
        precision_indices = range(len(self.quantization_precisions))
        if self.n_precisions_per_step > 0 and self.n_precisions_per_step < len(precision_indices):
            precision_indices = random.sample(precision_indices, self.n_precisions_per_step)

        # Always compute with original (non-quantized) embeddings first for caching,
        # but only include it in the loss if "float32" is explicitly in the precisions list
        loss = None
        if "float32" in self.quantization_precisions:
            float32_idx = self.quantization_precisions.index("float32")
            weight = self.quantization_weights[float32_idx]
            loss = weight * self.fn(reps, *args, **kwargs)
        else:
            # Just cache, don't include in loss
            _ = self.fn(reps, *args, **kwargs)

        # Now compute loss for each quantized version
        for idx in precision_indices:
            precision = self.quantization_precisions[idx]
            # Skip float32 if we already handled it above
            if precision == "float32":
                continue

            weight = self.quantization_weights[idx]

            # Quantize embeddings
            quantized = [[self._quantize(r, precision) for r in minibatch] for minibatch in reps]
            compute_gradients = torch.is_grad_enabled()

            # we need to detach the quantized embeddings,
            # otherwise the first backward pass of the underlying function will clear the computation graph of the embedding quantization
            if compute_gradients:
                quantized_reps = [[r.detach().requires_grad_() for r in minibatch] for minibatch in quantized]
            else:
                quantized_reps = quantized

            step_loss = weight * self.fn(quantized_reps, *args, **kwargs)
            loss = step_loss if loss is None else loss + step_loss

            # After computing the gradients in minibatches, we need to continue the backward pass through the quantization calculation
            # the gradients must be multiplied with the weights because otherwise the quantization weights are not considered in the backward pass
            if compute_gradients:
                for q_minibatch, d_minibatch in zip(quantized, quantized_reps):
                    for q, d in zip(q_minibatch, d_minibatch):
                        if d.grad is not None:
                            q.backward(weight * d.grad)

        return loss

    def _quantize(self, tensor: Tensor, precision: str) -> Tensor:
        """Quantize a tensor to the specified precision (differentiable)"""
        return quantize_embeddings_torch(tensor, precision)


class QuantizationAwareLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        quantization_precisions: Sequence[Literal["float32", "int8", "uint8", "binary", "ubinary"]],
        quantization_weights: Sequence[float] | Sequence[int] | None = None,
        n_precisions_per_step: int = -1,
    ) -> None:
        """
        The QuantizationAwareLoss can be seen as a loss *modifier* that allows you to use other loss functions with
        various quantization precisions. This is useful for Quantization-Aware Training (QAT), where you want to train
        a model that performs well even after quantization to lower precision formats.

        This loss is also compatible with the Cached... losses, which are in-batch negative losses that allow for
        higher batch sizes. The higher batch sizes allow for more negatives, and often result in a stronger model.

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            quantization_precisions: A list of quantization precisions to be used
                for the loss function, e.g. ["int8", "binary"]. The loss is always
                computed first with the original (float32) embeddings for caching,
                then computed again with each specified quantization precision.
            quantization_weights: A list of weights to be used for the
                loss function, e.g. [1, 1, 1]. If None, then the
                weights will be set to 1 for all precisions.
            n_precisions_per_step: The number of precisions to use per step.
                If -1, then all precisions are used. If > 0, then a
                random sample of n_precisions_per_step precisions are used per
                step. The default value is -1.

        References:
            - Quantization-Aware Training: https://arxiv.org/abs/1712.05877
            - `Quantization <../../../examples/sentence_transformer/training/quantization/README.html>`_

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

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
                loss = losses.QuantizationAwareLoss(
                    model, loss, quantization_precisions=["int8", "binary"]
                )

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

        if not quantization_precisions:
            raise ValueError("You must provide at least one quantization precision in quantization_precisions.")

        # Validate precisions
        valid_precisions = ["float32", "int8", "uint8", "binary", "ubinary"]
        for precision in quantization_precisions:
            if precision not in valid_precisions:
                raise ValueError(
                    f"Invalid precision '{precision}'. Valid precisions are: {', '.join(valid_precisions)}"
                )

        if quantization_weights is None:
            quantization_weights = [1] * len(quantization_precisions)
        elif len(quantization_weights) != len(quantization_precisions):
            raise ValueError("quantization_weights must be the same length as quantization_precisions.")

        self.quantization_precisions = tuple(quantization_precisions)
        self.quantization_weights = tuple(quantization_weights)
        self.n_precisions_per_step = n_precisions_per_step

        # The Cached... losses require a special treatment as their backward pass is incompatible with the
        # ForwardDecorator approach. Instead, we use a CachedLossDecorator to compute the loss for each
        # quantization precision given pre-computed embeddings passed to `calculate_loss`.
        self.cached_losses = (
            CachedMultipleNegativesRankingLoss,
            CachedGISTEmbedLoss,
            CachedMultipleNegativesSymmetricRankingLoss,
        )
        if isinstance(loss, self.cached_losses):
            loss.calculate_loss = CachedLossDecorator(
                loss.calculate_loss, self.quantization_precisions, self.quantization_weights, n_precisions_per_step
            )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> dict[str, Tensor]:
        # For the Cached... losses, the CachedLossDecorator has been applied to the `calculate_loss` method,
        # so we can directly call the loss function.
        if isinstance(self.loss, self.cached_losses):
            return self.loss(sentence_features, labels)

        # Otherwise, we apply the ForwardDecorator to the model's forward pass, which will cache the output
        # embeddings on the first pass, then reuse them for subsequent quantization precisions.
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            # Select which precisions to use this step
            precision_indices = range(len(self.quantization_precisions))
            if self.n_precisions_per_step > 0 and self.n_precisions_per_step < len(precision_indices):
                precision_indices = random.sample(precision_indices, self.n_precisions_per_step)
                precision_indices.sort()

            losses = {}

            # First pass: compute with original (non-quantized) embeddings and cache them
            # Only include float32 loss if it's explicitly in the precisions list
            decorated_forward.start_caching()
            decorated_forward.set_precision(None)  # No quantization on first pass
            loss = self.loss(sentence_features, labels)
            if "float32" in self.quantization_precisions:
                float32_idx = self.quantization_precisions.index("float32")
                weight = self.quantization_weights[float32_idx]
                losses["qat_float32"] = weight * loss

            # Subsequent passes: use cached embeddings with quantization
            decorated_forward.use_cache()
            for idx in precision_indices:
                precision = self.quantization_precisions[idx]
                # Skip float32 if we already handled it above
                if precision == "float32":
                    continue

                weight = self.quantization_weights[idx]
                decorated_forward.set_precision(precision)

                # If the labels seem to be embeddings, quantize them to match the soon-to-be-quantized predicted embeddings
                # This allows for QuantizationAwareLoss with a direct distillation loss
                precision_labels = labels
                if isinstance(labels, Tensor) and len(labels.shape) >= 2:
                    precision_labels = quantize_embeddings_torch(labels, precision)

                step_loss = weight * self.loss(sentence_features, precision_labels)
                losses[f"qat_{precision}"] = step_loss

        finally:
            self.model.forward = original_forward
        return losses

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "quantization_precisions": self.quantization_precisions,
            "quantization_weights": self.quantization_weights,
            "n_precisions_per_step": self.n_precisions_per_step,
        }

    @property
    def citation(self) -> str:
        return """
@article{jacob2018quantization,
    title={Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
    author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
    journal={arXiv preprint arXiv:1712.05877},
    year={2018}
}
"""
