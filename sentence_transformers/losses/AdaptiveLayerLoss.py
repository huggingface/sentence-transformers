from __future__ import annotations

import random
import warnings
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss import (
    CachedMultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.models import Transformer


class TransformerDecorator:
    """
    Decorator that caches the embeddings of all layers of the transformer.
    When `layer_idx` is set, it returns the cached embeddings of that layer instead.

    This is meant to override the forward function of the Transformer.
    """

    def __init__(self, transformer: Transformer, original_forward) -> None:
        self.transformer = transformer
        self.original_forward = original_forward
        self.layer_idx = None

    def set_layer_idx(self, layer_idx) -> None:
        self.layer_idx = layer_idx

    def __call__(self, features) -> dict[str, Tensor]:
        if self.layer_idx is None:
            return self.call_grow_cache(features)
        return self.call_use_cache(features)

    def call_grow_cache(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Temporarily sets the output_hidden_states to True, runs the model, and then restores the original setting.
        Use the all_layer_embeddings to get the embeddings of all layers.
        """
        original_output_hidden_states = self.transformer.auto_model.config.output_hidden_states
        try:
            self.transformer.auto_model.config.output_hidden_states = True
            output = self.original_forward(features)
            self.num_layers = len(output["all_layer_embeddings"])
        finally:
            # Restore original setting
            self.transformer.auto_model.config.output_hidden_states = original_output_hidden_states

        return output

    def call_use_cache(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        return {**features, "token_embeddings": features["all_layer_embeddings"][self.layer_idx]}


class ForwardDecorator:
    """
    Decorator that caches the embeddings after all modules (e.g. pooling) of the model.
    Required to get the embeddings after all modules for the KL-divergence loss.

    This is meant to override the forward function of the SentenceTransformer.
    """

    def __init__(self, fn) -> None:
        self.fn = fn
        self.embeddings = []

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.fn(features)
        self.embeddings.append(output["sentence_embedding"])
        return output

    def get_embeddings(self) -> Tensor:
        embeddings = torch.concat(self.embeddings, dim=0)
        self.embeddings = []
        return embeddings


class AdaptiveLayerLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        n_layers_per_step: int = 1,
        last_layer_weight: float = 1.0,
        prior_layers_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ) -> None:
        """
        The AdaptiveLayerLoss can be seen as a loss *modifier* that allows you to use other loss functions at non-final
        layers of the Sentence Transformer model. This is useful for when you want to train a model where users have
        the option to lower the number of layers used to improve their inference speed and memory usage.

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            n_layers_per_step: The number of layers to use per step. If
                -1, then all layers are used. If > 0, then a random
                sample of `n_layers_per_step` layers are used per step,
                separate from the final layer, which is always used. The
                2DMSE paper uses `n_layers_per_step=1`. The default
                value is 1.
            last_layer_weight: The weight to use for the loss of the
                final layer. Increase this to focus more on the
                performance when using all layers. The default value is
                1.0.
            prior_layers_weight: The weight to use for the loss of the
                prior layers. Increase this to focus more on the
                performance when using fewer layers. The default value
                is 1.0.
            kl_div_weight: The weight to use for the KL-divergence loss
                that is used to make the prior layers match that of the
                last layer. Increase this to focus more on the
                performance when using fewer layers. The default value
                is 1.0.
            kl_temperature: The temperature to use for the KL-divergence
                loss. If 0, then the KL-divergence loss is not used. The
                default value is 1.0.

        References:
            - The concept was inspired by the 2DMSE paper: https://huggingface.co/papers/2402.14776
            - `Adaptive Layers <../../../examples/sentence_transformer/training/adaptive_layer/README.html>`_

        Requirements:
            1. The base loss cannot be :class:`CachedMultipleNegativesRankingLoss`,
               :class:`CachedMultipleNegativesSymmetricRankingLoss`, or :class:`CachedGISTEmbedLoss`.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`Matryoshka2dLoss` uses this loss in combination with :class:`MatryoshkaLoss` which allows for
                output dimensionality reduction for faster downstream tasks (e.g. retrieval).

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model=model)
                loss = losses.AdaptiveLayerLoss(model, loss)

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
        self.n_layers_per_step = n_layers_per_step
        self.last_layer_weight = last_layer_weight
        self.prior_layers_weight = prior_layers_weight
        self.kl_div_weight = kl_div_weight
        self.kl_temperature = kl_temperature
        assert isinstance(self.model[0], Transformer)
        if isinstance(
            loss,
            (CachedMultipleNegativesRankingLoss, CachedMultipleNegativesSymmetricRankingLoss, CachedGISTEmbedLoss),
        ):
            warnings.warn(f"MatryoshkaLoss is not compatible with {loss.__class__.__name__}.", stacklevel=2)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Decorate the forward function of the transformer to cache the embeddings of all layers
        original_transformer_forward = self.model[0].forward
        transformer_decorator = TransformerDecorator(self.model[0], original_transformer_forward)
        self.model[0].forward = transformer_decorator

        # Decorate the forward function of the model to get the embeddings after all modules (e.g. pooling)
        original_forward = self.model.forward
        forward_decorator = ForwardDecorator(original_forward)
        self.model.forward = forward_decorator

        # Run the loss normally: i.e. the final layer, but 1) use the transformers decorator to cache
        # the embeddings of all layers and 2) use the forward decorator to get the embeddings after all modules
        # for the KL-divergence loss
        loss = self.loss(sentence_features, labels) * self.last_layer_weight
        if self.kl_temperature > 0:
            final_embeddings = forward_decorator.get_embeddings()
            final_embeddings = F.softmax(final_embeddings / self.kl_temperature, dim=-1)

        num_layers = transformer_decorator.num_layers
        layer_indices = range(num_layers - 1)
        if self.n_layers_per_step > 0 and self.n_layers_per_step < num_layers - 1:
            layer_indices = random.sample(layer_indices, self.n_layers_per_step)

        # This loop is over `num_layer - 1` layers because we already computed the loss over the final layer
        for layer_idx in layer_indices:
            # Add regular loss for each layer by using the cached embeddings of that layer
            transformer_decorator.set_layer_idx(layer_idx)
            layer_loss = self.loss(sentence_features, labels)
            loss = loss + layer_loss / (1 + layer_idx) / len(layer_indices) * self.prior_layers_weight

            # and KL-divergence loss between the current layer and the final layer
            # Note: we use "batchmean" reduction as that aligns with the mathematical definition
            if self.kl_temperature > 0:
                embeddings = forward_decorator.get_embeddings()
                kl_div_loss = F.kl_div(
                    F.log_softmax(embeddings / self.kl_temperature, dim=-1),
                    final_embeddings,
                    reduction="batchmean",
                )
                loss = loss + kl_div_loss * self.kl_temperature * self.kl_div_weight

        self.model[0].forward = original_transformer_forward
        self.model.forward = original_forward

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "n_layers_per_step": self.n_layers_per_step,
            "last_layer_weight": self.last_layer_weight,
            "prior_layers_weight": self.prior_layers_weight,
            "kl_div_weight": self.kl_div_weight,
            "kl_temperature": self.kl_temperature,
        }

    @property
    def citation(self) -> str:
        return """
@misc{li20242d,
    title={2D Matryoshka Sentence Embeddings},
    author={Xianming Li and Zongxi Li and Jing Li and Haoran Xie and Qing Li},
    year={2024},
    eprint={2402.14776},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""
