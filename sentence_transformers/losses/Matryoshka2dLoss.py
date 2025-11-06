from __future__ import annotations

from typing import Any

from torch.nn import Module

from sentence_transformers.SentenceTransformer import SentenceTransformer

from .AdaptiveLayerLoss import AdaptiveLayerLoss
from .MatryoshkaLoss import MatryoshkaLoss


class Matryoshka2dLoss(AdaptiveLayerLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float | int] | None = None,
        n_layers_per_step: int = 1,
        n_dims_per_step: int = 1,
        last_layer_weight: float = 1.0,
        prior_layers_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ) -> None:
        """
        The Matryoshka2dLoss can be seen as a loss *modifier* that combines the :class:`AdaptiveLayerLoss` and the
        :class:`MatryoshkaLoss`. This allows you to train an embedding model that 1) allows users to specify the number
        of model layers to use, and 2) allows users to specify the output dimensions to use.

        The former is useful for when you want users to have the option to lower the number of layers used to improve
        their inference speed and memory usage, and the latter is useful for when you want users to have the option to
        lower the output dimensions to improve the efficiency of their downstream tasks (e.g. retrieval) or to lower
        their storage costs.

        Note, this uses `n_layers_per_step=1` and `n_dims_per_step=1` as default, following the original 2DMSE
        implementation.

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
            n_layers_per_step: The number of layers to use per step. If
                -1, then all layers are used. If > 0, then a random
                sample of n_layers_per_step layers are used per step.
                The 2DMSE paper uses `n_layers_per_step=1`. The default
                value is -1.
            n_dims_per_step: The number of dimensions to use per step.
                If -1, then all dimensions are used. If > 0, then a
                random sample of n_dims_per_step dimensions are used per
                step. The default value is -1.
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
            - See the 2D Matryoshka Sentence Embeddings (2DMSE) paper: https://huggingface.co/papers/2402.14776
            - `Matryoshka Embeddings <../../../examples/sentence_transformer/training/matryoshka/README.html>`_
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
            - :class:`MatryoshkaLoss` is used in this loss, and it is responsible for the dimensionality reduction.
            - :class:`AdaptiveLayerLoss` is used in this loss, and it is responsible for the layer reduction.

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
                loss = losses.Matryoshka2dLoss(model, loss, [768, 512, 256, 128, 64])

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        matryoshka_loss = MatryoshkaLoss(
            model,
            loss,
            matryoshka_dims,
            matryoshka_weights=matryoshka_weights,
            n_dims_per_step=n_dims_per_step,
        )
        super().__init__(
            model,
            matryoshka_loss,
            n_layers_per_step=n_layers_per_step,
            last_layer_weight=last_layer_weight,
            prior_layers_weight=prior_layers_weight,
            kl_div_weight=kl_div_weight,
            kl_temperature=kl_temperature,
        )

    def get_config_dict(self) -> dict[str, Any]:
        return {
            **super().get_config_dict(),
            **self.loss.get_config_dict(),
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
