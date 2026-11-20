from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any

from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns
from sentence_transformers.sparse_encoder.model import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder
from sentence_transformers.util import select_max_active_dims


class SparseMatryoshkaLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        loss: nn.Module,
        max_active_dims: Sequence[int | None],
        query_max_active_dims: Sequence[int | None] | None = None,
        matryoshka_weights: Sequence[float] | Sequence[int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        """
        The SparseMatryoshkaLoss is a loss *modifier* that computes another loss at several sparsity budgets.
        For every budget, each embedding is pruned to its ``max_active_dims`` largest values, as
        :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>` does with its
        ``max_active_dims`` argument, and the underlying loss is computed on the pruned embeddings. This trains
        the model for the setting where embeddings are pruned at inference time to reduce index size and
        retrieval latency, instead of only for the unpruned embeddings.

        Unlike :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss`, which keeps the first
        dimensions of a dense embedding, this loss keeps the largest values of each sparse embedding, and it does
        not normalize the result. Gradients only flow through the values that are kept.

        Use this as the ``loss`` of a :class:`SpladeLoss` or :class:`CachedSpladeLoss`. Only the underlying
        loss sees the additional pruning. The wrapper's regularizers are unchanged.

        Args:
            model: SparseEncoder model
            loss: The loss function to compute at every budget. It must have a ``compute_loss_from_embeddings``
                method, like the losses that :class:`SpladeLoss` accepts.
            max_active_dims: A list of numbers of active dimensions to keep in each embedding, e.g.
                [None, 256, 128, 64]. ``None`` applies no additional pruning. Applied to every input column,
                unless ``query_max_active_dims`` is given.
            query_max_active_dims: The numbers of active dimensions to keep in the embeddings of the first
                input column (the queries), paired by position with ``max_active_dims``, which then only applies
                to the other columns (the documents). If None, ``max_active_dims`` is used for every column.
            matryoshka_weights: One weight per entry of ``max_active_dims``. Defaults to ``1 / len(max_active_dims)``
                for every entry, averaging the losses to preserve their scale relative to the regularizers.
                Explicit weights are used as given, without normalization.
            n_dims_per_step: The number of entries of ``max_active_dims`` to use per step. If -1, then all
                entries are used. If > 0, then a random sample of n_dims_per_step entries are used per step.
                Sampled losses are multiplied by the total number of budgets divided by the sample size,
                preserving the full weighted objective in expectation. The default value is -1.

        References:
            - The concept was introduced in this paper: https://huggingface.co/papers/2205.13147

        Requirements:
            1. The input requirements are those of the underlying loss.
            2. For sparse autoencoders, set ``k`` to at least the largest budget.

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss` is the equivalent of
              this loss for dense embedding models, truncating the embedding dimension instead.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                    }
                )
                base_loss = losses.SparseMultipleNegativesRankingLoss(model)
                matryoshka_loss = losses.SparseMatryoshkaLoss(
                    model,
                    loss=base_loss,
                    max_active_dims=[None, 256, 128, 64],
                    query_max_active_dims=[None, 32, 16, 8],
                )
                loss = losses.SpladeLoss(
                    model,
                    loss=matryoshka_loss,
                    document_regularizer_weight=3e-5,
                    query_regularizer_weight=5e-5,
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        if not hasattr(loss, "compute_loss_from_embeddings"):
            raise ValueError(
                "The provided loss does not have a 'compute_loss_from_embeddings' method, which is required for "
                "SparseMatryoshkaLoss. This method must have the signature "
                "`compute_loss_from_embeddings(embeddings: List[Tensor], labels: Tensor | None = None)`."
            )
        if not max_active_dims:
            raise ValueError("You must provide at least one dimension in max_active_dims.")
        if query_max_active_dims is None:
            query_max_active_dims = max_active_dims
        elif len(query_max_active_dims) != len(max_active_dims):
            raise ValueError("query_max_active_dims must be the same length as max_active_dims.")
        if matryoshka_weights is None:
            matryoshka_weights = [1 / len(max_active_dims)] * len(max_active_dims)
        elif len(matryoshka_weights) != len(max_active_dims):
            raise ValueError("matryoshka_weights must be the same length as max_active_dims.")
        if any(dim is not None and dim <= 0 for dim in (*max_active_dims, *query_max_active_dims)):
            raise ValueError("All dimensions passed to a matryoshka loss must be positive or None.")

        model_embedding_dim = model.get_embedding_dimension()
        if model_embedding_dim is not None:
            for name, dims in (("max_active_dims", max_active_dims), ("query_max_active_dims", query_max_active_dims)):
                if any(dim is not None and dim > model_embedding_dim for dim in dims):
                    raise ValueError(
                        f"Dimensions in {name} cannot exceed the model's embedding dimension of {model_embedding_dim}."
                    )
        output_module = model[-1]
        if isinstance(output_module, SparseAutoEncoder) and any(
            dim is not None and dim > output_module.k for dim in (*max_active_dims, *query_max_active_dims)
        ):
            raise ValueError(
                f"SparseAutoEncoder already prunes embeddings to k={output_module.k}. "
                "Set k to at least the largest requested budget before constructing SparseMatryoshkaLoss."
            )

        self.model = model
        self.loss = loss

        self.max_active_dims = tuple(max_active_dims)
        self.query_max_active_dims = tuple(query_max_active_dims)
        self.matryoshka_weights = tuple(matryoshka_weights)
        self.n_dims_per_step = n_dims_per_step

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> Tensor | dict[str, Tensor]:
        embeddings = embed_columns(self.model, sentence_features)
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(
        self, embeddings: list[Tensor], labels: Tensor | None = None
    ) -> Tensor | dict[str, Tensor]:
        dim_indices = range(len(self.max_active_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)

        total: Tensor | dict[str, Tensor] | None = None
        for idx in dim_indices:
            pruned = [select_max_active_dims(embeddings[0], self.query_max_active_dims[idx])]
            pruned += [select_max_active_dims(embedding, self.max_active_dims[idx]) for embedding in embeddings[1:]]
            loss = self.loss.compute_loss_from_embeddings(pruned, labels)
            weight = self.matryoshka_weights[idx] * len(self.max_active_dims) / len(dim_indices)
            if isinstance(loss, dict):
                total = total or {}
                for key, value in loss.items():
                    total[key] = total[key] + weight * value if key in total else weight * value
            else:
                total = weight * loss if total is None else total + weight * loss
        return total

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "max_active_dims": self.max_active_dims,
            "query_max_active_dims": self.query_max_active_dims,
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
