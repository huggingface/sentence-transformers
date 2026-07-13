from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.losses.gradcache import (
    CachedLossMixin,
    reconstruct_loss_components,
)
from sentence_transformers.sparse_encoder.losses.splade import SpladeLoss
from sentence_transformers.sparse_encoder.model import SparseEncoder

logger = logging.getLogger(__name__)


class CachedSpladeLoss(CachedLossMixin, SpladeLoss):
    def __init__(
        self,
        model: SparseEncoder,
        loss: nn.Module,
        document_regularizer_weight: float,
        query_regularizer_weight: float | None = None,
        document_regularizer: nn.Module | None = None,
        query_regularizer: nn.Module | None = None,
        document_regularizer_threshold: int | None = None,
        query_regularizer_threshold: int | None = None,
        use_document_regularizer_only: bool = False,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        """
        Cached version of :class:`SpladeLoss` that uses the GradCache technique to allow for much larger
        effective batch sizes without additional GPU memory usage.

        By performing the GradCache mini-batch embedding at the SpladeLoss level, both the base loss and
        regularizers still receive pre-computed embeddings via ``compute_loss_from_embeddings()``, no
        changes to base losses or regularizers are needed.

        In detail, the GradCache technique works as follows:

            (1) A quick embedding step without gradients/computation graphs to get all embeddings in mini-batches;
            (2) Calculate the combined loss (base + regularizers), backward up to the embeddings and cache the
                gradients w.r.t. the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the
                backward chain.

        Args:
            model: SparseEncoder model
            loss: The principal loss function to use can be any of the SparseEncoder losses except CSR related
                losses and flops loss. Must have a ``compute_loss_from_embeddings`` method.
            document_regularizer_weight: Weight for the document regularization term. This term encourages sparsity
                in the document embeddings. In some papers, this parameter is referred to as "lambda_d" (document)
                or "lambda_c" (corpus).
            query_regularizer_weight: Weight for the query regularization term. This term encourages sparsity in
                the query embeddings. If None, no query regularization will be applied. In some papers, this
                parameter is referred to as "lambda_q" (query).
            document_regularizer: Optional regularizer to use specifically for document regularization instead of the
                default FlopsLoss.
            query_regularizer: Optional regularizer to use specifically for query regularization instead of the
                default FlopsLoss.
            document_regularizer_threshold: Optional threshold for the number of non-zero (active) elements in the
                document embeddings to be considered in the FlopsLoss.
            query_regularizer_threshold: Optional threshold for the number of non-zero (active) elements in the
                query embeddings to be considered in the FlopsLoss.
            use_document_regularizer_only: If True, all input embeddings are treated as documents and regularized
                together with document_regularizer_weight.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used
                during training and evaluation. The larger the mini-batch size, the more memory efficient the
                training is, but the slower the training will be. It's recommended to set it as high as your GPU
                memory allows. The default value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training.

        References:
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup:
              https://huggingface.co/papers/2101.06983
            - From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective:
              https://huggingface.co/papers/2205.04733

        Requirements:
            1. Input requirements depend on the chosen loss
            2. Should be used with large ``per_device_train_batch_size`` and low ``mini_batch_size`` for superior
               performance, but slower training time than :class:`SpladeLoss`.

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
                loss = losses.CachedSpladeLoss(
                    model=model,
                    loss=losses.SparseMultipleNegativesRankingLoss(model),
                    document_regularizer_weight=3e-5,
                    query_regularizer_weight=5e-5,
                    mini_batch_size=32,
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__(
            model=model,
            loss=loss,
            document_regularizer_weight=document_regularizer_weight,
            query_regularizer_weight=query_regularizer_weight,
            document_regularizer=document_regularizer,
            query_regularizer=query_regularizer,
            document_regularizer_threshold=document_regularizer_threshold,
            query_regularizer_threshold=query_regularizer_threshold,
            use_document_regularizer_only=use_document_regularizer_only,
        )
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the total loss (base loss + regularizers) from the per-mini-batch embeddings."""
        embeddings = [torch.cat(r) for r in reps]

        # Base loss
        base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(base_loss, dict):
            total_loss = sum(base_loss.values())
        else:
            total_loss = base_loss
        self._base_loss_value = total_loss.detach().item()

        # Document regularizer
        if self.use_document_regularizer_only:
            document_emb = torch.cat(embeddings)
        else:
            document_emb = torch.cat(embeddings[1:])
        doc_reg_loss = self.document_regularizer.compute_loss_from_embeddings(document_emb)
        weighted_doc_reg = doc_reg_loss * self.document_regularizer_weight
        self._doc_reg_value = weighted_doc_reg.detach().item()
        total_loss = total_loss + weighted_doc_reg

        # Query regularizer
        if self.query_regularizer_weight is not None:
            query_reg_loss = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
            weighted_query_reg = query_reg_loss * self.query_regularizer_weight
            self._query_reg_value = weighted_query_reg.detach().item()
            total_loss = total_loss + weighted_query_reg
        else:
            self._query_reg_value = None

        if with_backward:
            total_loss.backward()
            total_loss = total_loss.detach()

        return total_loss

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        total = self.forward_cached(sentence_features, labels)

        # Rebuild the per-component dict for the trainer's logging, around the single
        # gradient-carrying total: base_loss is adjusted so that sum(values) == total exactly.
        components = {
            "base_loss": torch.tensor(self._base_loss_value, device=total.device),
            "document_regularizer_loss": torch.tensor(self._doc_reg_value, device=total.device),
        }
        if self._query_reg_value is not None:
            components["query_regularizer_loss"] = torch.tensor(self._query_reg_value, device=total.device)
        return reconstruct_loss_components(total, components)

    def get_config_dict(self) -> dict[str, Any]:
        config = super().get_config_dict()
        config["mini_batch_size"] = self.mini_batch_size
        return config

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{formal2022distillationhardnegativesampling,
    title={From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
    author={Thibault Formal and Carlos Lassance and Benjamin Piwowarski and St\\'ephane Clinchant},
    year={2022},
    eprint={2205.04733},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
}
"""
