from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext, _backward_hook
from sentence_transformers.sparse_encoder.losses.SpladeLoss import SpladeLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)


class CachedSpladeLoss(SpladeLoss):
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
        regularizers still receive pre-computed embeddings via ``compute_loss_from_embeddings()`` — no
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
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of sentences."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches of sentences for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        batch_size = input_ids.shape[0]
        for i, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], labels: Tensor | None) -> Tensor:
        """Calculate the combined loss (base + regularizers) and cache gradients w.r.t. the embeddings."""
        loss = self._compute_total_loss(reps, labels, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def _compute_total_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None, with_backward: bool = False
    ) -> Tensor:
        """Compute total loss from base loss + regularizers on mini-batch reps."""
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
    ) -> dict[str, Tensor] | Tensor:
        sentence_features = list(sentence_features)

        # Step (1): Embed all mini-batches without gradients to get all embeddings
        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the combined loss, backward to embeddings and cache gradients
            loss = self.calculate_loss_and_cache_gradients(reps, labels)

            # Step (3): Register backward hook to chain cached gradients back through the model
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))

            # Build dict for loss component logging while preserving gradient flow through `loss`.
            # The trainer sums all dict values for backward, so we put the gradient-carrying tensor
            # in "base_loss" and use detached tensors for regularizer entries.
            device = loss.device
            result = {}
            non_base_sum = torch.tensor(0.0, device=device)

            doc_reg_detached = torch.tensor(self._doc_reg_value, device=device)
            result["document_regularizer_loss"] = doc_reg_detached
            non_base_sum = non_base_sum + doc_reg_detached

            if self._query_reg_value is not None:
                query_reg_detached = torch.tensor(self._query_reg_value, device=device)
                result["query_regularizer_loss"] = query_reg_detached
                non_base_sum = non_base_sum + query_reg_detached

            # base_loss = loss - regularizers, so trainer's sum(values) = loss (exact gradient flow)
            result["base_loss"] = loss - non_base_sum

            return result
        else:
            # Eval mode: no caching needed, compute losses directly
            embeddings = [torch.cat(r) for r in reps]
            losses = {}

            base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
            if isinstance(base_loss, dict):
                losses.update(base_loss)
            else:
                losses["base_loss"] = base_loss

            if self.use_document_regularizer_only:
                document_emb = torch.cat(embeddings)
            else:
                document_emb = torch.cat(embeddings[1:])
            doc_reg_loss = self.document_regularizer.compute_loss_from_embeddings(document_emb)
            losses["document_regularizer_loss"] = doc_reg_loss * self.document_regularizer_weight

            if self.query_regularizer_weight is not None:
                query_reg_loss = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
                losses["query_regularizer_loss"] = query_reg_loss * self.query_regularizer_weight

            return losses

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
