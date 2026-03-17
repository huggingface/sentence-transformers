from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import (
    BaseImageProcessor,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.model_card import (
    SentenceTransformerModelCardCallback,
    SentenceTransformerModelCardData,
)
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_datasets_available
from sentence_transformers.util.decorators import deprecated_kwargs

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset

logger = logging.getLogger(__name__)


class SentenceTransformerTrainer(BaseTrainer):
    model_class = SentenceTransformer
    model_card_data_class = SentenceTransformerModelCardData
    model_card_callback_class = SentenceTransformerModelCardCallback
    data_collator_class = SentenceTransformerDataCollator
    training_args_class = SentenceTransformerTrainingArguments

    @deprecated_kwargs(tokenizer="processing_class")
    def __init__(
        self,
        model: SentenceTransformer | None = None,
        args: SentenceTransformerTrainingArguments | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: nn.Module
        | dict[str, nn.Module]
        | Callable[[SentenceTransformer], torch.nn.Module]
        | dict[str, Callable[[SentenceTransformer], torch.nn.Module]]
        | None = None,
        evaluator: BaseEvaluator | list[BaseEvaluator] | None = None,
        data_collator: SentenceTransformerDataCollator | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], SentenceTransformer] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
            data_collator=data_collator,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.model: SentenceTransformer
        self.args: SentenceTransformerTrainingArguments
        self.data_collator: SentenceTransformerDataCollator

    def get_default_loss(self, model: SentenceTransformer) -> torch.nn.Module:
        from sentence_transformers.sentence_transformer.losses import CoSENTLoss

        loss = CoSENTLoss(model)
        logger.info(f"No `loss` passed, using `{loss.__class__.__name__}` as a default option.")
        return loss
