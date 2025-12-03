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
from transformers.utils.deprecation import deprecate_kwarg

from sentence_transformers.base.evaluation import SentenceEvaluator
from sentence_transformers.base.models import Router
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.model_card import (
    SentenceTransformerModelCardCallback,
    SentenceTransformerModelCardData,
)
from sentence_transformers.sentence_transformer.models import Pooling
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

logger = logging.getLogger(__name__)


class SentenceTransformerTrainer(BaseTrainer):
    model_class = SentenceTransformer
    model_card_data_class = SentenceTransformerModelCardData
    model_card_callback_class = SentenceTransformerModelCardCallback
    data_collator_class = SentenceTransformerDataCollator
    training_args_class = SentenceTransformerTrainingArguments

    # TODO: I think we might be able to only put deprecate_kwarg on the base?
    @deprecate_kwarg("tokenizer", new_name="processing_class", version="6.0.0", raise_if_both_names=True)
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
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
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

        # Notify the data collator whether to include prompt lengths during batch preparation
        if hasattr(self.data_collator, "include_prompt_lengths"):
            self.data_collator.include_prompt_lengths = self._include_prompt_length()

    def get_default_loss(self, model: SentenceTransformer) -> torch.nn.Module:
        from sentence_transformers.sentence_transformer.losses import CoSENTLoss

        loss = CoSENTLoss(model)
        logger.info(f"No `loss` passed, using `{loss.__class__.__name__}` as a default option.")
        return loss

    def load_data_collator(
        self,
        model: SentenceTransformer,
        args: SentenceTransformerTrainingArguments,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
    ) -> SentenceTransformerDataCollator:
        if Router in [module.__class__ for module in model.children()] and not args.router_mapping:
            raise ValueError(
                "You are using a Router module in your model, but you did not provide a `router_mapping` in the "
                "training arguments. This means that the Router module will not be able to route the inputs to "
                "the correct submodules. Please provide a `router_mapping` that maps column names to routes, "
                "e.g. {'column_one': 'query', 'column_two': 'document', 'column_three': 'document'}."
            )

        all_special_ids = set()
        if processing_class is not None and hasattr(processing_class, "all_special_ids"):
            all_special_ids = set(processing_class.all_special_ids)
        return self.data_collator_class(
            tokenize_fn=model.preprocess,
            router_mapping=args.router_mapping,
            prompts=args.prompts,
            all_special_ids=all_special_ids,
        )

    def should_dataset_name_column_be_added(
        self,
        dataset: DatasetDict | Dataset | None,
        args: SentenceTransformerTrainingArguments,
        loss: nn.Module | dict[str, nn.Module],
    ) -> bool:
        """
        We should add a dataset name column to the dataset, if the dataset is a DatasetDict, *and* one of:

        a. The loss is a dictionary, or
        b. The prompts contain a mapping of dataset names, or
        c. The router_mapping contains a mapping of dataset names.
        """
        return isinstance(dataset, (DatasetDict, IterableDatasetDict)) and (
            isinstance(loss, dict)
            or (args.prompts and isinstance(args.prompts, dict))
            or (
                args.router_mapping
                and isinstance(args.router_mapping, dict)
                and isinstance(next(iter(args.router_mapping.values())), dict)
            )
        )

    def _include_prompt_length(self) -> bool:
        """
        Return whether the prompt length should be passed to the model's forward method.

        True if the model does not include the prompt in the pooling layer. Can be
        overridden by the user if it's useful to include the prompt length.
        """
        for module in self.model:
            if isinstance(module, Pooling):
                return not module.include_prompt
        return False
