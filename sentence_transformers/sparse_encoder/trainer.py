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

from sentence_transformers.base.models import Router
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
from sentence_transformers.sparse_encoder.callbacks.splade_callbacks import SpladeRegularizerWeightSchedulerCallback
from sentence_transformers.sparse_encoder.data_collator import SparseEncoderDataCollator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.model import SparseEncoder
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardCallback, SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

logger = logging.getLogger(__name__)


class SparseEncoderTrainer(BaseTrainer):
    """
    SparseEncoderTrainer is a simple but feature-complete training and eval loop for PyTorch
    based on the SentenceTransformerTrainer that based on 🤗 Transformers :class:`~transformers.Trainer`.

    This trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

    - :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if `wandb` is installed
    - :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if `tensorboard` is accessible.
    - :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if `codecarbon` is installed.

        - Note: These carbon emissions will be included in your automatically generated model card.

    See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
    documentation for more information on the integrated callbacks and how to write your own callbacks.

    Args:
        model (:class:`~sentence_transformers.SparseEncoder`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
        args (:class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments`, *optional*):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        train_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, :class:`datasets.IterableDataset`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for training. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/sentence_transformer/training_overview.html#dataset-format>`_.
        eval_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, :class:`datasets.IterableDataset`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for evaluation. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/sentence_transformer/training_overview.html#dataset-format>`_.
        loss (Optional[Union[:class:`torch.nn.Module`, Dict[str, :class:`torch.nn.Module`],\
            Callable[[:class:`~sentence_transformers.SparseEncoder`], :class:`torch.nn.Module`],\
            Dict[str, Callable[[:class:`~sentence_transformers.SparseEncoder`]]]], *optional*):
            The loss function to use for training. Can either be a loss class instance, a dictionary mapping
            dataset names to loss class instances, a function that returns a loss class instance given a model,
            or a dictionary mapping dataset names to functions that return a loss class instance given a model.
            In practice, the latter two are primarily used for hyper-parameter optimization. Will default to
            :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` if no ``loss`` is provided.
        evaluator (Union[:class:`~sentence_transformers.sentence_transformer.evaluation.SentenceEvaluator`,\
            List[:class:`~sentence_transformers.sentence_transformer.evaluation.SentenceEvaluator`]], *optional*):
            The evaluator instance for useful evaluation metrics during training. You can use an ``evaluator`` with
            or without an ``eval_dataset``, and vice versa. Generally, the metrics that an ``evaluator`` returns
            are more useful than the loss value returned from the ``eval_dataset``. A list of evaluators will be
            wrapped in a :class:`~sentence_transformers.sentence_transformer.evaluation.SequentialEvaluator` to run them sequentially.
        callbacks (List of [:class:`transformers.TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[:class:`torch.optim.Optimizer`, :class:`torch.optim.lr_scheduler.LambdaLR`]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of :class:`torch.optim.AdamW`
            on your model and a scheduler given by :func:`transformers.get_linear_schedule_with_warmup` controlled by `args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)
    """

    model_class = SparseEncoder
    model_card_data_class = SparseEncoderModelCardData
    model_card_callback_class = SparseEncoderModelCardCallback
    data_collator_class = SparseEncoderDataCollator
    training_args_class = SparseEncoderTrainingArguments

    @deprecate_kwarg("tokenizer", new_name="processing_class", version="6.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: SparseEncoder | None = None,
        args: SparseEncoderTrainingArguments | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: nn.Module
        | dict[str, nn.Module]
        | Callable[[SparseEncoder], torch.nn.Module]
        | dict[str, Callable[[SparseEncoder], torch.nn.Module]]
        | None = None,
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
        data_collator: SparseEncoderDataCollator | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], SparseEncoder] | None = None,
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
        self.model: SparseEncoder
        self.args: SparseEncoderTrainingArguments
        self.data_collator: SparseEncoderDataCollator

    def get_default_loss(self, model: SparseEncoder) -> torch.nn.Module:
        logger.info(
            "No `loss` passed, using `sentence_transformers.sparse_encoder.losses.SpladeLoss` as a default option. with "
            "`SparseMultipleNegativesRankingLoss` as the default loss function."
            "Be careful, we also set the `query_regularizer_weight` and `document_regularizer_weight`, but these are "
            "really sensitive parameters and should be tuned for your task."
        )
        return SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=5e-5,  # Weight for query loss
            document_regularizer_weight=3e-5,  # Weight for document loss
        )

    def load_data_collator(
        self,
        model: SparseEncoder,
        args: SparseEncoderTrainingArguments,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
    ) -> SparseEncoderDataCollator:
        """
        Load the data collator for the trainer.

        Args:
            model (:class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`):
                The model to train, evaluate or use for predictions.
            args (:class:`~sentence_transformers.sentence_transformer.training_args.BaseTrainingArguments`):
                The arguments to tweak for training.
            processing_class (Union[:class:`transformers.PreTrainedTokenizerBase`, :class:`transformers.BaseImageProcessor`, :class:`transformers.FeatureExtractionMixin`, :class:`transformers.ProcessorMixin`], *optional*):
                The processing class to use for tokenization or image processing.
        Returns:
            :class:`BaseDataCollator`: The data collator to use for the trainer

        .. note::

            This method can be overridden by subclassing the trainer to use a custom data collator.
        """
        if Router in [module.__class__ for module in model.children()] and not args.router_mapping:
            raise ValueError(
                "You are using a Router module in your model, but you did not provide a `router_mapping` in the "
                "training arguments. This means that the Router module will not be able to route the inputs to "
                "the correct submodules. Please provide a `router_mapping` that maps column names to routes, "
                "e.g. {'column_one': 'query', 'column_two': 'document', 'column_three': 'document'}."
            )

        return self.data_collator_class(tokenize_fn=model.preprocess, router_mapping=args.router_mapping)

    def should_dataset_name_column_be_added(
        self,
        dataset: DatasetDict | Dataset | None,
        args: SparseEncoderTrainingArguments,
        loss: nn.Module | dict[str, nn.Module],
    ) -> bool:
        """
        We should add a dataset name column to the dataset, if the dataset is a DatasetDict, *and* one of:

        a. The loss is a dictionary, or
        b. The router_mapping contains a mapping of dataset names.
        """

        return isinstance(dataset, (DatasetDict, IterableDatasetDict)) and (
            isinstance(loss, dict)
            or (
                args.router_mapping
                and isinstance(args.router_mapping, dict)
                and isinstance(next(iter(args.router_mapping.values())), dict)
            )
        )

    def prepare_loss(
        self,
        loss: Callable[[SparseEncoder], torch.nn.Module] | torch.nn.Module,
        model: SparseEncoder,
    ) -> torch.nn.Module:
        if isinstance(loss, torch.nn.Module):
            loss = loss.to(model.device)
        else:
            loss = loss(model).to(model.device)

        is_splade_loss = isinstance(loss, SpladeLoss) if loss is not None else False
        splade_scheduler_callback_index = None
        for idx, callback in enumerate(self.callback_handler.callbacks):
            if isinstance(callback, SpladeRegularizerWeightSchedulerCallback):
                splade_scheduler_callback_index = idx
                break

        # If we're using SpladeLoss but don't have a scheduler callback, add one or if it's not the second one in the list
        if is_splade_loss and (splade_scheduler_callback_index is None or splade_scheduler_callback_index > 1):
            if splade_scheduler_callback_index is not None:
                splade_callback = self.callback_handler.callbacks.pop(splade_scheduler_callback_index)

            else:
                logger.warning(
                    "SpladeLoss detected without SpladeRegularizerWeightSchedulerCallback. "
                    "Adding default SpladeRegularizerWeightSchedulerCallback to gradually increase weight values from 0 to their maximum."
                )

                # Create and insert the callback after the default callback informing the trainer when to log, evaluate, save, etc.
                splade_callback = SpladeRegularizerWeightSchedulerCallback(loss=loss)
            self.callback_handler.callbacks.insert(1, splade_callback)

        return loss
