from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import (
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.image_processing_utils import BaseImageProcessor

from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardCallback, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.util import fullname, is_datasets_available
from sentence_transformers.util.decorators import deprecated_kwargs

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset

logger = logging.getLogger(__name__)


class CrossEncoderTrainer(BaseTrainer):
    """
    CrossEncoderTrainer is a simple but feature-complete training and eval loop for PyTorch
    based on the 🤗 Transformers :class:`~transformers.Trainer`.

    This trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

    - :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if `wandb` is installed
    - :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if `tensorboard` is accessible.
    - :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if `codecarbon` is installed.

        - Note: These carbon emissions will be included in your automatically generated model card.

    See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
    documentation for more information on the integrated callbacks and how to write your own callbacks.

    Args:
        model (:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
        args (:class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, *optional*):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments` with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        train_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for training. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/cross_encoder/training_overview.html#dataset-format>`_.
        eval_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for evaluation. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/cross_encoder/training_overview.html#dataset-format>`_.
        loss (Optional[Union[:class:`torch.nn.Module`, Dict[str, :class:`torch.nn.Module`],\
            Callable[[:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`], :class:`torch.nn.Module`],\
            Dict[str, Callable[[:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`]]]], *optional*):
            The loss function to use for training. Can either be a loss class instance, a dictionary mapping
            dataset names to loss class instances, a function that returns a loss class instance given a model,
            or a dictionary mapping dataset names to functions that return a loss class instance given a model.
            In practice, the latter two are primarily used for hyper-parameter optimization. Will default to
            :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` if no ``loss`` is provided.
        evaluator (Union[:class:`~sentence_transformers.base.evaluation.BaseEvaluator`,\
            List[:class:`~sentence_transformers.base.evaluation.BaseEvaluator`]], *optional*):
            The evaluator instance for useful evaluation metrics during training. You can use an ``evaluator`` with
            or without an ``eval_dataset``, and vice versa. Generally, the metrics that an ``evaluator`` returns
            are more useful than the loss value returned from the ``eval_dataset``. A list of evaluators will be
            wrapped in a :class:`~sentence_transformers.base.evaluation.SequentialEvaluator` to run them sequentially.
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

    model_class = CrossEncoder
    model_card_data_class = CrossEncoderModelCardData
    model_card_callback_class = CrossEncoderModelCardCallback
    data_collator_class = CrossEncoderDataCollator
    training_args_class = CrossEncoderTrainingArguments

    @deprecated_kwargs(tokenizer="processing_class")
    def __init__(
        self,
        model: CrossEncoder | None = None,
        args: CrossEncoderTrainingArguments | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: nn.Module
        | dict[str, nn.Module]
        | Callable[[CrossEncoder], torch.nn.Module]
        | dict[str, Callable[[CrossEncoder], torch.nn.Module]]
        | None = None,
        evaluator: BaseEvaluator | list[BaseEvaluator] | None = None,
        data_collator: CrossEncoderDataCollator | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], CrossEncoder] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
            if isinstance(dataset, IterableDataset) or (
                isinstance(dataset, dict) and any(isinstance(d, IterableDataset) for d in dataset.values())
            ):
                # In short: `accelerate` will concatenate batches from the IterableDataset, expecting every
                # key-value pair after the data collator to only contain torch.Tensor values. However,
                # the CrossEncoderDataCollator returns a dictionary with string values (expecting the tokenization
                # to be done in the loss function). This will raise an error in `accelerate`.
                raise ValueError(
                    f"CrossEncoderTrainer does not support an IterableDataset for the `{dataset_name}_dataset`. "
                    "Please convert the dataset to a `Dataset` or `DatasetDict` before passing it to the trainer."
                )

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
        self.model: CrossEncoder
        self.args: CrossEncoderTrainingArguments
        self.data_collator: CrossEncoderDataCollator

    def get_default_loss(self, model: CrossEncoder) -> torch.nn.Module:
        if model.num_labels == 1:
            loss = BinaryCrossEntropyLoss(model)
        else:
            loss = CrossEntropyLoss(model)
        logger.info(f"No `loss` passed, using `{fullname(loss)}` as a default option.")
        return loss

    def collect_features(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor | None]:
        """Turn the inputs from the dataloader into the separate model inputs & the labels."""
        labels = inputs.pop("label", None)
        features = list(inputs.values())
        return features, labels

    def compute_loss(
        self,
        model: CrossEncoder,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        dataset_name = inputs.pop("dataset_name", None)
        prompt = inputs.pop("prompt", None)
        task = inputs.pop("task", None)
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        if model == self.model_wrapped and hasattr(loss_fn, "model") and loss_fn.model != model:
            loss_fn = self.override_model_in_loss(loss_fn, model)
        try:
            loss = loss_fn(features, labels, prompt=prompt, task=task)
        except TypeError:
            loss = loss_fn(features, labels)
            logger.warning_once(
                f"The `prompt` and `task` parameters could not be passed to {type(loss_fn).__name__}. "
                "These parameters are required for advanced preprocessing in losses. "
                "Please update your loss function to accept `prompt` and `task` keyword arguments."
            )
        if isinstance(loss, dict):
            self.track_loss_components(loss)
            loss = torch.stack(list(loss.values())).sum()
        if return_outputs:
            return loss, {}
        return loss
