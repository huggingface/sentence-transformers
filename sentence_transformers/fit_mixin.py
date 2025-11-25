from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import transformers
from packaging import version
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from transformers import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers.datasets.NoDuplicatesDataLoader import NoDuplicatesDataLoader
from sentence_transformers.datasets.SentenceLabelDataset import SentenceLabelDataset
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import batch_to_device, fullname, is_datasets_available

from .evaluation import SentenceEvaluator
from .model_card_templates import ModelCardTemplate

if is_datasets_available():
    from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.readers.InputExample import InputExample
    from sentence_transformers.SentenceTransformer import SentenceTransformer


class SaveModelCallback(TrainerCallback):
    """A Callback to save the model to the `output_dir`.

    There are two cases:
    1. save_best_model is True and evaluator is defined:
        We save on evaluate, but only if the new model is better than the currently saved one
        according to the evaluator.
    2. If evaluator is not defined:
        We save after the model has been trained.
    """

    def __init__(self, output_dir: str, evaluator: SentenceEvaluator | None, save_best_model: bool) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.save_best_model = save_best_model
        self.best_metric = None

    def is_better(self, new_metric: float) -> bool:
        if getattr(self.evaluator, "greater_is_better", True):
            return new_metric > self.best_metric
        return new_metric < self.best_metric

    def on_evaluate(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        if self.evaluator is not None and self.save_best_model:
            metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
            for key, value in metrics.items():
                if key.endswith(metric_key):
                    if self.best_metric is None or self.is_better(value):
                        self.best_metric = value
                        model.save(self.output_dir)

    def on_train_end(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        if self.evaluator is None:
            model.save(self.output_dir)


class EvaluatorCallback(TrainerCallback):
    """The SentenceTransformers.fit method always ran the evaluator on every epoch,
    in addition to every "evaluation_steps". This callback is responsible for that.

    The `.trainer` must be provided after the trainer has been created.
    """

    def __init__(self, evaluator: SentenceEvaluator, output_path: str | None = None) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = os.path.join(self.output_path, "eval")
            os.makedirs(self.output_path, exist_ok=True)

        self.metric_key_prefix = "eval"
        self.trainer = None

    def on_epoch_end(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        evaluator_metrics = self.evaluator(
            model, output_path=self.output_path, epoch=state.epoch, steps=state.global_step
        )
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{self.metric_key_prefix}_"):
                evaluator_metrics[f"{self.metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        if self.trainer is not None:
            self.trainer.callback_handler.on_evaluate(args, state, control, metrics=evaluator_metrics)


class OriginalCallback(TrainerCallback):
    """A Callback to invoke the original callback function that was provided to SentenceTransformer.fit()

    This callback has the following signature: `(score: float, epoch: int, steps: int) -> None`
    """

    def __init__(self, callback: Callable[[float, int, int], None], evaluator: SentenceEvaluator) -> None:
        super().__init__()
        self.callback = callback
        self.evaluator = evaluator

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        **kwargs,
    ) -> None:
        metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
        for key, value in metrics.items():
            if key.endswith(metric_key):
                return self.callback(value, state.epoch, state.global_step)


class FitMixin:
    """Mixin class for injecting the `fit` and `old_fit` methods into SentenceTransformer models"""

    def fit(
        self,
        train_objectives: Iterable[tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator | None = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        optimizer_params: dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str | None = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str | None = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        resume_from_checkpoint: bool = False,
    ) -> None:
        """
        Deprecated training method from before Sentence Transformers v3.0, it is recommended to use
        :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method uses
        :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` behind the scenes, but does
        not provide as much flexibility as the Trainer itself.

        This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader
        is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the
        smallest one to make sure of equal training with each dataset, i.e. round robin sampling.

        This method should produce equivalent results in v3.0+ as before v3.0, but if you encounter any issues
        with your existing training scripts, then you may wish to use
        :meth:`SentenceTransformer.old_fit <sentence_transformers.SentenceTransformer.old_fit>` instead.
        That uses the old training method from before v3.0.

        Args:
            train_objectives: Tuples of (DataLoader, LossFunction). Pass
                more than one for multi-task learning
            evaluator: An evaluator (sentence_transformers.evaluation)
                evaluates the model performance during training on held-
                out dev data. It is used to determine the best model
                that is saved to disk.
            epochs: Number of epochs for training
            steps_per_epoch: Number of training steps per epoch. If set
                to None (default), one epoch is equal the DataLoader
                size from train_objectives.
            scheduler: Learning rate scheduler. Available schedulers:
                constantlr, warmupconstant, warmuplinear, warmupcosine,
                warmupcosinewithhardrestarts
            warmup_steps: Behavior depends on the scheduler. For
                WarmupLinear (default), the learning rate is increased
                from o up to the maximal learning rate. After these many
                training steps, the learning rate is decreased linearly
                back to zero.
            optimizer_class: Optimizer
            optimizer_params: Optimizer parameters
            weight_decay: Weight decay for model parameters
            evaluation_steps: If > 0, evaluate the model using evaluator
                after each number of training steps
            output_path: Storage path for the model and evaluation files
            save_best_model: If true, the best model (according to
                evaluator) is stored at output_path
            max_grad_norm: Used for gradient normalization.
            use_amp: Use Automatic Mixed Precision (AMP). Only for
                Pytorch >= 1.6.0
            callback: Callback function that is invoked after each
                evaluation. It must accept the following three
                parameters in this order: `score`, `epoch`, `steps`
            show_progress_bar: If True, output a tqdm progress bar
            checkpoint_path: Folder to save checkpoints during training
            checkpoint_save_steps: Will save a checkpoint after so many
                steps
            checkpoint_save_total_limit: Total number of checkpoints to
                store
            resume_from_checkpoint: If true, searches for checkpoints
                to continue training from.
        """
        if not is_datasets_available():
            raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

        # Delayed import to counter the SentenceTransformers -> FitMixin -> SentenceTransformerTrainer -> SentenceTransformers circular import
        from sentence_transformers.trainer import SentenceTransformerTrainer

        data_loaders, loss_fns = zip(*train_objectives)

        # Clear the dataloaders from collate functions as we just want raw InputExamples
        def identity(batch):
            return batch

        for data_loader in data_loaders:
            data_loader.collate_fn = identity

        batch_size = 8
        batch_sampler = BatchSamplers.BATCH_SAMPLER
        # Convert dataloaders into a DatasetDict
        # TODO: This is rather inefficient, as we load all data into memory. We might benefit from a more efficient solution
        train_dataset_dict = {}
        for loader_idx, data_loader in enumerate(data_loaders, start=1):
            if isinstance(data_loader, NoDuplicatesDataLoader):
                batch_sampler = BatchSamplers.NO_DUPLICATES
            elif hasattr(data_loader, "dataset") and isinstance(data_loader.dataset, SentenceLabelDataset):
                batch_sampler = BatchSamplers.GROUP_BY_LABEL

            batch_size = getattr(data_loader, "batch_size", batch_size)
            texts = []
            labels = []
            for batch in data_loader:
                batch_texts, batch_labels = zip(*[(example.texts, example.label) for example in batch])
                texts += batch_texts
                labels += batch_labels
            dataset = Dataset.from_dict({f"sentence_{idx}": text for idx, text in enumerate(zip(*texts))})
            # Add label column, unless all labels are 0 (the default value for `labels` in InputExample)
            add_label_column = True
            try:
                if set(labels) == {0}:
                    add_label_column = False
            except TypeError:
                pass
            if add_label_column:
                dataset = dataset.add_column("label", labels)
            train_dataset_dict[f"_dataset_{loader_idx}"] = dataset

        train_dataset_dict = DatasetDict(train_dataset_dict)

        def _default_checkpoint_dir() -> str:
            dir_name = "checkpoints/model"
            idx = 1
            while Path(dir_name).exists() and len(list(Path(dir_name).iterdir())) != 0:
                dir_name = f"checkpoints/model_{idx}"
                idx += 1
            return dir_name

        # Convert loss_fns into a dict with `dataset_{idx}` keys
        loss_fn_dict = {f"_dataset_{idx}": loss_fn for idx, loss_fn in enumerate(loss_fns, start=1)}

        # Use steps_per_epoch to perhaps set max_steps
        max_steps = -1
        if steps_per_epoch is not None and steps_per_epoch > 0:
            if epochs == 1:
                max_steps = steps_per_epoch
            else:
                logger.warning(
                    "Setting `steps_per_epoch` alongside `epochs` > 1 no longer works. "
                    "We will train with the full datasets per epoch."
                )
                steps_per_epoch = None

        # Transformers renamed `evaluation_strategy` to `eval_strategy` in v4.41.0
        eval_strategy_key = (
            "eval_strategy"
            if version.parse(transformers.__version__) >= version.parse("4.41.0")
            else "evaluation_strategy"
        )
        args = SentenceTransformerTrainingArguments(
            output_dir=checkpoint_path or _default_checkpoint_dir(),
            batch_sampler=batch_sampler,
            multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            max_steps=max_steps,
            **{
                eval_strategy_key: "steps" if evaluation_steps is not None and evaluation_steps > 0 else "no",
            },
            eval_steps=evaluation_steps,
            max_grad_norm=max_grad_norm,
            fp16=use_amp,
            disable_tqdm=not show_progress_bar,
            save_strategy="steps" if checkpoint_path is not None else "no",
            save_steps=checkpoint_save_steps,
            save_total_limit=checkpoint_save_total_limit,
        )

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(train_dataset) // batch_size for train_dataset in train_dataset_dict.values()])
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizer & scheduler
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = self._get_scheduler(
            optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
        )

        # Create callbacks
        callbacks = []
        if evaluator is not None:
            callbacks.append(EvaluatorCallback(evaluator, output_path))
            if callback is not None:
                callbacks.append(OriginalCallback(callback, evaluator))

        trainer = SentenceTransformerTrainer(
            model=self,
            args=args,
            train_dataset=train_dataset_dict,
            eval_dataset=None,
            loss=loss_fn_dict,
            evaluator=evaluator,
            optimizers=(optimizer, scheduler_obj),
            callbacks=callbacks,
        )
        # Set the trainer on the EvaluatorCallback, required for logging the metrics
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EvaluatorCallback):
                callback.trainer = trainer

        if output_path is not None:
            trainer.add_callback(SaveModelCallback(output_path, evaluator, save_best_model))

        if checkpoint_path is not None and resume_from_checkpoint:
            if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                logger.info(f"Looking for checkpoints in: {checkpoint_path}")

                all_checkpoints = [
                    checkpoint
                    for checkpoint in os.listdir(checkpoint_path)
                    if checkpoint.startswith("checkpoint-") and checkpoint.split("-")[-1].isdigit()
                ]

                if all_checkpoints:
                    latest_checkpoint = max(all_checkpoints, key=lambda x: int(x.split("-")[-1]))
                    resume_from_checkpoint = os.path.join(checkpoint_path, latest_checkpoint)
                    logger.info(f"Resuming from latest checkpoint: {resume_from_checkpoint}")
                else:
                    logger.warning(f"No checkpoints found in checkpoint directory: {checkpoint_path}")
                    resume_from_checkpoint = None
            else:
                logger.warning(f"Checkpoint directory does not exist or is not a directory: {checkpoint_path}")
                resume_from_checkpoint = None

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int) -> LambdaLR:
        """
        Returns the correct learning rate scheduler. Available scheduler:

        - constantlr,
        - warmupconstant,
        - warmuplinear,
        - warmupcosine,
        - warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == "constantlr":
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == "warmupconstant":
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == "warmuplinear":
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosinewithhardrestarts":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

    def smart_batching_collate(self, batch: list[InputExample]) -> tuple[list[dict[str, Tensor]], Tensor]:
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        Args:
            batch: a batch from a SmartBatchingDataset

        Returns:
            a batch of tensors for the model
        """
        texts = [example.texts for example in batch]
        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        labels = [example.label for example in batch]

        # Use torch.from_numpy to convert the numpy array directly to a tensor,
        # which is the recommended approach for converting numpy arrays to tensors
        if labels and isinstance(labels[0], np.ndarray):
            labels_tensor = torch.from_numpy(np.stack(labels))
        else:
            labels_tensor = torch.tensor(labels)

        return sentence_features, labels_tensor

    """
    Temporary methods that will be removed when this refactor is complete:
    """

    def old_fit(
        self,
        train_objectives: Iterable[tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator | None = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        optimizer_params: dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str | None = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str | None = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
    ) -> None:
        """
        Deprecated training method from before Sentence Transformers v3.0, it is recommended to use
        :class:`sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method should
        only be used if you encounter issues with your existing training scripts after upgrading to v3.0+.

        This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader
        is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the
        smallest one to make sure of equal training with each dataset, i.e. round robin sampling.

        Args:
            train_objectives: Tuples of (DataLoader, LossFunction). Pass
                more than one for multi-task learning
            evaluator: An evaluator (sentence_transformers.evaluation)
                evaluates the model performance during training on held-
                out dev data. It is used to determine the best model
                that is saved to disk.
            epochs: Number of epochs for training
            steps_per_epoch: Number of training steps per epoch. If set
                to None (default), one epoch is equal the DataLoader
                size from train_objectives.
            scheduler: Learning rate scheduler. Available schedulers:
                constantlr, warmupconstant, warmuplinear, warmupcosine,
                warmupcosinewithhardrestarts
            warmup_steps: Behavior depends on the scheduler. For
                WarmupLinear (default), the learning rate is increased
                from o up to the maximal learning rate. After these many
                training steps, the learning rate is decreased linearly
                back to zero.
            optimizer_class: Optimizer
            optimizer_params: Optimizer parameters
            weight_decay: Weight decay for model parameters
            evaluation_steps: If > 0, evaluate the model using evaluator
                after each number of training steps
            output_path: Storage path for the model and evaluation files
            save_best_model: If true, the best model (according to
                evaluator) is stored at output_path
            max_grad_norm: Used for gradient normalization.
            use_amp: Use Automatic Mixed Precision (AMP). Only for
                Pytorch >= 1.6.0
            callback: Callback function that is invoked after each
                evaluation. It must accept the following three
                parameters in this order: `score`, `epoch`, `steps`
            show_progress_bar: If True, output a tqdm progress bar
            checkpoint_path: Folder to save checkpoints during training
            checkpoint_save_steps: Will save a checkpoint after so many
                steps
            checkpoint_save_total_limit: Total number of checkpoints to
                store
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {
                "evaluator": fullname(evaluator),
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "optimizer_class": str(optimizer_class),
                "optimizer_params": optimizer_params,
                "weight_decay": weight_decay,
                "evaluation_steps": evaluation_steps,
                "max_grad_norm": max_grad_norm,
            },
            indent=4,
            sort_keys=True,
        )
        self._model_card_text = None
        self._model_card_vars["{TRAINING_SECTION}"] = ModelCardTemplate.__TRAINING_SECTION__.replace(
            "{LOSS_FUNCTIONS}", info_loss_functions
        ).replace("{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device)
                    features = list(map(lambda batch: batch_to_device(batch, self.device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback) -> None:
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step) -> None:
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({"step": int(subdir), "path": os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
                shutil.rmtree(old_checkpoints[0]["path"])
