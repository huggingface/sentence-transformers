from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import transformers
from packaging import version
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import TrainerCallback, TrainerControl, TrainerState, is_torch_npu_available
from transformers.tokenization_utils_base import BatchEncoding

from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.datasets.NoDuplicatesDataLoader import NoDuplicatesDataLoader
from sentence_transformers.datasets.SentenceLabelDataset import SentenceLabelDataset
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
    from sentence_transformers.readers.InputExample import InputExample


logger = logging.getLogger(__name__)


class SaveModelCallback(TrainerCallback):
    """A Callback to save the model to the `output_dir`.

    If save_best_model is True and evaluator is defined, then we save on evaluate, but only if the new model is
    better than the currently saved one according to the evaluator.

    This differs from the `SaveModelCallback` used in SentenceTransformer.fit where the model is saved after
    training as well.
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
        args: CrossEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        model: CrossEncoder,
        **kwargs,
    ) -> None:
        if self.evaluator is not None and self.save_best_model:
            metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
            for key, value in metrics.items():
                if key.endswith(metric_key):
                    if self.best_metric is None or self.is_better(value):
                        self.best_metric = value
                        model.save(self.output_dir)


class EvaluatorCallback(TrainerCallback):
    """The CrossEncoder.fit method always ran the evaluator on every epoch,
    in addition to every "evaluation_steps". This callback is responsible for that.

    The `.trainer` must be provided after the trainer has been created.
    """

    def __init__(self, evaluator: CrossEncoder, output_path: str | None = None) -> None:
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
        args: CrossEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: CrossEncoder,
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
    """A Callback to invoke the original callback function that was provided to CrossEncoder.fit()

    This callback has the following signature: `(score: float, epoch: int, steps: int) -> None`
    """

    def __init__(self, callback: Callable[[float, int, int], None], evaluator: SentenceEvaluator) -> None:
        super().__init__()
        self.callback = callback
        self.evaluator = evaluator

    def on_evaluate(
        self,
        args: CrossEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        **kwargs,
    ) -> None:
        metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
        for key, value in metrics.items():
            if key.endswith(metric_key):
                return self.callback(value, state.epoch, state.global_step)


class FitMixinLoss(nn.Module):
    """A wrapper around the torch loss function that just accepts logits and labels, to be used in CrossEncoder.fit()"""

    def __init__(self, model: CrossEncoder, loss_fct: nn.Module, activation_fn: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.activation_fn = activation_fn

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"BinaryCrossEntropyLoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]
        if self.model.config.num_labels == 1:
            logits = logits.view(-1)
            labels = labels.float()
        else:
            labels = labels.long()
        logits = self.activation_fn(logits)
        loss = self.loss_fct(logits, labels)
        return loss


class FitMixin:
    """Mixin class for injecting the `fit` and `old_fit` methods into the CrossEncoder class"""

    def fit(
        self: CrossEncoder,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator | None = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
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
    ) -> None:
        """
        Deprecated training method from before Sentence Transformers v4.0, it is recommended to use
        :class:`~sentence_transformers.trainer.CrossEncoderTrainer` instead. This method uses
        :class:`~sentence_transformers.trainer.CrossEncoderTrainer` behind the scenes, but does
        not provide as much flexibility as the Trainer itself.

        This training approach uses a DataLoader and Loss function to train the model.

        This method should produce equivalent results in v4.0 as before v4.0, but if you encounter any issues
        with your existing training scripts, then you may wish to use
        :meth:`CrossEncoder.old_fit <sentence_transformers.cross_encoder.CrossEncoder.old_fit>` instead.
        That uses the old training method from before v4.0.

        Args:
            train_dataloader: The DataLoader with InputExample instances
            evaluator: An evaluator (sentence_transformers.cross_encoder.evaluation)
                evaluates the model performance during training on held-
                out dev data. It is used to determine the best model
                that is saved to disk.
            epochs: Number of epochs for training
            loss_fct: Which loss function to use for training. If None,
                will use BinaryCrossEntropy() if self.config.num_labels == 1
                else CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits
                output of model.
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
        """
        if not is_datasets_available():
            raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

        # Delayed import to counter the CrossEncoder -> FitMixin -> CrossEncoderTrainer -> CrossEncoder circular import
        from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
        from sentence_transformers.cross_encoder.losses.CrossEntropyLoss import CrossEntropyLoss
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

        # Clear the dataloaders from collate functions as we just want raw InputExamples
        def identity(batch):
            return batch

        train_dataloader.collate_fn = identity

        batch_size = getattr(train_dataloader, "batch_size", 8)
        if isinstance(train_dataloader, NoDuplicatesDataLoader):
            batch_sampler = BatchSamplers.NO_DUPLICATES
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, SentenceLabelDataset):
            batch_sampler = BatchSamplers.GROUP_BY_LABEL
        else:
            batch_sampler = BatchSamplers.BATCH_SAMPLER

        texts = []
        labels = []
        for batch in train_dataloader:
            batch_texts, batch_labels = zip(*[(example.texts, example.label) for example in batch])
            texts += batch_texts
            labels += batch_labels
        train_dataset = Dataset.from_dict({f"sentence_{idx}": text for idx, text in enumerate(zip(*texts))})
        # Add label column, unless all labels are 0 (the default value for `labels` in InputExample)
        add_label_column = True
        try:
            if set(labels) == {0}:
                add_label_column = False
        except TypeError:
            pass
        if add_label_column:
            train_dataset = train_dataset.add_column("label", labels)

        def _default_checkpoint_dir() -> str:
            dir_name = "checkpoints/model"
            idx = 1
            while Path(dir_name).exists() and len(list(Path(dir_name).iterdir())) != 0:
                dir_name = f"checkpoints/model_{idx}"
                idx += 1
            return dir_name

        # Transformers renamed `evaluation_strategy` to `eval_strategy` in v4.41.0
        eval_strategy_key = (
            "eval_strategy"
            if version.parse(transformers.__version__) >= version.parse("4.41.0")
            else "evaluation_strategy"
        )
        args = CrossEncoderTrainingArguments(
            output_dir=_default_checkpoint_dir(),
            batch_sampler=batch_sampler,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            **{
                eval_strategy_key: "steps" if evaluation_steps is not None and evaluation_steps > 0 else "no",
            },
            eval_steps=evaluation_steps,
            max_grad_norm=max_grad_norm,
            fp16=use_amp,
            disable_tqdm=not show_progress_bar,
            save_strategy="no",
        )

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
        if isinstance(scheduler, str):
            steps_per_epoch = len(train_dataset) // batch_size
            num_train_steps = int(steps_per_epoch * epochs)
            scheduler_obj = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            if self.config.num_labels == 1:
                loss_fct = BinaryCrossEntropyLoss(self, activation_fn=activation_fct)
            else:
                loss_fct = CrossEntropyLoss(self, activation_fn=activation_fct)
        else:
            loss_fct = FitMixinLoss(self, loss_fct=loss_fct, activation_fn=activation_fct)

        # Create callbacks
        callbacks = []
        if evaluator is not None:
            callbacks.append(EvaluatorCallback(evaluator, output_path))
            if callback is not None:
                callbacks.append(OriginalCallback(callback, evaluator))

        trainer = CrossEncoderTrainer(
            model=self,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=None,
            loss=loss_fct,
            evaluator=evaluator,
            optimizers=(optimizer, scheduler_obj),
            callbacks=callbacks,
        )

        if output_path is not None:
            trainer.add_callback(SaveModelCallback(output_path, evaluator, save_best_model))

        trainer.train()

    def smart_batching_collate(self, batch: list[InputExample]) -> tuple[BatchEncoding, Tensor]:
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(
            *texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
        )
        assert self.max_length is None or tokenized["input_ids"].shape[0] <= self.max_length
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self.model.device
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.model.device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch: list[InputExample]) -> BatchEncoding:
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(
            *texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        assert self.max_length is None or tokenized["input_ids"].shape[0] <= self.max_length

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.model.device)

        return tokenized

    def old_fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator | None = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
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
    ) -> None:
        """
        Deprecated training method from before Sentence Transformers v4.0, it is recommended to use
        :class:`~sentence_transformers.trainer.CrossEncoderTrainer` instead. This method should
        only be used if you encounter issues with your existing training scripts after upgrading to v4.0.

        This training approach uses a DataLoader and Loss function to train the model.

        Args:
            train_dataloader: The DataLoader with InputExample instances
            evaluator: An evaluator (sentence_transformers.cross_encoder.evaluation)
                evaluates the model performance during training on held-
                out dev data. It is used to determine the best model
                that is saved to disk.
            epochs: Number of epochs for training
            loss_fct: Which loss function to use for training. If None,
                will use BinaryCrossEntropy() if self.config.num_labels == 1
                else CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits
                output of model.
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
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if use_amp:
                    with torch.autocast(device_type=self.model.device.type):
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback) -> None:
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if isinstance(score, dict) and hasattr(evaluator, "primary_metric") and evaluator.primary_metric in score:
                score = score[evaluator.primary_metric]
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
