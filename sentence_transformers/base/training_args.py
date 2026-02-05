from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union

from packaging.version import parse as parse_version
from transformers import TrainingArguments as TransformersTrainingArguments
from transformers import __version__ as transformers_version
from transformers.training_args import ParallelMode

from sentence_transformers.base.sampler import (
    BatchSamplers,
    DefaultBatchSampler,
    MultiDatasetBatchSamplers,
    MultiDatasetDefaultBatchSampler,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainingArguments(TransformersTrainingArguments):
    r"""
    SentenceTransformerTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        batch_sampler (Union[:class:`~sentence_transformers.sentence_transformer.training_args.BatchSamplers`, `str`, :class:`~sentence_transformers.base.sampler.DefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.base.sampler.DefaultBatchSampler`]], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.sentence_transformer.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.sentence_transformer.training_args.MultiDatasetBatchSamplers`, `str`, :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler`]], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.sentence_transformer.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
        learning_rate_mapping (`Dict[str, float] | None`, *optional*):
            A mapping of parameter name regular expressions to learning rates. This allows you to set different
            learning rates for different parts of the model, e.g., `{'SparseStaticEmbedding\.*': 1e-3}` for the
            SparseStaticEmbedding module. This is useful when you want to fine-tune specific parts of the model
            with different learning rates.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    _VALID_DICT_FIELDS = [
        "accelerator_config",
        "fsdp_config",
        "deepspeed",
        "gradient_checkpointing_kwargs",
        "lr_scheduler_kwargs",
        "learning_rate_mapping",
    ]

    batch_sampler: Union[BatchSamplers, str, DefaultBatchSampler, Callable[..., DefaultBatchSampler]] = field(  # noqa: UP007
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: Union[  # noqa: UP007
        MultiDatasetBatchSamplers, str, MultiDatasetDefaultBatchSampler, Callable[..., MultiDatasetDefaultBatchSampler]
    ] = field(
        default=MultiDatasetBatchSamplers.PROPORTIONAL, metadata={"help": "The multi-dataset batch sampler to use."}
    )
    learning_rate_mapping: Union[str, None, dict[str, float]] = field(  # noqa: UP007
        default_factory=dict,
        metadata={
            "help": "A mapping of parameter name regular expressions to learning rates. "
            "This allows you to set different learning rates for different parts of the model, e.g., "
            r"{'SparseStaticEmbedding\.*': 1e-3} for the SparseStaticEmbedding module."
        },
    )
    # Explicitly add warmup_ratio as transformers will remove it in the future, and I'd like to keep it until Sentence
    # Transformers drops support for transformers v4. The default mirrors that of transformers, regardless of version.
    warmup_ratio: float | None = field(
        default_factory=lambda: None if parse_version(transformers_version) >= parse_version("5.0.0") else 0.0,
        metadata={
            "help": "This argument is deprecated and will be removed in the future. If you're on Transformers v5+, "
            "then you should use `warmup_steps` instead as it also works with float values."
        },
    )

    def __post_init__(self):
        # Handle compatibility for warmup arguments across different transformers versions. Transformers v5+ only
        # supports warmup_steps (which can be an int or a float ratio) and removes warmup_ratio, while older versions
        # only support warmup_ratio and warmup_steps as an int number of steps. The logic here ensures that users can
        # use either argument across transformers versions, until we drop support for transformers <v5, at which point
        # users should only use warmup_steps.
        if parse_version(transformers_version) >= parse_version("5.0.0"):
            # For transformers >= 5: if the user provided a warmup_ratio but did not
            # explicitly set warmup_steps, treat warmup_ratio as the ratio and store
            # it in warmup_steps (which now accepts floats).
            if self.warmup_ratio is not None and self.warmup_steps == 0:
                self.warmup_steps = self.warmup_ratio
                self.warmup_ratio = None  # Reset to default

                logger.warning(
                    "The `warmup_ratio` argument is deprecated in Transformers v5+, and will also be removed from "
                    "Sentence Transformers once support for Transformers v4 is dropped. Since you're using "
                    "Transformers v5+, please use `warmup_steps` (as a float) to specify the warmup ratio instead."
                )
        else:
            # For transformers < 5: allow users to pass a float in warmup_steps to
            # denote a warmup ratio (0 < ratio < 1) and convert it into warmup_ratio.
            # NOTE: For transformers v4, warmup_ratio defaults to 0.0, while in v5 it defaults to None.
            if isinstance(self.warmup_steps, float) and 0.0 < self.warmup_steps < 1.0 and self.warmup_ratio == 0.0:
                self.warmup_ratio = self.warmup_steps
                self.warmup_steps = 0  # Set to default

        super().__post_init__()

        self.batch_sampler = (
            BatchSamplers(self.batch_sampler) if isinstance(self.batch_sampler, str) else self.batch_sampler
        )
        self.multi_dataset_batch_sampler = (
            MultiDatasetBatchSamplers(self.multi_dataset_batch_sampler)
            if isinstance(self.multi_dataset_batch_sampler, str)
            else self.multi_dataset_batch_sampler
        )

        self.learning_rate_mapping = self.learning_rate_mapping if self.learning_rate_mapping is not None else {}
        if isinstance(self.learning_rate_mapping, str):
            # Note that we allow a stringified dictionary for learning_rate_mapping, but then it should have been
            # parsed by the superclass's `__post_init__` method already
            raise ValueError(
                "The `learning_rate_mapping` argument must be a dictionary mapping parameter name regular expressions "
                "to learning rates. A stringified dictionary also works."
            )

        # The `compute_loss` method in `SentenceTransformerTrainer` is overridden to only compute the prediction loss,
        # so we set `prediction_loss_only` to `True` here to avoid
        self.prediction_loss_only = True

        # Disable broadcasting of buffers to avoid `RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation.` when training with DDP & a BertModel-based model.
        self.ddp_broadcast_buffers = False

        if self.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "Currently using DataParallel (DP) for multi-gpu training, while DistributedDataParallel (DDP) is recommended for faster training. "
                    "See https://sbert.net/docs/sentence_transformer/training/distributed.html for more information."
                )

        elif self.parallel_mode == ParallelMode.DISTRIBUTED and not self.dataloader_drop_last:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "When using DistributedDataParallel (DDP), it is recommended to set `dataloader_drop_last=True` to avoid hanging issues with an uneven last batch. "
                    "Setting `dataloader_drop_last=True`."
                )
            self.dataloader_drop_last = True

    def to_dict(self):
        training_args_dict = super().to_dict()
        if callable(training_args_dict["batch_sampler"]):
            del training_args_dict["batch_sampler"]
        if callable(training_args_dict["multi_dataset_batch_sampler"]):
            del training_args_dict["multi_dataset_batch_sampler"]
        return training_args_dict
