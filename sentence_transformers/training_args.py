from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union

from transformers import TrainingArguments as TransformersTrainingArguments
from transformers.training_args import ParallelMode
from transformers.utils import ExplicitEnum

from sentence_transformers.sampler import DefaultBatchSampler, MultiDatasetDefaultBatchSampler

logger = logging.getLogger(__name__)


class BatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for batch samplers.

    The batch sampler is responsible for determining how samples are grouped into batches during training.
    Valid options are:

    - ``BatchSamplers.BATCH_SAMPLER``: **[default]** Uses :class:`~sentence_transformers.sampler.DefaultBatchSampler`, the default
      PyTorch batch sampler.
    - ``BatchSamplers.NO_DUPLICATES``: Uses :class:`~sentence_transformers.sampler.NoDuplicatesBatchSampler`,
      ensuring no duplicate samples in a batch. Recommended for losses that use in-batch negatives, such as:

        - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.losses.MegaBatchMarginLoss`
        - :class:`~sentence_transformers.losses.GISTEmbedLoss`
        - :class:`~sentence_transformers.losses.CachedGISTEmbedLoss`
    - ``BatchSamplers.GROUP_BY_LABEL``: Uses :class:`~sentence_transformers.sampler.GroupByLabelBatchSampler`,
      ensuring that each batch has 2+ samples from the same label. Recommended for losses that require multiple
      samples from the same label, such as:

        - :class:`~sentence_transformers.losses.BatchAllTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardSoftMarginTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardTripletLoss`
        - :class:`~sentence_transformers.losses.BatchSemiHardTripletLoss`

    If you want to use a custom batch sampler, then you can subclass
    :class:`~sentence_transformers.sampler.DefaultBatchSampler` and pass the class (not an instance) to the
    ``batch_sampler`` argument in :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`
    (or :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, etc.).
    Alternatively, you can pass a function that accepts ``dataset``, ``batch_size``, ``drop_last``,
    ``valid_label_columns``, ``generator``, and ``seed`` and returns a
    :class:`~sentence_transformers.sampler.DefaultBatchSampler` instance.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.training_args import BatchSamplers
            from sentence_transformers.losses import MultipleNegativesRankingLoss
            from datasets import Dataset

            model = SentenceTransformer("microsoft/mpnet-base")
            train_dataset = Dataset.from_dict({
                "anchor": ["It's nice weather outside today.", "He drove to work."],
                "positive": ["It's so sunny.", "He took the car to the office."],
            })
            loss = MultipleNegativesRankingLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                batch_sampler=BatchSamplers.NO_DUPLICATES,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"


class MultiDatasetBatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for multi-dataset batch samplers.

    The multi-dataset batch sampler is responsible for determining in what order batches are sampled from multiple
    datasets during training. Valid options are:

    - ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Uses :class:`~sentence_transformers.sampler.RoundRobinBatchSampler`,
      which uses round-robin sampling from each dataset until one is exhausted.
      With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled
      from equally.
    - ``MultiDatasetBatchSamplers.PROPORTIONAL``: **[default]** Uses :class:`~sentence_transformers.sampler.ProportionalBatchSampler`,
      which samples from each dataset in proportion to its size.
      With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

    If you want to use a custom multi-dataset batch sampler, then you can subclass
    :class:`~sentence_transformers.sampler.MultiDatasetDefaultBatchSampler` and pass the class (not an instance) to the
    ``multi_dataset_batch_sampler`` argument in :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`.
    (or :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, etc.). Alternatively,
    you can pass a function that accepts ``dataset`` (a :class:`~torch.utils.data.ConcatDataset`), ``batch_samplers``
    (i.e. a list of batch sampler for each of the datasets in the :class:`~torch.utils.data.ConcatDataset`), ``generator``,
    and ``seed`` and returns a :class:`~sentence_transformers.sampler.MultiDatasetDefaultBatchSampler` instance.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.training_args import MultiDatasetBatchSamplers
            from sentence_transformers.losses import CoSENTLoss
            from datasets import Dataset, DatasetDict

            model = SentenceTransformer("microsoft/mpnet-base")
            train_general = Dataset.from_dict({
                "sentence_A": ["It's nice weather outside today.", "He drove to work."],
                "sentence_B": ["It's so sunny.", "He took the car to the bank."],
                "score": [0.9, 0.4],
            })
            train_medical = Dataset.from_dict({
                "sentence_A": ["The patient has a fever.", "The doctor prescribed medication.", "The patient is sweating."],
                "sentence_B": ["The patient feels hot.", "The medication was given to the patient.", "The patient is perspiring."],
                "score": [0.8, 0.6, 0.7],
            })
            train_legal = Dataset.from_dict({
                "sentence_A": ["This contract is legally binding.", "The parties agree to the terms and conditions."],
                "sentence_B": ["Both parties acknowledge their obligations.", "By signing this agreement, the parties enter into a legal relationship."],
                "score": [0.7, 0.8],
            })
            train_dataset = DatasetDict({
                "general": train_general,
                "medical": train_medical,
                "legal": train_legal,
            })

            loss = CoSENTLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    ROUND_ROBIN = "round_robin"  # Round-robin sampling from each dataset
    PROPORTIONAL = "proportional"  # Sample from each dataset in proportion to its size [default]


@dataclass
class SentenceTransformerTrainingArguments(TransformersTrainingArguments):
    r"""
    SentenceTransformerTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        prompts (`Union[Dict[str, Dict[str, str]], Dict[str, str], str]`, *optional*):
            The prompts to use for each column in the training, evaluation and test datasets. Four formats are accepted:

            1. `str`: A single prompt to use for all columns in the datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping column names to prompts, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            3. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your training/evaluation/test
               datasets are a :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.
            4. `Dict[str, Dict[str, str]]`: A dictionary mapping dataset names to dictionaries mapping column names to
               prompts. This should only be used if your training/evaluation/test datasets are a
               :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`, :class:`~sentence_transformers.sampler.DefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.sampler.DefaultBatchSampler`]], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`, :class:`~sentence_transformers.sampler.MultiDatasetDefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.sampler.MultiDatasetDefaultBatchSampler`]], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
        router_mapping (`Dict[str, str] | Dict[str, Dict[str, str]]`, *optional*):
            A mapping of dataset column names to Router routes, like "query" or "document". This is used to specify
            which Router submodule to use for each dataset. Two formats are accepted:

            1. `Dict[str, str]`: A mapping of column names to routes.
            2. `Dict[str, Dict[str, str]]`: A mapping of dataset names to a mapping of column names to routes for
               multi-dataset training/evaluation.
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
        "prompts",
        "router_mapping",
        "learning_rate_mapping",
    ]

    prompts: Union[str, None, dict[str, str], dict[str, dict[str, str]]] = field(  # noqa: UP007
        default=None,
        metadata={
            "help": "The prompts to use for each column in the datasets. "
            "Either 1) a single string prompt, 2) a mapping of column names to prompts, 3) a mapping of dataset names "
            "to prompts, or 4) a mapping of dataset names to a mapping of column names to prompts."
        },
    )
    batch_sampler: Union[BatchSamplers, str, DefaultBatchSampler, Callable[..., DefaultBatchSampler]] = field(  # noqa: UP007
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: Union[  # noqa: UP007
        MultiDatasetBatchSamplers, str, MultiDatasetDefaultBatchSampler, Callable[..., MultiDatasetDefaultBatchSampler]
    ] = field(
        default=MultiDatasetBatchSamplers.PROPORTIONAL, metadata={"help": "The multi-dataset batch sampler to use."}
    )
    router_mapping: Union[str, None, dict[str, str], dict[str, dict[str, str]]] = field(  # noqa: UP007
        default_factory=dict,
        metadata={
            "help": 'A mapping of dataset column names to Router routes, like "query" or "document". '
            "Either 1) a mapping of column names to routes or 2) a mapping of dataset names to a mapping "
            "of column names to routes for multi-dataset training/evaluation. "
        },
    )
    learning_rate_mapping: Union[str, None, dict[str, float]] = field(  # noqa: UP007
        default_factory=dict,
        metadata={
            "help": "A mapping of parameter name regular expressions to learning rates. "
            "This allows you to set different learning rates for different parts of the model, e.g., "
            r"{'SparseStaticEmbedding\.*': 1e-3} for the SparseStaticEmbedding module."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.batch_sampler = (
            BatchSamplers(self.batch_sampler) if isinstance(self.batch_sampler, str) else self.batch_sampler
        )
        self.multi_dataset_batch_sampler = (
            MultiDatasetBatchSamplers(self.multi_dataset_batch_sampler)
            if isinstance(self.multi_dataset_batch_sampler, str)
            else self.multi_dataset_batch_sampler
        )

        self.router_mapping = self.router_mapping if self.router_mapping is not None else {}
        if isinstance(self.router_mapping, str):
            # Note that we allow a stringified dictionary for router_mapping, but then it should have been
            # parsed by the superclass's `__post_init__` method already
            raise ValueError(
                "The `router_mapping` argument must be a dictionary mapping dataset column names to Router routes, "
                "like 'query' or 'document'. A stringified dictionary also works."
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
