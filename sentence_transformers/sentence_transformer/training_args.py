from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from sentence_transformers.base.sampler import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.base.training_args import BaseTrainingArguments


@dataclass
class SentenceTransformerTrainingArguments(BaseTrainingArguments):
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

        router_mapping (`Dict[str, str] | Dict[str, Dict[str, str]]`, *optional*):
            A mapping of dataset column names to Router routes, like "query" or "document". This is used to specify
            which Router submodule to use for each dataset. Two formats are accepted:

            1. `Dict[str, str]`: A mapping of column names to routes.
            2. `Dict[str, Dict[str, str]]`: A mapping of dataset names to a mapping of column names to routes for
               multi-dataset training/evaluation.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    _VALID_DICT_FIELDS = BaseTrainingArguments._VALID_DICT_FIELDS + ["prompts", "router_mapping"]

    prompts: Union[str, None, dict[str, str], dict[str, dict[str, str]]] = field(  # noqa: UP007
        default=None,
        metadata={
            "help": "The prompts to use for each column in the datasets. "
            "Either 1) a single string prompt, 2) a mapping of column names to prompts, 3) a mapping of dataset names "
            "to prompts, or 4) a mapping of dataset names to a mapping of column names to prompts."
        },
    )
    router_mapping: Union[str, None, dict[str, str], dict[str, dict[str, str]]] = field(  # noqa: UP007
        default_factory=dict,
        metadata={
            "help": 'A mapping of dataset column names to Router routes, like "query" or "document". '
            "Either 1) a mapping of column names to routes or 2) a mapping of dataset names to a mapping "
            "of column names to routes for multi-dataset training/evaluation. "
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.router_mapping = self.router_mapping if self.router_mapping is not None else {}
        if isinstance(self.router_mapping, str):
            # Note that we allow a stringified dictionary for router_mapping, but then it should have been
            # parsed by the superclass's `__post_init__` method already
            raise ValueError(
                "The `router_mapping` argument must be a dictionary mapping dataset column names to Router routes, "
                "like 'query' or 'document'. A stringified dictionary also works."
            )


__all__ = ["SentenceTransformerTrainingArguments", "BatchSamplers", "MultiDatasetBatchSamplers"]
