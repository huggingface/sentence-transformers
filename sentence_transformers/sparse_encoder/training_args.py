from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.training_args import BaseTrainingArguments


@dataclass
class SparseEncoderTrainingArguments(BaseTrainingArguments):
    r"""
    SparseEncoderTrainingArguments extends :class:`~SentenceTransformerTrainingArguments` which itself extend
    :class:`~transformers.TrainingArguments` with additional arguments specific to Sentence Transformers.
    See :class:`~transformers.TrainingArguments` for the complete list of available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
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
