from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.training_args import BaseTrainingArguments


@dataclass
class CrossEncoderTrainingArguments(BaseTrainingArguments):
    r"""
    CrossEncoderTrainingArguments extends :class:`~sentence_transformers.base.training_args.BaseTrainingArguments`
    with additional arguments specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for
    the complete list of available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        prompts (`Union[Dict[str, str], str]`, *optional*):
            The prompts to use in the training, evaluation and test datasets. Because CrossEncoder inputs from
            multiple columns are combined into pairs, per-column prompts are not supported. Two formats are accepted:

            1. `str`: A single prompt to use for all datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your
               training/evaluation/test datasets are a :class:`datasets.DatasetDict` or a dictionary of
               :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.sentence_transformer.training_args.BatchSamplers`, `str`, :class:`~sentence_transformers.base.sampler.DefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.base.sampler.DefaultBatchSampler`]], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.sentence_transformer.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.sentence_transformer.training_args.MultiDatasetBatchSamplers`, `str`, :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler`, Callable[[...], :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler`]], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.sentence_transformer.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
        router_mapping (`Dict[str, str]`, *optional*):
            A mapping of dataset names to Router routes, like "slow", "fast". Because CrossEncoder inputs from multiple
            columns are combined into pairs, per-column router mappings are not supported. Only a per-dataset
            mapping is accepted, e.g. ``{'dataset_a': 'slow', 'dataset_b': 'fast'}``.
        learning_rate_mapping (`Dict[str, float] | None`, *optional*):
            A mapping of parameter name regular expressions to learning rates. This allows you to set different
            learning rates for different parts of the model, e.g., `{'SparseStaticEmbedding\.*': 1e-3}` for the
            SparseStaticEmbedding module. This is useful when you want to fine-tune specific parts of the model
            with different learning rates.
    """
