from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from sentence_transformers.base.training_args import BaseTrainingArguments


@dataclass
class MultiVectorEncoderTrainingArguments(BaseTrainingArguments):
    """Training arguments for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` training.

    Inherits all fields from :class:`~sentence_transformers.base.training_args.BaseTrainingArguments`, and adds:

    Args:
        max_length (`Union[int, Dict[str, int]]`, *optional*):
            Maximum token length applied when tokenizing training and evaluation-loss batches, without changing
            the model's own configuration: evaluators, inference, and the saved model keep the lengths configured
            on the model (set those via ``processor_kwargs`` globally or ``processing_kwargs`` per call). Two
            formats are accepted:

            1. `int`: One maximum length for every column.
            2. `Dict[str, int]`: A mapping of tasks (e.g. ``"query"``, ``"document"``) to maximum lengths. The
               data collator gives column 0 the ``"query"`` task and the rest ``"document"``, which
               ``router_mapping`` overrides per column.

            Tight training caps can be much faster and measurably stronger than uncapped training, see the
            multi-vector MS MARCO knowledge distillation example. Queries under a fixed-width
            ``query_expansion`` (``strategy="fixed"``) ignore this override, as the expansion length already
            fixes their width. Under ``strategy="min"`` it applies as the ceiling, never below the expansion
            length.
    """

    _VALID_DICT_FIELDS = [*BaseTrainingArguments._VALID_DICT_FIELDS, "max_length"]

    max_length: Union[int, None, dict[str, int]] = field(  # noqa: UP007
        default=None,
        metadata={
            "help": "Maximum token length for training and evaluation-loss tokenization. Either 1) an int "
            "applied to every column, or 2) a mapping of tasks ('query', 'document') to lengths. Evaluators, "
            "inference, and the saved model keep the model's own configuration."
        },
    )
