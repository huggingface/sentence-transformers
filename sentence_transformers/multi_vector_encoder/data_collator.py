from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class MultiVectorEncoderDataCollator(BaseDataCollator):
    """Data collator for :class:`~sentence_transformers.MultiVectorEncoder` training.

    Differs from :class:`~sentence_transformers.base.data_collator.BaseDataCollator` only in the
    default task assignment: when ``router_mapping`` does not specify a task for a column, the first
    column defaults to ``"query"`` and the rest to ``"document"``. This matches the losses, which
    assign positionally (column 0 = query). Column names are not consulted. Use ``router_mapping``
    to override per-column.
    """

    def _get_task_for_column(self, column_name: str, column_position: int, router_mapping: dict[str, str]) -> str:
        task = router_mapping.get(column_name)
        if task is None:
            task = "query" if column_position == 0 else "document"
        return task
