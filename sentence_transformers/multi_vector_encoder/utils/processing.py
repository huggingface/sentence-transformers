from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from transformers.utils import logging as transformers_logging

if TYPE_CHECKING:
    import datasets

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


# TODO: Do I want to advertise/support this? If so, how do I want to structure it?
class KDProcessing:
    """Dataset processing class for knowledge-distillation training.

    Joins a ``(query_id, document_ids, scores)`` dataset against external ``queries`` and ``documents``
    datasets (typically loaded from a HF Hub config like ``lightonai/ms-marco-en-bge``) so that the
    in-memory batch contains the resolved ``query`` text, the ``documents`` list of texts, and the matching
    ``scores`` list. Supports an ``n_ways`` cap to truncate the per-row document / score list.

    Pass the result via :meth:`datasets.Dataset.set_transform` (lazy, no caching) or :meth:`map` (eager).

    Args:
        queries: Dataset (or DatasetDict) with ``query_id`` and ``text`` columns.
        documents: Dataset (or DatasetDict) with ``document_id`` and ``text`` columns.
        split: Split name to read when the inputs are :class:`datasets.DatasetDict`. Defaults to ``"train"``.
        n_ways: Maximum number of (document_id, score) pairs to keep per row, taking the first n as
            stored. Note that this is not a top-n by teacher score: e.g. the ``lightonai/ms-marco-en-bge``
            rows are not score-sorted. Defaults to 32.

    Example:
        ::

            from datasets import load_dataset
            from sentence_transformers.multi_vector_encoder.utils import KDProcessing

            train = load_dataset("lightonai/ms-marco-en-bge", "train", split="train")
            queries = load_dataset("lightonai/ms-marco-en-bge", "queries", split="train")
            documents = load_dataset("lightonai/ms-marco-en-bge", "documents", split="train")

            train.set_transform(KDProcessing(queries=queries, documents=documents).transform)
    """

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        n_ways: int = 32,
    ) -> None:
        try:
            import datasets as _datasets
        except ImportError as e:
            raise ImportError(
                "KDProcessing requires the `datasets` library. Install it with `pip install datasets`."
            ) from e

        self.queries = queries[split] if isinstance(queries, _datasets.DatasetDict) else queries
        self.documents = documents[split] if isinstance(documents, _datasets.DatasetDict) else documents
        self.n_ways = n_ways

        self.queries_index = {qid: i for i, qid in enumerate(self.queries["query_id"])}
        self.documents_index = {did: i for i, did in enumerate(self.documents["document_id"])}

    def transform(self, examples: dict) -> dict:
        """Batched transform suitable for :meth:`datasets.Dataset.set_transform`.

        Returns only the resolved columns, in the order the losses expect: ``query`` (the anchor) first,
        ``documents`` (the per-row candidate list) second, then ``scores`` (the teacher labels). The raw
        ``query_id`` / ``document_ids`` columns are dropped so the first column is the query.
        """
        document_ids = [self._parse_list(d)[: self.n_ways] for d in examples["document_ids"]]
        return {
            "query": [self.queries[self.queries_index[qid]]["text"] for qid in examples["query_id"]],
            "documents": [[self._lookup_document(did) for did in row_ids] for row_ids in document_ids],
            "scores": [self._parse_list(s)[: self.n_ways] for s in examples["scores"]],
        }

    def map(self, example: dict) -> dict:
        """Per-row transform suitable for :meth:`datasets.Dataset.map` (non-batched).

        Returns ``query`` / ``documents`` / ``scores`` in the loss-expected order. Pass
        ``remove_columns=["query_id", "document_ids"]`` to :meth:`~datasets.Dataset.map` to drop the raw
        ID columns so the query ends up first.
        """
        document_ids = self._parse_list(example["document_ids"])[: self.n_ways]
        return {
            "query": self.queries[self.queries_index[example["query_id"]]]["text"],
            "documents": [self._lookup_document(did) for did in document_ids],
            "scores": self._parse_list(example["scores"])[: self.n_ways],
        }

    def _parse_list(self, value):
        if isinstance(value, str):
            logger.warning_once(
                "KDProcessing received a string where a list was expected; falling back to ast.literal_eval. "
                "Consider preprocessing the dataset to native list columns."
            )
            return ast.literal_eval(value)
        return value

    def _lookup_document(self, document_id) -> str:
        try:
            return self.documents[self.documents_index[document_id]]["text"]
        except KeyError:
            logger.warning(f"Unable to find document: {document_id}")
            return ""
