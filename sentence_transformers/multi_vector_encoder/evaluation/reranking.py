from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers.sentence_transformer.evaluation.reranking import RerankingEvaluator
from sentence_transformers.util.similarity import maxsim

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


def _maxsim_one_to_many(query_embeddings: Tensor | list[Tensor], document_embeddings: list[Tensor]) -> Tensor:
    """MaxSim score of a single query against many documents, returning shape ``(1, num_documents)``.

    :class:`RerankingEvaluator` calls ``similarity_fct(query_emb, docs_emb)`` with the query as a single
    token matrix (batched path) or a length-1 list (per-sample path); both are normalised to a one-query
    batch for :func:`~sentence_transformers.util.maxsim`.
    """
    queries = query_embeddings if isinstance(query_embeddings, list) else [query_embeddings]
    return maxsim(queries, document_embeddings)


class MultiVectorRerankingEvaluator(RerankingEvaluator):
    """Reranking evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Scores each query's candidate documents with MaxSim and reports MAP, MRR@k, and NDCG@k, treating the
    ``positive`` documents as the relevance ground truth. Useful for evaluating a multi-vector model as a
    second-stage reranker: a first-stage retriever returns candidates per query (positives mixed with
    distractors) and the multi-vector model rescores them.

    See :class:`~sentence_transformers.sentence_transformer.evaluation.RerankingEvaluator` for the full
    argument list and the ``samples`` format. This subclass differs only by scoring with MaxSim and encoding
    queries / documents asymmetrically (via ``encode_query`` / ``encode_document``).
    """

    def __init__(
        self,
        samples: list[dict],
        at_k: int = 10,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ) -> None:
        super().__init__(
            samples=samples,
            at_k=at_k,
            name=name,
            write_csv=write_csv,
            similarity_fct=_maxsim_one_to_many,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences,
        encode_fn_name: str | None = None,
        show_progress_bar: bool | None = None,
        **kwargs,
    ) -> list[Tensor]:
        # MultiVectorEncoder.encode returns a ragged list[Tensor] and does not accept truncate_dim.
        kwargs.pop("truncate_dim", None)
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            **kwargs,
        )
