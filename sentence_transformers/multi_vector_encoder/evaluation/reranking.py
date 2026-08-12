from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers.sentence_transformer.evaluation.reranking import RerankingEvaluator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorRerankingEvaluator(RerankingEvaluator):
    """Reranking evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Scores each query's candidate documents with the model's multi-vector similarity (MaxSim by
    default) and reports MAP, MRR@k, and NDCG@k, treating the ``positive`` documents as the relevance
    ground truth. Useful for evaluating a multi-vector model as a second-stage reranker: a first-stage
    retriever returns candidates per query (positives mixed with distractors) and the multi-vector
    model rescores them.

    See :class:`~sentence_transformers.sentence_transformer.evaluation.RerankingEvaluator` for the full
    argument list and the ``samples`` format. This subclass differs only by resolving the scoring from
    the model at call time and encoding queries / documents asymmetrically (via ``encode_query`` /
    ``encode_document``).

    Example:
        ::

            from datasets import load_dataset

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorRerankingEvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            # Each sample pairs a query with its relevant documents and the first-stage distractors.
            eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(100))
            samples = [
                {
                    "query": sample["query"],
                    "positive": [
                        text
                        for selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
                        if selected
                    ],
                    "negative": [
                        text
                        for selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
                        if not selected
                    ],
                }
                for sample in eval_dataset
            ]

            evaluator = MultiVectorRerankingEvaluator(samples=samples, name="ms-marco-dev")
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(
        self,
        samples: list[dict],
        *,
        at_k: int = 10,
        name: str = "",
        write_csv: bool = True,
        similarity_fct: Callable[[Tensor | list[Tensor], list[Tensor]], Tensor] | None = None,
        batch_size: int = 16,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
    ) -> None:
        self._user_similarity_fct = similarity_fct
        super().__init__(
            samples=samples,
            at_k=at_k,
            name=name,
            write_csv=write_csv,
            similarity_fct=similarity_fct,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            use_batched_encoding=use_batched_encoding,
        )

    def __call__(self, model: MultiVectorEncoder, *args, **kwargs) -> dict[str, float]:
        if self._user_similarity_fct is None:
            # The parent calls similarity_fct(query_emb, docs_emb) with the query as a single token
            # matrix (batched path) or a length-1 list (per-sample path): normalise to a one-query
            # batch for the model's all-pairs similarity.
            def one_to_many(query_embeddings: Tensor | list[Tensor], document_embeddings: list[Tensor]) -> Tensor:
                queries = query_embeddings if isinstance(query_embeddings, list) else [query_embeddings]
                return model.similarity(queries, document_embeddings)

            self.similarity_fct = one_to_many
        return super().__call__(model, *args, **kwargs)

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | Sequence[SingleInput],
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
