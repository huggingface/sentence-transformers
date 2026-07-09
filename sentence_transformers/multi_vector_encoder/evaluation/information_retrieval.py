from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from torch import Tensor

from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.util.similarity import SimilarityFunction, maxsim

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """Evaluates a :class:`~sentence_transformers.MultiVectorEncoder` model on an information-retrieval
    (IR) task. For each query, the model retrieves the top-k closest documents from the corpus using
    ColBERT-style MaxSim scoring, then reports the standard IR metrics (MRR@k, NDCG@k, Recall@k,
    Precision@k, Accuracy@k, MAP@k) against the supplied relevance judgements.

    Args:
        queries (Dict[str, str]): Mapping of query IDs to query text.
        corpus (Dict[str, str]): Mapping of document IDs to document text.
        relevant_docs (Dict[str, Set[str]]): Mapping of query ID to the set of relevant document IDs.
        corpus_chunk_size (int): How many documents to encode and score per round-trip. Larger values
            mean more encoded doc embeddings live in memory at once but fewer encode-pass round-trips.
            Defaults to 50000.
        document_chunk_size (int, optional): Per-call chunk size for the MaxSim matmul. Bounds the 4D
            ``(batch_q, chunk, q_tokens, d_tokens)`` scoring intermediate independently of
            ``corpus_chunk_size``. Defaults to 32. Pass ``None`` to score the whole ``corpus_chunk_size``
            in one shot.
        score_functions (Dict[str, Callable], optional): Override the default ``{"maxsim": maxsim}``
            scoring. The chosen callable receives ``(queries, documents)`` token tensors and must
            return a ``(num_queries, num_documents)`` score matrix. XTR scoring is not supported here
            because it does a global top-k across the whole candidate set, which is incompatible with
            this evaluator's per-chunk corpus scoring.
        mrr_at_k (List[int]): k-values for MRR. Defaults to ``[10]``.
        ndcg_at_k (List[int]): k-values for NDCG. Defaults to ``[10]``.
        accuracy_at_k (List[int]): k-values for accuracy. Defaults to ``[1, 3, 5, 10]``.
        precision_recall_at_k (List[int]): k-values for precision and recall. Defaults to ``[1, 3, 5, 10]``.
        map_at_k (List[int]): k-values for MAP. Defaults to ``[100]``.
        show_progress_bar (bool): Show a progress bar during evaluation. Defaults to False.
        batch_size (int): Per-input batch size used while encoding. Defaults to 32.
        name (str): Evaluation name (used as the dataset stem in CSV / prediction filenames).
            Defaults to ``""``.
        write_csv (bool): Append per-call metric values to ``Information-Retrieval_evaluation_<name>_results.csv``.
            Defaults to True.
        truncate_dim (int, optional): Not supported: multi-vector token embeddings have no
            Matryoshka-style truncation, so any non-None value raises a ValueError. Defaults to None.
        main_score_function (str or SimilarityFunction, optional): Which score-function key to treat
            as the primary metric for the model card / trainer. Defaults to None (use the first /
            only key in ``score_functions``).
        query_prompt (str, optional): Prompt prepended to every query during encoding. Defaults to None.
        query_prompt_name (str, optional): Name of a prompt registered on the model to prepend to
            queries. Mutually exclusive with ``query_prompt``. Defaults to None.
        corpus_prompt (str, optional): Prompt prepended to every corpus document. Defaults to None.
        corpus_prompt_name (str, optional): Name of a prompt registered on the model to prepend to
            corpus documents. Mutually exclusive with ``corpus_prompt``. Defaults to None.
        write_predictions (bool): Write per-query top-k predictions to a JSONL file, suitable as
            input to :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator`.
            Defaults to False.

    Example:
        ::

            from datasets import load_dataset

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            # Load NanoMSMARCO subsets and convert to the evaluator's dict format.
            corpus_ds = load_dataset("sentence-transformers/NanoBEIR-en", "corpus", split="NanoMSMARCO")
            queries_ds = load_dataset("sentence-transformers/NanoBEIR-en", "queries", split="NanoMSMARCO")
            qrels_ds = load_dataset("sentence-transformers/NanoBEIR-en", "qrels", split="NanoMSMARCO")

            corpus = {row["_id"]: row["text"] for row in corpus_ds}
            queries = {row["_id"]: row["text"] for row in queries_ds}
            relevant_docs = {}
            for row in qrels_ds:
                relevant_docs.setdefault(row["query-id"], set()).add(row["corpus-id"])

            evaluator = MultiVectorInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="NanoMSMARCO",
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(
        self,
        queries: dict[str, SingleInput],
        corpus: dict[str, SingleInput],
        relevant_docs: dict[str, set[str]],
        corpus_chunk_size: int = 50000,
        document_chunk_size: int | None = 32,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("truncate_dim") is not None:
            raise ValueError(
                "truncate_dim is not supported by MultiVectorEncoder evaluators: multi-vector token "
                "embeddings have no Matryoshka-style truncation. Remove truncate_dim to evaluate at "
                "the full dimension."
            )
        if score_functions is None:
            scoring_fn = (
                maxsim if document_chunk_size is None else partial(maxsim, document_chunk_size=document_chunk_size)
            )
            score_functions = {SimilarityFunction.MAXSIM.value: scoring_fn}
        else:
            # XTR's global top-k would be taken per corpus chunk, silently wrong for any corpus > corpus_chunk_size.
            from sentence_transformers.multi_vector_encoder.scoring import XTRScores, xtr_scores

            for name, fn in score_functions.items():
                target = fn.func if isinstance(fn, partial) else fn
                if target is xtr_scores or isinstance(target, XTRScores):
                    raise ValueError(
                        f"score_functions[{name!r}] uses XTR scoring, which is incompatible with this "
                        "evaluator's per-chunk corpus scoring (top-k would be per-chunk). Use MaxSim instead."
                    )
        super().__init__(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            corpus_chunk_size=corpus_chunk_size,
            score_functions=score_functions,
            **kwargs,
        )

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | list[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> list[Tensor] | Tensor:
        # MultiVectorEncoder.encode doesn't accept truncate_dim, drop it before forwarding.
        kwargs.pop("truncate_dim", None)
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        # Pre-pad queries (reused across every corpus chunk) so per-chunk maxsim does the cheap zero-row
        # mask check instead of re-running pad_sequence each round. Documents are encoded per chunk.
        return encode_fn(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            convert_to_padded_tensor=encode_fn_name == "query",
            **kwargs,
        )
