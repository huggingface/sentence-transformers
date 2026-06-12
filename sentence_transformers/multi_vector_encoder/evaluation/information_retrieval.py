from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor

from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.util.similarity import SimilarityFunction, maxsim

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """Information-retrieval evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Extends the dense :class:`~sentence_transformers.evaluation.InformationRetrievalEvaluator` with:

    1. Multi-vector encoding via :meth:`encode_query` / :meth:`encode_document` (returning per-input
       variable-length token-embedding tensors).
    2. Default scoring with :func:`~sentence_transformers.util.maxsim`.
    3. A lower default ``corpus_chunk_size`` of 5,000 (versus 50,000 for the dense evaluator): MaxSim
       intermediates of shape ``(batch_q, chunk, q_tokens, d_tokens)`` are much larger than the dense
       ``(batch_q, chunk)`` matrix, so the default is tuned for memory.

    All other arguments and the metric definitions (MRR@k, NDCG@k, Recall@k, MAP@k, Accuracy@k) are
    identical to the parent.
    """

    def __init__(
        self,
        queries: dict[str, str],
        corpus: dict[str, str],
        relevant_docs: dict[str, set[str]],
        corpus_chunk_size: int = 5000,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        **kwargs,
    ) -> None:
        if score_functions is None:
            score_functions = {SimilarityFunction.MAXSIM.value: maxsim}
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
        sentences: str | list[str] | np.ndarray,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> list[Tensor]:
        # MultiVectorEncoder.encode doesn't accept truncate_dim; drop it before forwarding.
        kwargs.pop("truncate_dim", None)
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        return encode_fn(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            **kwargs,
        )
