from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from transformers.utils import logging as transformers_logging

from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


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
            Defaults to 5000.
        document_chunk_elements (int, optional): Element budget for the 4D
            ``(batch_q, chunk, q_tokens, d_tokens)`` MaxSim scoring intermediate, forwarded to
            :func:`~sentence_transformers.util.maxsim`, which packs document chunks under it,
            adapting to the query count and document lengths. Defaults to None (maxsim's 100M-element
            budget, at most ~400 MB, half that in bf16 / fp16). Lower it to cut evaluation memory.
        score_functions (Dict[str, Callable], optional): Override the default scoring, which resolves
            from the model's ``similarity_fn_name`` at call time (with ``document_chunk_elements``
            applied if one was given). The chosen callable receives ``(queries, documents)`` token tensors and must
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
        *,
        corpus_chunk_size: int = 5000,
        document_chunk_elements: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("truncate_dim") is not None:
            raise ValueError(
                "truncate_dim is not supported by MultiVectorEncoder evaluators: multi-vector token "
                "embeddings have no Matryoshka-style truncation. Remove truncate_dim to evaluate at "
                "the full dimension."
            )
        if score_functions is not None:
            if document_chunk_elements is not None:
                raise ValueError(
                    "document_chunk_elements only configures the default model-resolved scoring, so it "
                    "would be silently ignored alongside score_functions. Bind the budget into your own "
                    "callable instead, e.g. functools.partial(maxsim, document_chunk_elements=...)."
                )
            # XTR's global top-k would be taken per corpus chunk, silently wrong for any corpus > corpus_chunk_size.
            from sentence_transformers.multi_vector_encoder.scoring import XTRScores, xtr_scores

            for name, fn in score_functions.items():
                target = fn.func if isinstance(fn, partial) else fn
                # torch.compile wraps the callable (the XTR docstring itself recommends compiling): unwrap.
                target = getattr(target, "_torchdynamo_orig_callable", target)
                if target is xtr_scores or isinstance(target, XTRScores):
                    raise ValueError(
                        f"score_functions[{name!r}] uses XTR scoring, which is incompatible with this "
                        "evaluator's per-chunk corpus scoring (top-k would be per-chunk). Use MaxSim instead."
                    )
        # When score_functions is None, scoring resolves from the model at call time (see __call__),
        # so models carrying a different multi-vector similarity are scored and labeled with it.
        self.document_chunk_elements = document_chunk_elements
        super().__init__(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            corpus_chunk_size=corpus_chunk_size,
            score_functions=score_functions,
            **kwargs,
        )

    def __call__(
        self,
        model: MultiVectorEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        for explicit_prompt, param_name, role_keys in (
            (self.query_prompt, "query_prompt", ("query",)),
            (self.corpus_prompt, "corpus_prompt", ("document", "passage", "corpus")),
        ):
            if explicit_prompt is None:
                continue
            # Mirror encode_query / encode_document: the first present role key is the prompt the
            # model would have used.
            role = next((key for key in role_keys if key in model.prompts), None)
            registered = model.prompts.get(role) if role is not None else None
            if registered and not explicit_prompt.startswith(registered):
                logger.warning_once(
                    f"The explicit {param_name} replaces the model's registered {role!r} prompt "
                    f"({registered!r}) rather than composing with it. Include the model's "
                    f"marker prompt in {param_name} if the trained prompt should be kept."
                )
        # Resolve the default scoring from the model here instead of letting the parent fall back to
        # bare model.similarity. Without an explicit document_chunk_elements, maxsim's default
        # element budget bounds the scoring intermediate on its own.
        if self.score_functions is None:
            scoring_fn: Callable[..., Tensor] = SimilarityFunction.to_similarity_fn(model.similarity_fn_name)
            if self.document_chunk_elements is not None:
                scoring_fn = partial(scoring_fn, document_chunk_elements=self.document_chunk_elements)
            self.score_functions = {model.similarity_fn_name: scoring_fn}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)
        return super().__call__(model, output_path, epoch, steps, *args, **kwargs)

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | Sequence[SingleInput] | np.ndarray,
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
        embeddings = encode_fn(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            **kwargs,
        )
        if encode_fn_name == "query":
            # Pre-pad queries (reused across every corpus chunk) so per-chunk maxsim does the cheap
            # zero-row mask check instead of re-running pad_sequence each round. Documents are
            # encoded per chunk.
            embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        return embeddings
