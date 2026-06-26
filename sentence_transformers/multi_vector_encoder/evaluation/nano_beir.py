from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentence_transformers.multi_vector_encoder.evaluation.information_retrieval import (
    MultiVectorInformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.evaluation.nano_beir import NanoBEIREvaluator

if TYPE_CHECKING:
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorNanoBEIREvaluator(NanoBEIREvaluator):
    """Evaluates a :class:`~sentence_transformers.MultiVectorEncoder` model on the
    `NanoBEIR collection <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_.

    NanoBEIR is a downsized version of `BEIR <https://github.com/beir-cellar/beir>`_ (around 50 queries
    and 5,000 documents per subset) used for quick retrieval-quality benchmarking before running a
    full-scale BEIR evaluation. This evaluator runs a :class:`MultiVectorInformationRetrievalEvaluator`
    on each requested Nano-* subset, reports the same IR metrics per dataset (MRR@k, NDCG@k, Recall@k,
    Precision@k, Accuracy@k, MAP@k), and aggregates them across datasets at the end.

    Args:
        dataset_names (List[str], optional): Short names of NanoBEIR subsets to evaluate
            (``"climatefever"``, ``"dbpedia"``, ``"fever"``, ``"fiqa2018"``, ``"hotpotqa"``,
            ``"msmarco"``, ``"nfcorpus"``, ``"nq"``, ``"quoraretrieval"``, ``"scidocs"``,
            ``"arguana"``, ``"scifact"``, ``"touche2020"``). Defaults to every subset.
        dataset_id (str): The HuggingFace dataset ID hosting the corpus / queries / qrels subsets.
            Defaults to ``"sentence-transformers/NanoBEIR-en"``. Swap in a translated variant from the
            NanoBEIR collection for non-English evaluation.
        corpus_chunk_size (int): How many documents to encode + score per round-trip. Larger values
            mean more encoded doc embeddings live in memory at once but fewer encode-pass round-trips.
            Defaults to 50000.
        document_chunk_size (int, optional): Per-call chunk size for the MaxSim matmul. Bounds the 4D
            ``(batch_q, chunk, q_tokens, d_tokens)`` scoring intermediate independently of
            ``corpus_chunk_size``. Defaults to 32. Pass ``None`` to disable inner chunking.
        mrr_at_k (List[int]): k-values for MRR. Defaults to ``[10]``.
        ndcg_at_k (List[int]): k-values for NDCG. Defaults to ``[10]``.
        accuracy_at_k (List[int]): k-values for accuracy. Defaults to ``[1, 3, 5, 10]``.
        precision_recall_at_k (List[int]): k-values for precision and recall. Defaults to ``[1, 3, 5, 10]``.
        map_at_k (List[int]): k-values for MAP. Defaults to ``[100]``.
        show_progress_bar (bool): Show a progress bar during evaluation. Defaults to False.
        batch_size (int): Per-input batch size used while encoding. Defaults to 32.
        write_csv (bool): Append per-call metric values to a CSV file (one row per evaluation call).
            Defaults to True.
        truncate_dim (int, optional): Single-vector dimension truncation. The multi-vector encoder
            ignores this. Defaults to None.
        score_functions (Dict[str, Callable], optional): Override the default per-subset scoring.
            See :class:`MultiVectorInformationRetrievalEvaluator`. Defaults to None.
        main_score_function (str or SimilarityFunction, optional): Score-function key to treat as the
            primary metric for the model card / trainer. Defaults to None.
        aggregate_fn (Callable[[List[float]], float]): How to aggregate the per-subset scores into the
            top-level summary. Defaults to ``np.mean``.
        aggregate_key (str): Suffix used in the aggregated metric's key. Defaults to ``"mean"``.
        query_prompts (str or Dict[str, str], optional): Prompt prepended to queries. A string applies
            to every dataset. A dict keyed by ``dataset_names`` entries lets you set per-dataset
            prompts. Defaults to None.
        corpus_prompts (str or Dict[str, str], optional): Prompt prepended to corpus documents.
            Same string / dict semantics as ``query_prompts``. Defaults to None.
        write_predictions (bool): Write per-query top-k predictions to a JSONL file, suitable as
            input to :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator`.
            Defaults to False.

    Example:
        ::

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus"])
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    information_retrieval_class = MultiVectorInformationRetrievalEvaluator

    def __init__(
        self,
        *args,
        corpus_chunk_size: int = 50000,
        document_chunk_size: int | None = 32,
        **kwargs,
    ) -> None:
        # Set before super().__init__ because the base initializer iterates ``_load_dataset`` to
        # construct per-subset IR evaluators, and that needs the extra kwargs.
        self._ir_extra_kwargs: dict[str, Any] = {
            "corpus_chunk_size": corpus_chunk_size,
            "document_chunk_size": document_chunk_size,
        }
        super().__init__(*args, **kwargs)

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs):
        ir_evaluator_kwargs = {**ir_evaluator_kwargs, **self._ir_extra_kwargs}
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)

    def __call__(
        self,
        model: MultiVectorEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        # Overridden only to narrow the model type to MultiVectorEncoder; behavior is identical to the parent.
        return super().__call__(model, output_path, epoch, steps, *args, **kwargs)
