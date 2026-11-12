from __future__ import annotations

from typing import Any

from sentence_transformers.multi_vector_encoder.evaluation.nano_evaluator import MultiVectorNanoEvaluator
from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling
from sentence_transformers.sentence_transformer.evaluation.nano_beir import NanoBEIREvaluator


class MultiVectorNanoBEIREvaluator(NanoBEIREvaluator, MultiVectorNanoEvaluator):
    """Evaluates a :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` model on the
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
            Defaults to 5000.
        chunk_elements (int, optional): Element budget for the 4D
            ``(batch_q, chunk, q_tokens, d_tokens)`` MaxSim scoring intermediate, forwarded to
            :func:`~sentence_transformers.util.similarity.maxsim`, which packs document chunks under it,
            adapting to the query count and document lengths. Defaults to None (maxsim's 100M-element
            budget, at most ~400 MB, half that in bf16 / fp16). Lower it to cut evaluation memory.
        mrr_at_k (List[int]): k-values for MRR. Defaults to ``[10]``.
        ndcg_at_k (List[int]): k-values for NDCG. Defaults to ``[10]``.
        accuracy_at_k (List[int]): k-values for accuracy. Defaults to ``[1, 3, 5, 10]``.
        precision_recall_at_k (List[int]): k-values for precision and recall. Defaults to ``[1, 3, 5, 10]``.
        map_at_k (List[int]): k-values for MAP. Defaults to ``[100]``.
        show_progress_bar (bool): Show a progress bar during evaluation. Defaults to False.
        batch_size (int): Per-input batch size used while encoding. Defaults to 32.
        write_csv (bool): Append per-call metric values to a CSV file (one row per evaluation call).
            Defaults to True.
        truncate_dim (int, optional): Truncate and L2-normalize each token embedding before scoring.
            An explicit dimension is appended to evaluator names. Defaults to None (use the model's
            default dimension).
        token_pooling (BaseTokenPooling, optional): Pooling applied during encoding. The pooler's
            ``tasks`` select which inputs are pooled. Explicit poolers add a short configuration label
            to evaluator names. Defaults to None (use the model's default pooler).
        name (str): Optional prefix for aggregate and per-dataset metric names and output files.
            Use distinct names for poolers with the same short configuration label. Defaults to "".
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

    def __init__(
        self,
        *args,
        corpus_chunk_size: int = 5000,
        chunk_elements: int | None = None,
        token_pooling: BaseTokenPooling | None = None,
        name: str = "",
        **kwargs,
    ) -> None:
        self._name = name
        super().__init__(
            *args,
            corpus_chunk_size=corpus_chunk_size,
            chunk_elements=chunk_elements,
            token_pooling=token_pooling,
            **kwargs,
        )
        if name:
            self.name = f"{name}_{self.name}"
            self.csv_file = f"{name}_{self.csv_file}"

    def _get_human_readable_name(self, dataset_name: str) -> str:
        name = super()._get_human_readable_name(dataset_name)
        return f"{self._name}_{name}" if self._name else name

    def get_config_dict(self) -> dict[str, Any]:
        config = super().get_config_dict()
        if self._name:
            config["name"] = self._name
        return config
