from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentence_transformers.multi_vector_encoder.evaluation.information_retrieval import (
    MultiVectorInformationRetrievalEvaluator,
)
from sentence_transformers.multi_vector_encoder.evaluation.utils import pooling_config, pooling_name_suffix
from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling
from sentence_transformers.sentence_transformer.evaluation.nano_retrieval import NanoDatasetId, NanoRetrievalEvaluator

if TYPE_CHECKING:
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorNanoRetrievalEvaluator(NanoRetrievalEvaluator):
    """Evaluate a :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` on a Nano
    text retrieval collection using late interaction.

    Compatible collections are listed in the
    `Nano datasets collection <https://huggingface.co/collections/sentence-transformers/nano-datasets>`_.

    The dataset must contain corpus, queries, and qrels subsets as described by
    :class:`~sentence_transformers.sentence_transformer.evaluation.NanoRetrievalEvaluator`. This evaluator runs a
    :class:`MultiVectorInformationRetrievalEvaluator` on each requested split, reports the same retrieval
    metrics per split (MRR@k, NDCG@k, Recall@k, Precision@k, Accuracy@k, MAP@k), and aggregates them across splits.

    Args:
        dataset_id (NanoDatasetId | str): Hub dataset ID containing corpus, queries, and qrels subsets.
            ``NanoDatasetId`` provides autocomplete suggestions for known compatible collections.
            Other dataset IDs are accepted if they follow the same format.
        splits (list[str], optional): Tasks to evaluate. Defaults to all qrels splits.
        name (str, optional): Collection name used as the prefix of metric names and output files.
            Defaults to the last part of ``dataset_id``, e.g. ``"NanoMTEB-Dutch"``.
        corpus_chunk_size (int): Documents to encode and score per chunk. Larger values keep more document
            embeddings in memory but require fewer encoding passes. Defaults to 5000.
        chunk_elements (int, optional): Element budget for the 4D
            ``(batch_q, chunk, q_tokens, d_tokens)`` MaxSim scoring intermediate, forwarded to
            :func:`~sentence_transformers.util.similarity.maxsim`. Defaults to None (maxsim's 100M-element
            budget, at most about 400 MB in float32 or 200 MB in bf16 / fp16). Lower it to reduce scoring memory.
        token_pooling (BaseTokenPooling, optional): Pooling applied during encoding. The pooler's ``tasks``
            select which inputs are pooled. Explicit poolers add a short configuration label to evaluator names.
            Use distinct collection names for poolers with the same label. Defaults to None (use the model's pooler).
        mrr_at_k (list[int]): Rank cutoffs for MRR. Defaults to ``[10]``.
        ndcg_at_k (list[int]): Rank cutoffs for NDCG. Defaults to ``[10]``.
        accuracy_at_k (list[int]): Rank cutoffs for accuracy. Defaults to ``[1, 3, 5, 10]``.
        precision_recall_at_k (list[int]): Rank cutoffs for precision and recall. Defaults to ``[1, 3, 5, 10]``.
        map_at_k (list[int]): Rank cutoffs for MAP. Defaults to ``[100]``.
        show_progress_bar (bool): Show evaluation progress. Defaults to False.
        batch_size (int): Batch size used while encoding. Defaults to 32.
        write_csv (bool): Append per-call metric values to CSV files. Defaults to True.
        truncate_dim (int, optional): Truncate and L2-normalize each token embedding before scoring.
            An explicit dimension is appended to evaluator names. Defaults to None (use the model's default dimension).
        score_functions (dict[str, Callable], optional): Override scoring, which defaults to the model's
            ``similarity_fn_name`` at call time. Callables receive query and document token tensors and return
            a ``(num_queries, num_documents)`` score matrix. Cannot be combined with ``chunk_elements``.
            XTR scoring is not supported because it requires global scoring across the corpus. Defaults to None.
        main_score_function (str or SimilarityFunction, optional): Score-function key to use for the primary metric.
            Defaults to None (select the score function with the highest aggregate NDCG at the largest cutoff).
        aggregate_fn (Callable[[list[float]], float]): Function to aggregate per-split scores. Defaults to ``np.mean``.
        aggregate_key (str): Aggregate metric suffix. Defaults to ``"mean"``.
        query_prompts (str or dict[str, str], optional): Prompt prepended to queries. A string applies to every split.
            A dict must contain every selected split as a key. Defaults to None (use the model's query prompt).
        corpus_prompts (str or dict[str, str], optional): Prompt prepended to corpus documents, with the same
            string / dict semantics as ``query_prompts``. Defaults to None (use the model's document prompt).
        write_predictions (bool): Write per-query top-k predictions to JSONL files, suitable as input to
            :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator`. Defaults to False.

    Example:
        ::

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoRetrievalEvaluator

            model = MultiVectorEncoder("lightonai/mLateOn")
            evaluator = MultiVectorNanoRetrievalEvaluator("hakari-bench/NanoMTEB-Dutch", batch_size=8)
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    information_retrieval_class = MultiVectorInformationRetrievalEvaluator

    def __init__(
        self,
        dataset_id: NanoDatasetId | str,
        splits: list[str] | None = None,
        name: str | None = None,
        *,
        corpus_chunk_size: int = 5000,
        chunk_elements: int | None = None,
        token_pooling: BaseTokenPooling | None = None,
        **kwargs,
    ):
        self._ir_extra_kwargs = {
            "corpus_chunk_size": corpus_chunk_size,
            "chunk_elements": chunk_elements,
            "token_pooling": token_pooling,
        }
        self.token_pooling = token_pooling
        super().__init__(dataset_id=dataset_id, splits=splits, name=name, **kwargs)
        self.csv_file = f"{self.dataset_name}_evaluation_{self.name.removeprefix(self.dataset_name + '_')}_results.csv"

    def _get_aggregate_name(self) -> str:
        name = super()._get_aggregate_name()
        if self.token_pooling is not None:
            name += f"_{pooling_name_suffix(self.token_pooling)}"
        return name

    def _get_human_readable_name(self, dataset_name: str) -> str:
        human_readable_name = super()._get_human_readable_name(dataset_name)
        if self.token_pooling is not None:
            human_readable_name += f"_{pooling_name_suffix(self.token_pooling)}"
        return human_readable_name

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs):
        ir_evaluator_kwargs = {**ir_evaluator_kwargs, **self._ir_extra_kwargs}
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.token_pooling is not None:
            config_dict["token_pooling"] = pooling_config(self.token_pooling)
        return config_dict

    def __call__(
        self,
        model: MultiVectorEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        return super().__call__(model, output_path, epoch, steps, *args, **kwargs)
