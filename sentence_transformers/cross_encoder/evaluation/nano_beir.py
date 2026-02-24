from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from sentence_transformers.cross_encoder.evaluation.nano_evaluator import CrossEncoderNanoEvaluator

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]

DATASET_NAME_TO_HUMAN_READABLE: dict[str, str] = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class CrossEncoderNanoBEIREvaluator(CrossEncoderNanoEvaluator):
    """Evaluate a CrossEncoder model on NanoBEIR datasets.

    This evaluator is designed for reranking over NanoBEIR datasets. Rather
    than reranking the full corpus, it reranks only ``rerank_k`` first-stage
    candidates (``bm25`` by default), and reports MAP / MRR@k / nDCG@k.

    This class preserves the NanoBEIR short-name API while delegating shared
    reranking dataset mechanics to
    :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoEvaluator`.

    .. note::
        Result keys follow the NanoBEIR convention:
        ``NanoBEIR_R{rerank_k}_{aggregate_key}_{metric}``.
        Per-dataset keys use ``Nano{DatasetName}_R{rerank_k}_{metric}``.

    .. warning::
        By default (when ``dataset_names`` is not provided), this evaluator
        excludes ``arguana`` and ``touche2020`` because their argument-retrieval
        setup differs from the other NanoBEIR subsets.

    Args:
        dataset_names (List[str]): NanoBEIR short names to evaluate. If not
            provided, defaults to all except ``arguana`` and ``touche2020``.
        dataset_id (str): Hugging Face dataset ID. Must include ``corpus``,
            ``queries``, ``qrels``, and candidate subset data.
        rerank_k (int): Number of candidates to rerank per query.
        at_k (int): Metric cutoff for MRR/nDCG.
        always_rerank_positives (bool): Whether to enforce positives in rerank pool.
        batch_size (int): Cross-encoder evaluation batch size.
        show_progress_bar (bool): Whether to show progress bars.
        write_csv (bool): Whether to write CSV metrics.
        aggregate_fn (Callable[[list[float]], float]): Aggregation function across subsets.
        aggregate_key (str): Aggregate metric key prefix.
        candidate_subset_name (str): First-stage candidate subset name (default: ``bm25``).
        bm25_subset_name (str, optional): Deprecated alias for ``candidate_subset_name``.
        retrieved_corpus_ids_column (str): Column name containing candidate corpus IDs.
    """

    def __init__(
        self,
        dataset_names: list[DatasetNameType | str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        rerank_k: int = 100,
        at_k: int = 10,
        always_rerank_positives: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        candidate_subset_name: str = "bm25",
        bm25_subset_name: str | None = None,
        retrieved_corpus_ids_column: str = "corpus-ids",
    ) -> None:
        if dataset_names is None:
            dataset_names = [
                key for key in DATASET_NAME_TO_HUMAN_READABLE.keys() if key not in ["arguana", "touche2020"]
            ]

        super().__init__(
            dataset_names=[str(name) for name in dataset_names],
            dataset_id=dataset_id,
            rerank_k=rerank_k,
            at_k=at_k,
            always_rerank_positives=always_rerank_positives,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            dataset_name_to_human_readable=DATASET_NAME_TO_HUMAN_READABLE,
            split_prefix="Nano",
            strict_dataset_name_validation=True,
            auto_expand_splits_when_dataset_names_none=False,
            candidate_subset_name=candidate_subset_name,
            bm25_subset_name=bm25_subset_name,
            retrieved_corpus_ids_column=retrieved_corpus_ids_column,
            name="NanoBEIR",
        )

    def _validate_dataset_names(self) -> None:
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")

        missing_datasets = [
            dataset_name
            for dataset_name in self.dataset_names
            if dataset_name.lower() not in DATASET_NAME_TO_HUMAN_READABLE
        ]
        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} are not valid NanoBEIR datasets. "
                f"Valid dataset names are: {list(DATASET_NAME_TO_HUMAN_READABLE.keys())}"
            )

    def get_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "candidate_subset_name": self.candidate_subset_name,
            "retrieved_corpus_ids_column": self.retrieved_corpus_ids_column,
        }
        if self._bm25_subset_name_alias_input is not None:
            config_dict["bm25_subset_name"] = self.candidate_subset_name
        return config_dict
