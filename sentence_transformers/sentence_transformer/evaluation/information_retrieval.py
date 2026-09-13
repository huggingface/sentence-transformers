from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.util.similarity import SimilarityFunction
from sentence_transformers.util.statistics import bootstrap_confidence_interval, bootstrap_indices

if TYPE_CHECKING:
    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class InformationRetrievalEvaluator(BaseEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    The per-query values behind every metric are kept in ``per_query_metrics`` after each call, and can optionally be
    summarized with bootstrap confidence intervals over the queries (``bootstrap_resamples``) or sliced per query group
    (``query_groups``).

    Args:
        queries (Dict[str, str]): A dictionary mapping query IDs to queries.
        corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
        relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.
        corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        name (str): A name for the evaluation. Defaults to "".
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function of the model passed to each call, so a reused evaluator follows every model's own similarity.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.
        bootstrap_resamples (int, optional): Number of bootstrap resamples over the queries used to compute a
            confidence interval for every metric, reported as additional ``..._ci_low`` and ``..._ci_high`` keys.
            One set of resamples is shared by all metrics and score functions, so their intervals are comparable.
            Defaults to None, which disables the confidence intervals.
        bootstrap_confidence_level (float): Confidence level of the bootstrap confidence intervals, strictly between
            0 and 1. Defaults to 0.95.
        bootstrap_seed (int, optional): Seed of the bootstrap resampling. Defaults to 42.
        query_groups (Dict[str, str], optional): A dictionary mapping query IDs to a group label. Every metric is
            additionally reported per group as ``..._<group>`` keys (with ``_ci_low`` / ``_ci_high`` variants when
            ``bootstrap_resamples`` is set). Queries that are not in the mapping belong to no group. Defaults to None.

    Example:
        ::

            import logging
            import random

            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
            from datasets import load_dataset

            logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

            # Load a model
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # Load the Touche-2020 IR dataset (https://huggingface.co/datasets/mteb/webis-touche2020-v3)
            corpus = load_dataset("mteb/webis-touche2020-v3", "corpus", split="corpus")
            queries = load_dataset("mteb/webis-touche2020-v3", "queries", split="train")
            relevant_docs_data = load_dataset("mteb/webis-touche2020-v3", "default", split="test")

            # For this dataset, we want to concatenate the title and texts for the corpus
            corpus = corpus.map(lambda x: {"text": (x["title"] + " " + x["text"]).strip()}, remove_columns=["title"])

            # Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
            required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
            required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
            corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

            # Convert the datasets to dictionaries
            corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
            queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
            relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
                qid = str(qid)
                corpus_ids = str(corpus_ids)
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                relevant_docs[qid].add(corpus_ids)

            # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="mteb-touche2020-subset-test",
            )
            results = ir_evaluator(model)
            '''
            Information Retrieval Evaluation of the model on the mteb-touche2020-subset-test dataset:
            Queries: 49
            Corpus: 32446

            Score-Function: cosine
            Accuracy@1: 100.00%
            Accuracy@3: 100.00%
            Accuracy@5: 100.00%
            Accuracy@10: 100.00%
            Precision@1: 100.00%
            Precision@3: 93.88%
            Precision@5: 91.84%
            Precision@10: 91.63%
            Recall@1: 1.76%
            Recall@3: 4.98%
            Recall@5: 8.14%
            Recall@10: 16.25%
            MRR@10: 1.0000
            NDCG@10: 0.9295
            MAP@100: 0.4844
            '''
            print(ir_evaluator.primary_metric)
            # => "mteb-touche2020-subset-test_cosine_ndcg@10"
            print(results[ir_evaluator.primary_metric])
            # => 0.9294944073850905

            # With bootstrap confidence intervals over the queries, every metric line also reports its interval,
            # e.g. "NDCG@10: 0.9295 (95% CI: 0.9012 – 0.9531)", and the results gain "_ci_low" / "_ci_high" keys:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="mteb-touche2020-subset-test",
                bootstrap_resamples=1000,
            )
            results = ir_evaluator(model)
            print(results["mteb-touche2020-subset-test_cosine_ndcg@10_ci_low"])
            print(results["mteb-touche2020-subset-test_cosine_ndcg@10_ci_high"])
            # The per-query values behind the metrics are aligned with ir_evaluator.queries_ids, e.g. for
            # comparing two models on the same queries with sentence_transformers.util.paired_bootstrap_test:
            print(ir_evaluator.per_query_metrics["cosine"]["ndcg@10"].shape)
            # => (49,)
    """

    def __init__(
        self,
        queries: dict[str, SingleInput],  # qid => query
        corpus: dict[str, SingleInput],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
        write_predictions: bool = False,
        bootstrap_resamples: int | None = None,
        bootstrap_confidence_level: float = 0.95,
        bootstrap_seed: int | None = 42,
        query_groups: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if bootstrap_resamples is not None and (
            isinstance(bootstrap_resamples, bool)
            or not isinstance(bootstrap_resamples, int)
            or bootstrap_resamples < 1
        ):
            raise ValueError(f"bootstrap_resamples must be None or a positive integer, got {bootstrap_resamples!r}.")
        if not 0 < bootstrap_confidence_level < 1:
            raise ValueError(
                f"bootstrap_confidence_level must be strictly between 0 and 1, got {bootstrap_confidence_level!r}."
            )
        if query_groups is not None:
            for qid, group in query_groups.items():
                if not isinstance(group, str):
                    raise TypeError(
                        f"query_groups must map query IDs to group labels of type str, got {group!r} for query {qid!r}."
                    )

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self._corpus_id_ranks = None

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self._score_functions_from_model = score_functions is None
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)
        self.write_predictions = write_predictions
        if self.write_predictions:
            self.predictions_file = "Information-Retrieval_evaluation" + name + "_predictions.jsonl"

        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_confidence_level = bootstrap_confidence_level
        self.bootstrap_seed = bootstrap_seed
        self.query_groups = query_groups

        # Filled by compute_all_metrics on every call, all keyed by score function name first:
        # per_query_metrics: {"cosine": {"ndcg@10": np.ndarray of shape (n_queries,), ...}}, aligned with queries_ids
        self.per_query_metrics: dict[str, dict[str, np.ndarray]] = {}
        # confidence_intervals: {"cosine": {"ndcg@10": (low, high), ...}}, empty unless bootstrap_resamples is set
        self.confidence_intervals: dict[str, dict[str, tuple[float, float]]] = {}
        # group_scores: {"cosine": {"easy": {"ndcg@k": {10: value}, ...}}}, empty unless query_groups is set
        self.group_scores: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
        # group_confidence_intervals: {"cosine": {"easy": {"ndcg@10": (low, high), ...}}}, needs both options
        self.group_confidence_intervals: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def _model_score_functions(self, model: SentenceTransformer) -> dict[str, Callable[[Tensor, Tensor], Tensor]]:
        return {model.similarity_fn_name: model.similarity}

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        headers_changed = False
        if self._score_functions_from_model:
            # Resolved per call, so an evaluator reused across models scores and labels each of them
            # with its own similarity rather than with the first model's.
            self.score_functions = self._model_score_functions(model)
            if self.score_function_names != sorted(self.score_functions):
                headers_changed = bool(self.score_function_names)
                self.score_function_names = sorted(self.score_functions)
                self.csv_headers = ["epoch", "steps"]
                self._append_csv_headers(self.score_function_names)
                self.primary_metric = None

        scores = self.compute_all_metrics(model, output_path=output_path, *args, **kwargs)

        # Write results to disk
        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                if headers_changed:
                    logger.warning(
                        f"{self.csv_file} was written with different score function columns, so the "
                        f"rows appended for {self.score_function_names} are labeled with the previous "
                        "score function."
                    )
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["ndcg@k"][max(self.ndcg_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        for score_function, intervals in self.confidence_intervals.items():
            for metric, (low, high) in intervals.items():
                metrics[f"{score_function}_{metric}_ci_low"] = low
                metrics[f"{score_function}_{metric}_ci_high"] = high
        for score_function, groups in self.group_scores.items():
            for group, values_dict in groups.items():
                group_intervals = self.group_confidence_intervals.get(score_function, {}).get(group, {})
                for metric_name, values in values_dict.items():
                    for k, value in values.items():
                        metric = f"{metric_name.replace('@k', '')}@{k}"
                        metrics[f"{score_function}_{metric}_{group}"] = value
                        if metric in group_intervals:
                            low, high = group_intervals[metric]
                            metrics[f"{score_function}_{metric}_{group}_ci_low"] = low
                            metrics[f"{score_function}_{metric}_{group}_ci_high"] = high
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def get_corpus_id_ranks(self) -> Tensor:
        """Position of each corpus entry in the ascending ``corpus_ids`` order, cached across calls."""
        if self._corpus_id_ranks is None:
            order = sorted(range(len(self.corpus_ids)), key=self.corpus_ids.__getitem__)
            self._corpus_id_ranks = torch.empty(len(self.corpus_ids), dtype=torch.long)
            self._corpus_id_ranks[torch.tensor(order, dtype=torch.long)] = torch.arange(len(self.corpus_ids))
        return self._corpus_id_ranks

    def compute_all_metrics(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
        output_path: str | None = None,
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        query_embeddings = self.embed_inputs(
            model,
            self.queries,
            encode_fn_name="query",
            prompt_name=self.query_prompt_name,
            prompt=self.query_prompt,
        )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = self.embed_inputs(
                    corpus_model,
                    self.corpus[corpus_start_idx:corpus_end_idx],
                    encode_fn_name="document",
                    prompt_name=self.corpus_prompt_name,
                    prompt=self.corpus_prompt,
                )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarities
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                top_k = min(max_k, len(pair_scores[0]))
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, top_k, dim=1, largest=True, sorted=False
                )
                # torch.topk breaks score ties arbitrarily. Rows with ties at the top_k-th score are
                # reselected under the (-score, corpus_id) total order used for the final ranking, so
                # the reported metrics do not depend on corpus_chunk_size.
                thresholds = pair_scores_top_k_values.min(dim=1, keepdim=True).values
                has_boundary_ties = ((pair_scores >= thresholds).sum(dim=1) > top_k).cpu().tolist()
                if any(has_boundary_ties):
                    # Order the candidates by (-score, corpus_id) with an integer key, so that only
                    # top_k of them per query reach Python even when the whole chunk is tied: every
                    # score above the cutoff sorts first, then the ties with the lowest corpus_ids.
                    n_corpus = len(self.corpus_ids)
                    chunk_ranks = self.get_corpus_id_ranks()[corpus_start_idx:corpus_end_idx].to(pair_scores.device)
                    tie_keys = torch.where(pair_scores > thresholds, chunk_ranks, 2 * n_corpus)
                    tie_keys = torch.where(pair_scores == thresholds, chunk_ranks + n_corpus, tie_keys)
                    tie_idx = tie_keys.topk(top_k, dim=1, largest=False).indices
                    tie_scores = pair_scores.gather(1, tie_idx).cpu().tolist()
                    tie_idx = tie_idx.cpu().tolist()
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    if has_boundary_ties[query_itr]:
                        sub_corpus_ids = tie_idx[query_itr]
                        scores = tie_scores[query_itr]
                    else:
                        sub_corpus_ids = pair_scores_top_k_idx[query_itr]
                        scores = pair_scores_top_k_values[query_itr]

                    query_hits = queries_result_list[name][query_itr]
                    for sub_corpus_id, score in zip(sub_corpus_ids, scores):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # NOTE: TREC/BEIR/MTEB skips cases where the corpus_id is the same as the query_id, e.g.:
                        # if corpus_id == self.queries_ids[query_itr]:
                        #     continue
                        # This is not done here, as this might be unexpected behaviour if the user just uses
                        # sets of integers from 0 as query_ids and corpus_ids.
                        query_hits.append((score, corpus_id))
                    query_hits.sort(key=lambda x: (-x[0], x[1]))
                    del query_hits[max_k:]

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        if self.write_predictions and output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            for name in queries_result_list:
                base_filename = self.predictions_file.replace(".jsonl", f"_{name}.jsonl")
                json_path = os.path.join(output_path, base_filename)
                mode = "w"  # Always create a new file for each score function

                with open(json_path, mode=mode, encoding="utf-8") as fOut:
                    for query_itr in range(len(queries_result_list[name])):
                        query_id = self.queries_ids[query_itr]
                        query_text = self.queries[query_itr]
                        results = queries_result_list[name][query_itr]

                        # Sort results by descending score, breaking ties by ascending corpus_id
                        results = sorted(results, key=lambda x: (-x["score"], x["corpus_id"]))

                        prediction = {
                            "query_id": query_id,
                            "query": query_text,
                            "results": results,
                        }

                        fOut.write(json.dumps(prediction) + "\n")

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Per-query values behind the metrics, kept for the confidence intervals, the query groups and for comparing
        # models on the same queries. compute_metrics stays the override point for the reported scores.
        per_query_metrics = {
            name: self.compute_per_query_metrics(queries_result_list[name]) for name in self.score_functions
        }
        self.per_query_metrics = {
            name: {
                f"{metric_name.replace('@k', '')}@{k}": np.asarray(values, dtype=float)
                for metric_name, values_at_k in per_query.items()
                for k, values in values_at_k.items()
            }
            for name, per_query in per_query_metrics.items()
        }

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        self.confidence_intervals = {}
        if self.bootstrap_resamples is not None:
            # One draw of resamples shared by every metric and score function, so that their intervals are comparable
            indices = bootstrap_indices(len(self.queries), self.bootstrap_resamples, self.bootstrap_seed)
            self.confidence_intervals = {
                name: self._bootstrap_confidence_intervals(metrics, indices)
                for name, metrics in self.per_query_metrics.items()
            }

        self.group_scores = {}
        self.group_confidence_intervals = {}
        group_query_indices = self._group_query_indices()
        for group, query_indices in group_query_indices.items():
            group_indices = None
            if self.bootstrap_resamples is not None:
                group_indices = bootstrap_indices(len(query_indices), self.bootstrap_resamples, self.bootstrap_seed)
            for name in self.score_functions:
                group_per_query = {
                    metric_name: {
                        k: [values[query_itr] for query_itr in query_indices] for k, values in values_at_k.items()
                    }
                    for metric_name, values_at_k in per_query_metrics[name].items()
                }
                self.group_scores.setdefault(name, {})[group] = self.aggregate_per_query_metrics(group_per_query)
                if group_indices is not None:
                    group_arrays = {
                        metric: values[query_indices] for metric, values in self.per_query_metrics[name].items()
                    }
                    self.group_confidence_intervals.setdefault(name, {})[group] = self._bootstrap_confidence_intervals(
                        group_arrays, group_indices
                    )

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name], self.confidence_intervals.get(name))
            for group, group_scores in self.group_scores.get(name, {}).items():
                logger.info(f"Group '{group}' ({len(group_query_indices[group])} queries):")
                self.output_scores(group_scores, self.group_confidence_intervals.get(name, {}).get(group))

        return scores

    def _bootstrap_confidence_intervals(
        self, per_query_metrics: dict[str, np.ndarray], indices: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Confidence interval of every metric from the same precomputed resample ``indices``."""
        return {
            metric: bootstrap_confidence_interval(
                values, confidence_level=self.bootstrap_confidence_level, indices=indices
            )
            for metric, values in per_query_metrics.items()
        }

    def _group_query_indices(self) -> dict[str, list[int]]:
        """Positions in ``queries_ids`` of the evaluated queries of every query group, in sorted group order."""
        if self.query_groups is None:
            return {}
        query_indices = {group: [] for group in sorted(set(self.query_groups.values()))}
        for query_itr, query_id in enumerate(self.queries_ids):
            group = self.query_groups.get(query_id)
            if group is not None:
                query_indices[group].append(query_itr)
        for group in list(query_indices):
            if not query_indices[group]:
                logger.warning(f"Query group '{group}' has no evaluated queries and is skipped.")
                del query_indices[group]
        return query_indices

    # Backwards compatibility alias
    compute_metrices = compute_all_metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: SingleInput | Sequence[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        if encode_fn_name is None:
            encode_fn = model.encode
        elif encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        return encode_fn(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    def compute_per_query_metrics(self, queries_result_list: list[object]) -> dict[str, dict[int, list[float]]]:
        """
        Computes every metric for each query separately, in the order of ``queries_result_list`` (i.e. ``queries_ids``).

        Args:
            queries_result_list (List[object]): Per query, the retrieved hits as ``{"corpus_id": ..., "score": ...}``
                dictionaries.

        Returns:
            Dict[str, Dict[int, List[float]]]: The nesting of :meth:`compute_metrics` with a list of per-query values
            in place of each aggregate: ``{"accuracy@k": {k: [...]}, "precision@k": ..., "recall@k": ...,
            "ndcg@k": ..., "mrr@k": ..., "map@k": ...}``. Accuracy values are 0/1 integers and MRR values are the
            reciprocal rank of the first relevant hit, or 0.0 when none is retrieved within k.
        """
        # Init score computation values
        num_hits_at_k = {k: [] for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: [] for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores in descending order, breaking ties by ascending corpus_id
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: (-x["score"], x["corpus_id"]))
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                hit_at_k = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        hit_at_k = 1
                        break
                num_hits_at_k[k_val].append(hit_at_k)

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                reciprocal_rank = 0.0
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        reciprocal_rank = 1.0 / (rank + 1)
                        break
                MRR[k_val].append(reciprocal_rank)

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def aggregate_per_query_metrics(
        self, per_query_metrics: dict[str, dict[int, list[float]]]
    ) -> dict[str, dict[int, float]]:
        """
        Averages the per-query values of :meth:`compute_per_query_metrics` over their queries, e.g. over all queries
        or over the queries of one group.

        Args:
            per_query_metrics (Dict[str, Dict[int, List[float]]]): Per-query values as returned by
                :meth:`compute_per_query_metrics`, possibly restricted to a subset of the queries.

        Returns:
            Dict[str, Dict[int, float]]: ``{"accuracy@k": {k: value}, "precision@k": ..., "recall@k": ...,
            "ndcg@k": ..., "mrr@k": ..., "map@k": ...}``, the format of :meth:`compute_metrics`.
        """
        aggregated = {}
        for metric_name, values_at_k in per_query_metrics.items():
            aggregated[metric_name] = {}
            for k, values in values_at_k.items():
                if metric_name == "accuracy@k":
                    # Integer count of the queries with a relevant hit, divided once
                    aggregated[metric_name][k] = sum(values) / len(values)
                elif metric_name == "mrr@k":
                    # Running sum of the reciprocal ranks in query order
                    total = 0
                    for value in values:
                        total += value
                    aggregated[metric_name][k] = total / len(values)
                else:
                    aggregated[metric_name][k] = np.mean(values)
        return aggregated

    def compute_metrics(self, queries_result_list: list[object]):
        return self.aggregate_per_query_metrics(self.compute_per_query_metrics(queries_result_list))

    def output_scores(self, scores, confidence_intervals: dict[str, tuple[float, float]] | None = None):
        for metric_name, label, value_format, scale in (
            ("accuracy@k", "Accuracy", "{:.2f}%", 100),
            ("precision@k", "Precision", "{:.2f}%", 100),
            ("recall@k", "Recall", "{:.2f}%", 100),
            ("mrr@k", "MRR", "{:.4f}", 1),
            ("ndcg@k", "NDCG", "{:.4f}", 1),
            ("map@k", "MAP", "{:.4f}", 1),
        ):
            for k, value in scores[metric_name].items():
                line = f"{label}@{k}: " + value_format.format(value * scale)
                if confidence_intervals:
                    low, high = confidence_intervals[f"{metric_name.replace('@k', '')}@{k}"]
                    line += (
                        f" ({self.bootstrap_confidence_level * 100:g}% CI: "
                        f"{value_format.format(low * scale)} – {value_format.format(high * scale)})"
                    )
                logger.info(line)

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg

    def get_config_dict(self):
        config_dict = {}
        config_dict_candidate_keys = [
            "truncate_dim",
            "query_prompt",
            "query_prompt_name",
            "corpus_prompt",
            "corpus_prompt_name",
        ]
        for key in config_dict_candidate_keys:
            if getattr(self, key) is not None:
                config_dict[key] = getattr(self, key)
        if self.bootstrap_resamples is not None:
            config_dict["bootstrap_resamples"] = self.bootstrap_resamples
            config_dict["bootstrap_confidence_level"] = self.bootstrap_confidence_level
        if self.query_groups is not None:
            config_dict["num_query_groups"] = len(set(self.query_groups.values()))
        return config_dict
