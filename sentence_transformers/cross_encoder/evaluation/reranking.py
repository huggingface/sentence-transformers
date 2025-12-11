from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm import tqdm

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderRerankingEvaluator(SentenceEvaluator):
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, ``MRR@10``, ``NDCG@10`` and ``MAP`` are computed to measure the quality of the ranking.

    The evaluator expects a list of samples. Each sample is a dictionary with the mandatory "query" and "positive" keys,
    and either a "negative" or a "documents" key. The "query" is the search query, the "positive" is a list of relevant
    documents, and the "negative" is a list of irrelevant documents. Alternatively, the "documents" key can be used to
    provide a list of all documents, including the positive ones. In this case, the evaluator will assume that the list
    is already ranked by similarity, with the most similar documents first, and will report both the reranking performance
    as well as the performance before reranking. This can be useful to measure the improvement of the reranking on
    top of a first-stage retrieval (e.g. a SentenceTransformer model).

    Note that the maximum score is 1.0 by default, because all positive documents are included in the ranking. This
    can be toggled off by using samples with ``documents`` instead of ``negative``, i.e. ranked lists of all documents
    including the positive ones, together with ``always_rerank_positives=False``. ``always_rerank_positives=False`` only
    works when using ``documents`` instead of ``negative``.

    Args:
        samples (list): A list of dictionaries, where each dictionary represents a sample and has the following keys:
            - 'query' (mandatory): The search query.
            - 'positive' (mandatory): A list of positive (relevant) documents.
            - 'negative' (optional): A list of negative (irrelevant) documents. Mutually exclusive with 'documents'.
            - 'documents' (optional): A list of all documents, including the positive ones. This list is assumed to be
                ranked by similarity, with the most similar documents first. Mutually exclusive with 'negative'.
        at_k (int, optional): Only consider the top k most similar documents to each query for the evaluation. Defaults to 10.
        always_rerank_positives (bool): If True, always evaluate with all positives included. If False, only include
            the positives that are already in the documents list. Always set to True if your ``samples`` contain ``negative``
            instead of ``documents``. When using ``documents``, setting this to True will result in a more useful evaluation
            signal, but setting it to False will result in a more realistic evaluation. Defaults to True.
        name (str, optional): Name of the evaluator, used for logging, saving in a CSV, and the model card. Defaults to "".
        batch_size (int): Batch size to compute sentence embeddings. Defaults to 64.
        show_progress_bar (bool): Show progress bar when computing embeddings. Defaults to False.
        write_csv (bool): Write results to CSV file. Defaults to True.
        mrr_at_k (Optional[int], optional): Deprecated parameter. Please use `at_k` instead. Defaults to None.

    Example:
        ::

            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
            from datasets import load_dataset

            # Load a model
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

            # Load a dataset with queries, positives, and negatives
            eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

            samples = [
                {
                    "query": sample["query"],
                    "positive": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if is_selected],
                    "documents": sample["passages"]["passage_text"],
                    # or
                    # "negative": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if not is_selected],
                }
                for sample in eval_dataset
            ]

            # Initialize the evaluator
            reranking_evaluator = CrossEncoderRerankingEvaluator(
                samples=samples,
                name="ms-marco-dev",
                show_progress_bar=True,
            )
            results = reranking_evaluator(model)
            '''
            CrossEncoderRerankingEvaluator: Evaluating the model on the ms-marco-dev dataset:
            Queries: 10047    Positives: Min 0.0, Mean 1.1, Max 5.0   Negatives: Min 1.0, Mean 7.1, Max 10.0
                     Base  -> Reranked
            MAP:     34.03 -> 62.36
            MRR@10:  34.67 -> 62.96
            NDCG@10: 49.05 -> 71.05
            '''
            print(reranking_evaluator.primary_metric)
            # => ms-marco-dev_ndcg@10
            print(results[reranking_evaluator.primary_metric])
            # => 0.7104656857184184
    """

    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        always_rerank_positives: bool = True,  # TODO: This is also confusing, perhaps setting=""
        name: str = "",
        batch_size: int = 64,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        mrr_at_k: int | None = None,
    ):
        super().__init__()
        self.samples = samples
        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k
        self.always_rerank_positives = always_rerank_positives

        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CrossEncoderRerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = ["epoch", "steps", "MAP", f"MRR@{self.at_k}", f"NDCG@{self.at_k}"]
        self.write_csv = write_csv
        self.primary_metric = f"ndcg@{self.at_k}"

    def __call__(
        self, model: CrossEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"CrossEncoderRerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        base_mrr_scores = []
        base_ndcg_scores = []
        base_ap_scores = []
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in tqdm(self.samples, desc="Evaluating samples", disable=not self.show_progress_bar, leave=False):
            if "query" not in instance:
                raise ValueError("CrossEncoderRerankingEvaluator requires a 'query' key in each sample.")
            if "positive" not in instance:
                raise ValueError("CrossEncoderRerankingEvaluator requires a 'positive' key in each sample.")
            if ("negative" in instance and "documents" in instance) or (
                "negative" not in instance and "documents" not in instance
            ):
                raise ValueError(
                    "CrossEncoderRerankingEvaluator requires exactly one of 'negative' and 'documents' in each sample."
                )

            query = instance["query"]
            positive = instance["positive"]
            if isinstance(positive, str):
                positive = [positive]

            negative = instance.get("negative", None)
            documents = instance.get("documents", None)

            if documents:
                base_is_relevant = [int(sample in positive) for sample in documents]
                if sum(base_is_relevant) == 0:
                    base_mrr, base_ndcg, base_ap = 0, 0, 0
                else:
                    # If not all positives are in documents, we need to add them at the end
                    base_is_relevant += [1] * (len(positive) - sum(base_is_relevant))
                    base_pred_scores = np.array(range(len(base_is_relevant), 0, -1))
                    base_mrr, base_ndcg, base_ap = self.compute_metrics(base_is_relevant, base_pred_scores)
                base_mrr_scores.append(base_mrr)
                base_ndcg_scores.append(base_ndcg)
                base_ap_scores.append(base_ap)

                if self.always_rerank_positives:
                    docs = positive + [doc for doc in documents if doc not in positive]
                    is_relevant = [1] * len(positive) + [0] * (len(docs) - len(positive))
                else:
                    docs = documents
                    is_relevant = [int(sample in positive) for sample in documents]
            else:
                docs = positive + negative
                is_relevant = [1] * len(positive) + [0] * len(negative)

            num_queries += 1

            num_positives.append(len(positive))
            num_negatives.append(len(is_relevant) - sum(is_relevant))

            if sum(is_relevant) == 0:
                all_mrr_scores.append(0)
                all_ndcg_scores.append(0)
                all_ap_scores.append(0)
                continue

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(
                model_input, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
            )

            # Add the ignored positives at the end
            if num_ignored_positives := len(is_relevant) - len(pred_scores):
                pred_scores = np.concatenate([pred_scores, np.zeros(num_ignored_positives)])

            mrr, ndcg, ap = self.compute_metrics(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ndcg_scores.append(ndcg)
            all_ap_scores.append(ap)

        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        mean_ap = np.mean(all_ap_scores)
        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }

        logger.info(
            f"Queries: {num_queries}\t"
            f"Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f}\t"
            f"Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        if documents:
            mean_base_mrr = np.mean(base_mrr_scores)
            mean_base_ndcg = np.mean(base_ndcg_scores)
            mean_base_ap = np.mean(base_ap_scores)
            base_metrics = {
                "base_map": mean_base_ap,
                f"base_mrr@{self.at_k}": mean_base_mrr,
                f"base_ndcg@{self.at_k}": mean_base_ndcg,
            }
            logger.info(f"{' ' * len(str(self.at_k))}       Base  -> Reranked")
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_base_ap * 100:.2f} -> {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_base_mrr * 100:.2f} -> {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_base_ndcg * 100:.2f} -> {mean_ndcg * 100:.2f}")

            model_card_metrics = {
                "map": f"{mean_ap:.4f} ({mean_ap - mean_base_ap:+.4f})",
                f"mrr@{self.at_k}": f"{mean_mrr:.4f} ({mean_mrr - mean_base_mrr:+.4f})",
                f"ndcg@{self.at_k}": f"{mean_ndcg:.4f} ({mean_ndcg - mean_base_ndcg:+.4f})",
            }
            model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
            self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

            metrics.update(base_metrics)
            metrics = self.prefix_name_to_metrics(metrics, self.name)
        else:
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

            metrics = self.prefix_name_to_metrics(metrics, self.name)
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

        return metrics

    def compute_metrics(self, y_true, y_pred):
        ranking = np.argsort(y_pred)[::-1]

        mrr = 0
        for rank, index in enumerate(ranking[0 : self.at_k]):
            if y_true[index]:
                mrr = 1 / (rank + 1)
                break

        ndcg = ndcg_score([y_true], [y_pred], k=self.at_k)
        ap = average_precision_score(y_true, y_pred)
        return mrr, ndcg, ap

    def get_config_dict(self):
        config_dict = {
            "at_k": self.at_k,
        }
        if self.samples and "documents" in self.samples[0]:
            config_dict["always_rerank_positives"] = self.always_rerank_positives
        return config_dict
