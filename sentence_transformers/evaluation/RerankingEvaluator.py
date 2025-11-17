from __future__ import annotations

import csv
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score, ndcg_score

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import cos_sim

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.SentenceTransformer import SentenceTransformer


logger = logging.getLogger(__name__)


class RerankingEvaluator(SentenceEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10, NDCG@10 and MAP is compute to measure the quality of the ranking.

    Args:
        samples (list): A list of dictionaries, where each dictionary represents a sample and has the following keys:

            - 'query': The search query.
            - 'positive': A list of positive (relevant) documents.
            - 'negative': A list of negative (irrelevant) documents.
        at_k (int, optional): Only consider the top k most similar documents to each query for the evaluation. Defaults to 10.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        similarity_fct (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Similarity function between sentence embeddings. By default, cosine similarity. Defaults to cos_sim.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 64.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        use_batched_encoding (bool, optional): Whether or not to encode queries and documents in batches for greater speed, or 1-by-1 to save memory. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation dimension. Defaults to None.
        mrr_at_k (Optional[int], optional): Deprecated parameter. Please use `at_k` instead. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import RerankingEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Load a dataset with queries, positives, and negatives
            eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

            samples = [
                {
                    "query": sample["query"],
                    "positive": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if is_selected],
                    "negative": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if not is_selected],
                }
                for sample in eval_dataset
            ]

            # Initialize the evaluator
            reranking_evaluator = RerankingEvaluator(
                samples=samples,
                name="ms-marco-dev",
            )
            results = reranking_evaluator(model)
            '''
            RerankingEvaluator: Evaluating the model on the ms-marco-dev dataset:
            Queries: 9706      Positives: Min 1.0, Mean 1.1, Max 5.0   Negatives: Min 1.0, Mean 7.1, Max 9.0
            MAP: 56.07
            MRR@10: 56.70
            NDCG@10: 67.08
            '''
            print(reranking_evaluator.primary_metric)
            # => ms-marco-dev_ndcg@10
            print(results[reranking_evaluator.primary_metric])
            # => 0.6708042171399308
    """

    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        name: str = "",
        write_csv: bool = True,
        similarity_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
        truncate_dim: int | None = None,
        mrr_at_k: int | None = None,
    ):
        super().__init__()
        self.samples = samples
        self.name = name

        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k

        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.use_batched_encoding = use_batched_encoding
        self.truncate_dim = truncate_dim

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [
            sample for sample in self.samples if len(sample["positive"]) > 0 and len(sample["negative"]) > 0
        ]

        self.csv_file = "RerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "MAP",
            f"MRR@{self.at_k}",
            f"NDCG@{self.at_k}",
        ]
        self.write_csv = write_csv
        self.primary_metric = f"ndcg@{self.at_k}"

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """
        Evaluates the model on the dataset and returns the evaluation metrics.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to evaluate.
            output_path (str, optional): The output path to write the results. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to -1.
            steps (int, optional): The current step number. Defaults to -1.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"RerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model)
        mean_ap = scores["map"]
        mean_mrr = scores["mrr"]
        mean_ndcg = scores["ndcg"]

        #### Some stats about the dataset
        num_positives = [len(sample["positive"]) for sample in self.samples]
        num_negatives = [len(sample["negative"]) for sample in self.samples]

        logger.info(
            f"Queries: {len(self.samples)} \t Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f} \t Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        logger.info(f"MAP: {mean_ap * 100:.2f}")
        logger.info(f"MRR@{self.at_k}: {mean_mrr * 100:.2f}")
        logger.info(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

        #### Write results to disk
        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def compute_metrices(self, model: SentenceTransformer):
        """
        Computes the evaluation metrics for the given model.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        return (
            self.compute_metrices_batched(model)
            if self.use_batched_encoding
            else self.compute_metrices_individual(model)
        )

    def compute_metrices_batched(self, model: SentenceTransformer):
        """
        Computes the evaluation metrics in a batched way, by batching all queries and all documents together.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []

        all_query_embs = self.embed_inputs(
            model,
            [sample["query"] for sample in self.samples],
            encode_fn_name="query",
            show_progress_bar=self.show_progress_bar,
        )

        all_docs = []

        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])

        all_docs_embs = self.embed_inputs(
            model, all_docs, encode_fn_name="document", show_progress_bar=self.show_progress_bar
        )

        # Compute scores
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
            query_emb = all_query_embs[query_idx]
            query_idx += 1

            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
            pred_scores = pred_scores.cpu().tolist()

            # Compute MRR score
            is_relevant = [1] * num_pos + [0] * num_neg
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute NDCG score
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}

    def compute_metrices_individual(self, model: SentenceTransformer):
        """
        Computes the evaluation metrics individually by embedding every (query, positive, negative) tuple individually.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []

        for instance in tqdm.tqdm(self.samples, disable=not self.show_progress_bar, desc="Samples"):
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [1] * len(positive) + [0] * len(negative)

            query_emb = self.embed_inputs(model, [query], encode_fn_name="query", show_progress_bar=False)
            docs_emb = self.embed_inputs(model, docs, encode_fn_name="document", show_progress_bar=False)

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
            pred_scores = pred_scores.cpu().tolist()

            # Compute MRR score
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute NDCG score
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: str | list[str] | np.ndarray,
        encode_fn_name: str | None = None,
        show_progress_bar: bool | None = None,
        **kwargs,
    ) -> Tensor:
        if encode_fn_name is None:
            encode_fn = model.encode
        elif encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    def get_config_dict(self):
        config_dict = {"at_k": self.at_k}
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
