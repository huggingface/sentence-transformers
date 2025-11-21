from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm import tqdm

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

logger = logging.getLogger(__name__)


class ReciprocalRankFusionEvaluator(SentenceEvaluator):
    """
    This class evaluates a hybrid search approach using Reciprocal Rank Fusion (RRF).

    Given a query and two separate ranked lists of documents from different retrievers (e.g., sparse and dense),
    it combines them using the RRF formula and computes metrics like MRR@k, NDCG@k, and MAP.

    Args:
        dense_samples (list): A list of dictionaries for dense retriever results. Each dictionary should have:
            - 'query_id': The ID of the query
            - 'query': The search query text
            - 'positive': A list of relevant documents
            - 'documents': A list of all documents (including positives)
        sparse_samples (list): A list of dictionaries for sparse retriever results with the same format
        at_k (int): Only consider the top k documents for evaluation. Defaults to 10.
        rrf_k (int): Constant in the RRF formula. Defaults to 60.
        name (str): Name of the evaluator. Defaults to "".
        batch_size (int): Batch size used for the evaluation. Defaults to 32.
        show_progress_bar (bool): Output a progress bar. Defaults to False.
        write_csv (bool): Write results to CSV file. Defaults to True.
        write_predictions (bool): Whether to write the fused predictions to a JSONL file. Defaults to False.

    Example:
        See an example usage `Applications > Retrieve & Rerank <../../../examples/sparse_encoder/applications/retrieve_rerank/README.html>`_

    """

    def __init__(
        self,
        dense_samples: list[dict[str, str | list[str]]],
        sparse_samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        rrf_k: int = 60,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        write_predictions: bool = False,
    ):
        super().__init__()
        self.dense_samples = dense_samples
        self.sparse_samples = sparse_samples

        # Validate that both sample lists have the same length
        if len(dense_samples) != len(sparse_samples):
            raise ValueError(
                f"Dense samples ({len(dense_samples)}) and sparse samples ({len(sparse_samples)}) must have the same length"
            )

        # Validate that both lists have query_id field
        for i, (dense_sample, sparse_sample) in enumerate(zip(dense_samples, sparse_samples)):
            if "query_id" not in dense_sample or "query_id" not in sparse_sample:
                raise ValueError(f"Sample at index {i} missing 'query_id' field")

            if dense_sample["query_id"] != sparse_sample["query_id"]:
                raise ValueError(
                    f"Query ID mismatch at index {i}: {dense_sample['query_id']} != {sparse_sample['query_id']}"
                )

        self.at_k = at_k
        self.rrf_k = rrf_k
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_predictions = write_predictions

        self.csv_file = "ReciprocalRankFusion_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "Dense_MAP",
            f"Dense_MRR@{self.at_k}",
            f"Dense_NDCG@{self.at_k}",
            "Sparse_MAP",
            f"Sparse_MRR@{self.at_k}",
            f"Sparse_NDCG@{self.at_k}",
            "Fusion_MAP",
            f"Fusion_MRR@{self.at_k}",
            f"Fusion_NDCG@{self.at_k}",
        ]
        self.write_csv = write_csv
        self.primary_metric = f"ndcg@{self.at_k}"

        if self.write_predictions:
            self.predictions_file = (
                "ReciprocalRankFusion_evaluation" + ("_" + name if name else "") + "_predictions.jsonl"
            )

    def __call__(self, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"ReciprocalRankFusionEvaluator: Evaluating hybrid search on the {self.name} dataset{out_txt}:")
        logger.info(f"Processing {len(self.dense_samples)} samples")

        # Initialize scores
        dense_mrr_scores = []
        dense_ndcg_scores = []
        dense_ap_scores = []

        sparse_mrr_scores = []
        sparse_ndcg_scores = []
        sparse_ap_scores = []

        fusion_mrr_scores = []
        fusion_ndcg_scores = []
        fusion_ap_scores = []

        num_queries = 0
        num_positives = []
        fused_results_list = []

        # Process each pair of samples
        for i, (dense_sample, sparse_sample) in enumerate(
            tqdm(
                zip(self.dense_samples, self.sparse_samples),
                desc="Evaluating",
                disable=not self.show_progress_bar,
                total=len(self.dense_samples),
            )
        ):
            query_id = dense_sample["query_id"]

            # Verify query_id match (redundant since we checked in __init__, but good for safety)
            assert query_id == sparse_sample["query_id"], (
                f"Query ID mismatch: {query_id} != {sparse_sample['query_id']}"
            )

            query = dense_sample["query"]
            positive = dense_sample["positive"]
            if isinstance(positive, str):
                positive = [positive]

            # Get documents from both retrievers
            dense_docs = dense_sample["documents"]
            sparse_docs = sparse_sample["documents"]

            # Calculate base metrics for dense retriever
            dense_is_relevant = [int(sample in positive) for sample in dense_docs]

            # Skip if no relevant documents
            if sum(dense_is_relevant) == 0:
                dense_mrr, dense_ndcg, dense_ap = 0, 0, 0
            else:
                dense_is_relevant += [1] * (len(positive) - sum(dense_is_relevant))
                dense_pred_scores = np.array(range(len(dense_is_relevant), 0, -1))
                dense_mrr, dense_ndcg, dense_ap = self.compute_metrics(dense_is_relevant, dense_pred_scores)

            dense_mrr_scores.append(dense_mrr)
            dense_ndcg_scores.append(dense_ndcg)
            dense_ap_scores.append(dense_ap)

            # Calculate base metrics for sparse retriever
            sparse_is_relevant = [int(sample in positive) for sample in sparse_docs]

            # Skip if no relevant documents
            if sum(sparse_is_relevant) == 0:
                sparse_mrr, sparse_ndcg, sparse_ap = 0, 0, 0
            else:
                sparse_is_relevant += [1] * (len(positive) - sum(sparse_is_relevant))
                sparse_pred_scores = np.array(range(len(sparse_is_relevant), 0, -1))
                sparse_mrr, sparse_ndcg, sparse_ap = self.compute_metrics(sparse_is_relevant, sparse_pred_scores)

            sparse_mrr_scores.append(sparse_mrr)
            sparse_ndcg_scores.append(sparse_ndcg)
            sparse_ap_scores.append(sparse_ap)

            # Create rank maps for each retriever
            dense_ranks = {doc: rank for rank, doc in enumerate(dense_docs)}
            sparse_ranks = {doc: rank for rank, doc in enumerate(sparse_docs)}

            # Combine all unique documents
            all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())

            # Calculate RRF scores
            rrf_scores = {}
            for doc in all_docs:
                dense_rank = dense_ranks.get(doc, len(dense_docs))
                sparse_rank = sparse_ranks.get(doc, len(sparse_docs))
                rrf_scores[doc] = (1 / (self.rrf_k + dense_rank)) + (1 / (self.rrf_k + sparse_rank))

            # Sort documents by RRF scores in descending order
            fused_docs = sorted(rrf_scores.keys(), key=lambda doc: rrf_scores[doc], reverse=True)

            # Create binary relevance list for evaluation
            fusion_is_relevant = [int(sample in positive) for sample in fused_docs]

            num_queries += 1
            num_positives.append(len(positive))

            # Skip if no relevant documents in fusion results
            if sum(fusion_is_relevant) == 0:
                fusion_mrr, fusion_ndcg, fusion_ap = 0, 0, 0
            else:
                fusion_is_relevant += [1] * (len(positive) - sum(fusion_is_relevant))
                fusion_pred_scores = np.array(range(len(fusion_is_relevant), 0, -1))
                fusion_mrr, fusion_ndcg, fusion_ap = self.compute_metrics(fusion_is_relevant, fusion_pred_scores)

            fusion_mrr_scores.append(fusion_mrr)
            fusion_ndcg_scores.append(fusion_ndcg)
            fusion_ap_scores.append(fusion_ap)

            # Store fused results for prediction file if requested
            if self.write_predictions:
                fused_results_list.append(
                    {"query_id": query_id, "query": query, "positive": positive, "documents": fused_docs}
                )

        # Calculate mean scores
        mean_dense_mrr = np.mean(dense_mrr_scores)
        mean_dense_ndcg = np.mean(dense_ndcg_scores)
        mean_dense_ap = np.mean(dense_ap_scores)

        mean_sparse_mrr = np.mean(sparse_mrr_scores)
        mean_sparse_ndcg = np.mean(sparse_ndcg_scores)
        mean_sparse_ap = np.mean(sparse_ap_scores)

        mean_fusion_mrr = np.mean(fusion_mrr_scores)
        mean_fusion_ndcg = np.mean(fusion_ndcg_scores)
        mean_fusion_ap = np.mean(fusion_ap_scores)
        # Store metrics
        metrics = {
            "dense_map": mean_dense_ap,
            f"dense_mrr@{self.at_k}": mean_dense_mrr,
            f"dense_ndcg@{self.at_k}": mean_dense_ndcg,
            "sparse_map": mean_sparse_ap,
            f"sparse_mrr@{self.at_k}": mean_sparse_mrr,
            f"sparse_ndcg@{self.at_k}": mean_sparse_ndcg,
            "map": mean_fusion_ap,
            f"mrr@{self.at_k}": mean_fusion_mrr,
            f"ndcg@{self.at_k}": mean_fusion_ndcg,
        }

        # Log results
        logger.info(
            f"Queries: {num_queries}\t"
            f"Positives: Min {min(num_positives) if num_positives else 0:.1f}, "
            f"Mean {np.mean(num_positives) if num_positives else 0:.1f}, "
            f"Max {max(num_positives) if num_positives else 0:.1f}"
        )

        # Display metrics with comparison
        logger.info("=" * 75)
        logger.info(
            f"{'Metric':<7} | {'Dense':^8} | {'Sparse':^8} | {'Fusion':^8} | {'Gain vs Dense':^13} | {'Gain vs Sparse':^14} |"
        )
        logger.info("-" * 75)
        logger.info(
            f"MAP     | {mean_dense_ap:>8.2%} | {mean_sparse_ap:>8.2%} | {mean_fusion_ap:>8.2%} | {mean_fusion_ap - mean_dense_ap:>+13.2%} | {mean_fusion_ap - mean_sparse_ap:>+14.2%} |"
        )
        logger.info(
            f"MRR@{self.at_k:<3} | {mean_dense_mrr:>8.2%} | {mean_sparse_mrr:>8.2%} | {mean_fusion_mrr:>8.2%} | {mean_fusion_mrr - mean_dense_mrr:>+13.2%} | {mean_fusion_mrr - mean_sparse_mrr:>+14.2%} |"
        )
        logger.info(
            f"NDCG@{self.at_k:<2} | {mean_dense_ndcg:>8.2%} | {mean_sparse_ndcg:>8.2%} | {mean_fusion_ndcg:>8.2%} | {mean_fusion_ndcg - mean_dense_ndcg:>+13.2%} | {mean_fusion_ndcg - mean_sparse_ndcg:>+14.2%} |"
        )
        logger.info("=" * 75)

        # Write results to CSV if requested
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if self.write_csv:
                csv_path = os.path.join(output_path, self.csv_file)
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow(
                        [
                            epoch,
                            steps,
                            mean_dense_ap,
                            mean_dense_mrr,
                            mean_dense_ndcg,
                            mean_sparse_ap,
                            mean_sparse_mrr,
                            mean_sparse_ndcg,
                            mean_fusion_ap,
                            mean_fusion_mrr,
                            mean_fusion_ndcg,
                        ]
                    )

            # Write prediction results if requested
            if self.write_predictions and fused_results_list:
                json_path = os.path.join(output_path, self.predictions_file)
                with open(json_path, mode="w", encoding="utf-8") as f:
                    for result in fused_results_list:
                        f.write(json.dumps(result) + "\n")
                logger.info(f"Wrote fused ranking predictions to {json_path}")

        # Prefix metrics with name if provided
        metrics = self.prefix_name_to_metrics(metrics, self.name)

        return metrics

    def compute_metrics(self, y_true, y_pred):
        """Compute MRR, NDCG, and AP metrics using sklearn"""
        ranking = np.argsort(y_pred)[::-1]

        # Calculate MRR@k
        mrr = 0
        for rank, index in enumerate(ranking[0 : self.at_k]):
            if y_true[index]:
                mrr = 1 / (rank + 1)
                break

        # Calculate NDCG@k
        ndcg = ndcg_score([y_true], [y_pred], k=self.at_k)

        # Calculate MAP
        ap = average_precision_score(y_true, y_pred)

        return mrr, ndcg, ap

    def prefix_name_to_metrics(self, metrics: dict[str, float], prefix: str) -> dict[str, float]:
        """Prefix all metric names with the evaluator name."""
        if not prefix:
            return metrics
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def get_config_dict(self):
        return {
            "at_k": self.at_k,
            "rrf_k": self.rrf_k,
            "write_predictions": self.write_predictions,
        }
