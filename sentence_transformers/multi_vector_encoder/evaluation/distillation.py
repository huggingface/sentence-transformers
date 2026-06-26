from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import torch
from scipy.stats import spearmanr

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.util.similarity import maxsim_pairwise

if TYPE_CHECKING:
    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorDistillationEvaluator(BaseEvaluator):
    """Distillation evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Given queries with one teacher-scored document each, computes the model's MaxSim score and reports:

    - KL divergence between teacher and student score distributions (lower is better).
    - Spearman rank correlation between teacher and student scores (higher is better, the primary metric).

    The Spearman score is generally a more interpretable mid-training signal than raw KL.

    Args:
        queries: List of query texts.
        documents: List of document texts. Must have the same length as ``queries``.
        scores: 1-D list / tensor of teacher scores, one per ``(query, document)`` pair.
        name: Optional run name appended to CSV filenames.
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to show a progress bar.
        write_csv: Whether to write per-call results to a CSV file under ``output_path``.
    """

    def __init__(
        self,
        queries: list[SingleInput],
        documents: list[SingleInput],
        scores: list[float] | torch.Tensor,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ) -> None:
        super().__init__()
        if not (len(queries) == len(documents) == len(scores)):
            raise ValueError(
                f"queries ({len(queries)}), documents ({len(documents)}), and scores ({len(scores)}) "
                f"must all have the same length."
            )
        self.queries = list(queries)
        self.documents = list(documents)
        self.scores = torch.as_tensor(scores, dtype=torch.float32)
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        self.csv_file = "multi_vector_distillation_evaluation" + (f"_{name}" if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "kl_divergence", "spearman"]

    def __call__(
        self,
        model: MultiVectorEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        logger.info(f"MultiVectorDistillationEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        query_embeddings = model.encode_query(
            self.queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
        doc_embeddings = model.encode_document(
            self.documents,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        student_scores = maxsim_pairwise(query_embeddings, doc_embeddings).cpu()

        # TODO: PyLate computes the KL per query over that query's candidate set. Here there is one document
        # per query, so this softmaxes over the whole dataset (a single global distribution), and the KL value
        # is not comparable to PyLate's; `spearman` is the primary metric. Revisit if multi-document-per-query
        # KD evaluation is needed.
        teacher_log_probs = torch.log_softmax(self.scores, dim=-1)
        student_log_probs = torch.log_softmax(student_scores, dim=-1)
        kl = torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
        ).item()
        spearman = spearmanr(self.scores.numpy(), student_scores.numpy()).statistic
        if spearman != spearman:  # NaN if all scores are constant
            spearman = 0.0

        metrics = {"kl_divergence": kl, "spearman": float(spearman)}
        logger.info(f"KL divergence:\t{kl:.4f}")
        logger.info(f"Spearman correlation:\t{spearman:.4f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, kl, spearman])

        self.primary_metric = "spearman"
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
