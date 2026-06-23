from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sentence_transformers.sentence_transformer.evaluation.triplet import TripletEvaluator
from sentence_transformers.util.similarity import maxsim_pairwise

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorTripletEvaluator(TripletEvaluator):
    """Triplet evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Given ``(anchor, positive, negative)`` triplets, checks how often
    ``MaxSim(anchor, positive) > MaxSim(anchor, negative) + margin``. The anchors are encoded via
    :meth:`encode_query` (with the query prefix and length); positives and negatives are encoded via
    :meth:`encode_document`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.margin = {"maxsim": self.margin.get("maxsim", self.margin["cosine"])}

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
        logger.info(f"MultiVectorTripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        embeddings_anchors = self._embed(model, self.anchors, is_query=True)
        embeddings_positives = self._embed(model, self.positives, is_query=False)
        embeddings_negatives = self._embed(model, self.negatives, is_query=False)

        if not self.similarity_fn_names:
            self.similarity_fn_names = ["maxsim"]
            self._append_csv_headers(self.similarity_fn_names)

        margin = self.margin.get("maxsim", 0)
        positive_scores = maxsim_pairwise(embeddings_anchors, embeddings_positives)
        negative_scores = maxsim_pairwise(embeddings_anchors, embeddings_negatives)
        accuracy = (positive_scores > negative_scores + margin).float().mean().item()

        metrics = {"maxsim_accuracy": accuracy}
        logger.info(f"Accuracy MaxSim:\t{accuracy:.2%}")

        self.primary_metric = "maxsim_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def _embed(
        self,
        model: MultiVectorEncoder,
        sentences: list[str] | np.ndarray,
        is_query: bool,
    ) -> list:
        encode_fn = model.encode_query if is_query else model.encode_document
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
