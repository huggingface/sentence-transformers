from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers.multi_vector_encoder.evaluation.information_retrieval import (
    MultiVectorInformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.evaluation.nano_beir import NanoBEIREvaluator

if TYPE_CHECKING:
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorNanoBEIREvaluator(NanoBEIREvaluator):
    """NanoBEIR evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Same as the dense :class:`~sentence_transformers.evaluation.NanoBEIREvaluator` but uses
    :class:`MultiVectorInformationRetrievalEvaluator` under the hood for each Nano-* subset, which switches
    encoding to :meth:`MultiVectorEncoder.encode_query` / :meth:`MultiVectorEncoder.encode_document` and
    scoring to MaxSim.
    """

    information_retrieval_class = MultiVectorInformationRetrievalEvaluator

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
