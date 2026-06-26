from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from sentence_transformers.sentence_transformer.evaluation.triplet import TripletEvaluator
from sentence_transformers.util.similarity import maxsim_pairwise

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorTripletEvaluator(TripletEvaluator):
    """Triplet evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Given ``(anchor, positive, negative)`` triplets, checks how often
    ``MaxSim(anchor, positive) > MaxSim(anchor, negative) + margin``. The anchors are encoded via
    :meth:`encode_query` (with the query prefix and length); positives and negatives are encoded via
    :meth:`encode_document`.
    """

    def _get_similarity_functions(self) -> dict:
        return {
            "maxsim": lambda a, p, n: (maxsim_pairwise(a, p), maxsim_pairwise(a, n)),
        }

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | list[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        **kwargs,
    ) -> list[Tensor]:
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            **kwargs,
        )
