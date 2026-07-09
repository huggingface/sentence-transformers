from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from sentence_transformers.sentence_transformer.evaluation.triplet import TripletEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorTripletEvaluator(TripletEvaluator):
    """Triplet evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Given ``(anchor, positive, negative)`` triplets, checks how often
    ``MaxSim(anchor, positive) > MaxSim(anchor, negative) + margin``. The anchors are encoded via
    :meth:`encode_query` (with the query prefix and length); positives and negatives are encoded via
    :meth:`encode_document`. ``truncate_dim`` is not supported (multi-vector token embeddings have no
    Matryoshka-style truncation) and raises a ValueError.
    """

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("truncate_dim") is not None:
            raise ValueError(
                "truncate_dim is not supported by MultiVectorEncoder evaluators: multi-vector token "
                "embeddings have no Matryoshka-style truncation. Remove truncate_dim to evaluate at "
                "the full dimension."
            )
        super().__init__(*args, **kwargs)

    def _get_similarity_functions(self) -> dict:
        # All supported multi-vector similarities. The parent picks via similarity_fn_names (default: the model's).
        from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

        functions = {}
        for name in MultiVectorEncoder._SUPPORTED_SIMILARITY_FN_NAMES:
            pairwise = SimilarityFunction.to_similarity_pairwise_fn(name)
            functions[name] = lambda a, p, n, pairwise=pairwise: (pairwise(a, p), pairwise(a, n))
        return functions

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
