from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor

from sentence_transformers.multi_vector_encoder.evaluation.utils import pooling_config
from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling
from sentence_transformers.sentence_transformer.evaluation.triplet import TripletEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorTripletEvaluator(TripletEvaluator):
    """Triplet evaluator for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models.

    Given ``(anchor, positive, negative)`` triplets, checks how often
    ``MaxSim(anchor, positive) > MaxSim(anchor, negative) + margin``. The anchors are encoded via
    :meth:`encode_query` (with the query prefix and length). Positives and negatives are encoded via
    :meth:`encode_document`. ``truncate_dim`` truncates and L2-normalizes token embeddings, while
    ``token_pooling`` applies pooling according to the pooler's ``tasks``. When None, both use the
    model's encoding defaults.

    A ``margin`` dictionary must be keyed by a supported similarity, ``maxsim`` or ``meanmaxsim`` (a
    float applies to both). The two need different margins: MaxSim sums a per-token maximum over the
    query tokens, so scores grow with the query length (roughly one point per query token for
    normalized embeddings), while MeanMaxSim divides that by the token count and sits in cosine's
    ``[-1, 1]`` range.

    Example:
        ::

            from datasets import load_dataset

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorTripletEvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

            evaluator = MultiVectorTripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all-nli-dev",
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(self, *args, token_pooling: BaseTokenPooling | None = None, **kwargs) -> None:
        self.token_pooling = token_pooling
        super().__init__(*args, **kwargs)

    def _get_similarity_functions(self) -> dict:
        # All supported multi-vector similarities. The parent picks via similarity_fn_names (default: the model's).
        from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

        functions = {}
        for name in MultiVectorEncoder.SUPPORTED_SIMILARITY_FN_NAMES:
            pairwise = SimilarityFunction.to_similarity_pairwise_fn(name)
            functions[name] = lambda a, p, n, pairwise=pairwise: (pairwise(a, p), pairwise(a, n))
        return functions

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | Sequence[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        **kwargs,
    ) -> list[Tensor]:
        kwargs.setdefault("truncate_dim", self.truncate_dim)
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        embeddings = encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            token_pooling=self.token_pooling,
            **kwargs,
        )
        return embeddings

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.token_pooling is not None:
            config_dict["token_pooling"] = pooling_config(self.token_pooling)
        return config_dict
