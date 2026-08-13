from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns_padded
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_scores
from sentence_transformers.util import (
    all_gather_padded,
    get_rank,
    similarity_fct_name,
    stack_padded_token_embeddings,
)


class MultiVectorMultipleNegativesRankingLoss(nn.Module):
    """In-batch negatives contrastive loss for :class:`~sentence_transformers.MultiVectorEncoder` models.

    For each query in the batch, the matched positive document is treated as the positive sample, and all
    other in-batch documents (plus any explicitly-provided hard negatives) serve as negatives. Scoring uses
    MaxSim by default. Pass a different ``similarity_fct`` (e.g.
    :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores`) to switch scoring strategies
    without changing the loss.

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder` model.
        scale: ``1 / temperature``. Scores are multiplied by ``scale`` before cross-entropy. Defaults to
            ``1.0`` (``temperature=1.0``), matching PyLate. Unlike cosine similarity (bounded to
            ``[-1, 1]``, where ST's dense :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss`
            uses ``scale=20.0`` to amplify the narrow range), MaxSim is an unbounded sum over query-token
            similarities (range ``~[0, num_query_tokens]``), so a scale near ``1.0`` makes sense, the same
            reason the dense loss recommends ``scale=1`` for dot-product similarity. With MeanMaxSim scoring
            (:func:`~sentence_transformers.multi_vector_encoder.scoring.mean_colbert_scores`), each score
            is divided by its query's token count, so a scale of roughly the average query length is a
            reasonable start.
        similarity_fct: Scoring callable. Receives queries ``(Q, q_tokens, dim)`` and stacked documents
            ``(Q, N, d_tokens, dim)`` and returns ``(Q, Q*N)`` with query-major ordering. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores`. Pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` for XTR-style scoring.
        mini_batch_size: Maximum number of rows per model forward during the **embedding** phase. The
            document columns are merged into one ``batch_size * N``-row batch and split into row chunks,
            each embedded at its own width, so a single long outlier document only widens its own chunk.
            Chunking is exact (no cross-row interactions), but every chunk's activations stay live for
            the backward pass, so this bounds padding waste rather than total memory: reach for
            :class:`CachedMultiVectorMultipleNegativesRankingLoss` to bound memory by chunk size.
            ``None`` runs the merged batch in one forward.
        score_mini_batch_size: If set, queries are processed in chunks of this size during the scoring
            phase. Useful to bound transient scoring memory for large effective batch sizes. Gradients
            still flow through a single backward. It chunks the query axis: to chunk the document axis
            as well, bind a budget into the scorer with
            ``similarity_fct=partial(colbert_scores, chunk_elements=...)``.
        gather_across_devices: If True, AllGather document embeddings (and masks) across DDP ranks so that
            every rank's queries see the global batch of documents. Useful for very large effective batches.

    Requirements:
        1. (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n) n-tuples

    Inputs:
        +-------------------------------------------------+--------+
        | Inputs                                          | Labels |
        +=================================================+========+
        | (anchor, positive) pairs                        | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative) triplets           | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative_1, ..., negative_n) | none   |
        +-------------------------------------------------+--------+

    Recommendations:
        - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.sentence_transformer.training_args.BatchSamplers>`)
          to ensure that no in-batch negatives are duplicates of the anchor or positive samples.

    Relations:
        - :class:`CachedMultiVectorMultipleNegativesRankingLoss` is equivalent to this loss, but it caches the
          embedding pass so that much higher batch sizes fit in the same memory, at the cost of being slower.
        - The dense counterpart is
          :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss`, which scores
          pooled embeddings instead of per-token MaxSim.

    Example:
        ::

            from datasets import Dataset

            from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
            from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss

            model = MultiVectorEncoder("answerdotai/ModernBERT-base")
            train_dataset = Dataset.from_dict(
                {
                    "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                    "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                    "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                }
            )
            loss = MultiVectorMultipleNegativesRankingLoss(model)

            trainer = MultiVectorEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
            trainer.train()
    """

    # Enables per-sample media counting in Transformer.preprocess, so mini-batching can slice VLM
    # inputs (e.g. Qwen2-VL's flattened pixel_values) along the batch dimension.
    requires_media_counts = True

    def __init__(
        self,
        model: MultiVectorEncoder,
        *,
        scale: float = 1.0,
        similarity_fct: Callable | None = None,
        mini_batch_size: int | None = None,
        score_mini_batch_size: int | None = None,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        if not scale > 0:
            raise ValueError("Scale must be a positive value.")
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct if similarity_fct is not None else colbert_scores
        self.mini_batch_size = mini_batch_size
        self.score_mini_batch_size = score_mini_batch_size
        self.gather_across_devices = gather_across_devices

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": similarity_fct_name(self.similarity_fct),
            "mini_batch_size": self.mini_batch_size,
            "score_mini_batch_size": self.score_mini_batch_size,
            "gather_across_devices": self.gather_across_devices,
        }

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        # The collator stamps each column's tokenization task (column 0 is the query unless
        # router_mapping overrides it). The MultiVectorMask module reads `task` from forward
        # kwargs and rewrites the output attention_mask into the per-row scoring mask.
        query_features = sentence_features[0]
        query_outputs = self.model(query_features, task=query_features.get("task", "query"))
        query_embeddings = query_outputs["token_embeddings"]
        q_mask = query_outputs["attention_mask"].bool()

        batch_size = query_embeddings.size(0)
        N = len(sentence_features) - 1

        doc_embeddings, doc_masks = embed_columns_padded(
            self.model, sentence_features[1:], self.mini_batch_size, task_default="document"
        )

        if self.gather_across_devices:
            # Gather doc embeddings + masks across ranks so every rank's queries see the global doc
            # batch. Pad the token axis to the cross-rank max first: each rank pads its columns to its
            # own batch-longest, so T differs per rank and all_gather needs a uniform shape per rank.
            gathered = [all_gather_padded(e, m, with_grad=True) for e, m in zip(doc_embeddings, doc_masks)]
            doc_embeddings = [e for e, _ in gathered]
            doc_masks = [m for _, m in gathered]

        docs_stacked, docs_mask_stacked = stack_padded_token_embeddings(doc_embeddings, doc_masks)

        step = self.score_mini_batch_size or batch_size
        score_chunks = []
        for begin in range(0, batch_size, step):
            end = begin + step
            score_chunks.append(
                self.similarity_fct(
                    query_embeddings[begin:end],
                    docs_stacked,
                    queries_mask=q_mask[begin:end],
                    documents_mask=docs_mask_stacked,
                )
            )
        scores = torch.cat(score_chunks, dim=0)

        # Query-major layout: positive for query i is at column i*N (cf. colbert_scores).
        labels = torch.arange(batch_size, device=query_embeddings.device) * N
        if self.gather_across_devices:
            labels = labels + get_rank() * batch_size * N

        return F.cross_entropy(scores * self.scale, labels)

    @property
    def citation(self) -> str:
        return """
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""
